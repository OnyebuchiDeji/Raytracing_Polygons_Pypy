#version 430 core

/**
Program5.frag

Improves upon Program4.frag and Program4b.frag
Implements the Bounding Volume Hierarchy object
for hierarchical spatial partitioning.

1) `bmin` and `bmax` should be `vec4` not `vec3` dues to
the enforcement of std430. Issue was the CPU data that was
sent was 12 bytes (for vec3) and had no padding. But std430
interpreted vec3 as vec4, causing memory misalignment and
thoroughly making the bounds checks with bmin and bmax fail.
Solution1/Fix1: pad the data from CPU with 4 bytes, change the SSBO interpretation to be `vec4`, and anywhere using `bmin` or `bmax` should utilize the swizzle to access the data.
	So, `bmin` data should be accessed using `bmin.xyz`
	So, `bmax` data should be accessed using `bmax.xyz`

2) Check whether `sp < MAX_STACK` before pushing. The BVH build uses a MIDPOINT SPLIT (not SAG, not a median split), with a very fine leaf threshold of `count <= 2` (implemented in the CPU-side).
Midpoint splitting can degrade badly on non-uniform/clustered geometry (e.g. a dense detail area next to a sparse flat area), thereby producing local tree depth well beyond O(log N).

If `sp` ever reaches 128, `stack[sp++] = ...` writes out of bounds of a local array. This could cause undefined behaviour on the GPU (commonly silently clamped/wrapped by the compiler), which corrupts other pending stack entries and causes whoel subtrees to simply never be visited.

This could be another cause of the "missing polygons", typically concentrated in geomertrically dense regions of the mesh.

Solution2/Fix2: `pushNode` function


3) Issues with artefacts called "cracks" or watertightness failures, a common bug of BVH ray tracers.
These cracks show up as SPARSE SINGLE-PIXEL or FEW-PIXEL holes
concentrated at:
- Silhouette/grazing angles on a triangle,
- Boundaries between adjacent BVH leaf boxes,
- thin/elongated triangles

Root cause: the `maxT = min(tFar, closestT)` clamp is too tight
The ray exists the node's box at `tFar`. So, no hit inside that node box's triangles can have `t > tFar`.
Mathematically, that's true since a leaf's AABB is built directly from its traingles' vertice (`vs.min(axis = 0)/vs.max(axis=0)` in `build_node` on the CPU-side), and thus any hit point on one of its triangles must be geometrically inside that box, so its true `t` must lie within `[tNear, tFar]`.

The problem is these two `t` values are computed by two completely different formulas with different rounding behavior:
	- `tFar` comes from the slab test: `(boxMax - origin) * invDir`, then a `min()` reduction across axes.
	- The triangle's `t` comes from Möller–Trumbore:
		`dot(e2, q) * invDet`

These do not ROUND the same way. so for a hit point that lies VERY CLOSE to a boc face (which is extremely commmon --- it happens at every shared edge between two adjavent leaves, and at every grazing-angle triangle), it's routine for the Möller–Trumbore `t` to come out a handful of ULPs (Unit in the Last Place/Unit of Least Precision) LARGER than the slab-test `tFar`, even though the true hit is inside the box.
Then this line kills it:
	```
	if (t > maxT) return false; // rejects a geometrically valid hit
	```
Thus, if that triangle only lives in this one lead (which is normal as no clipping/duplication is done at build time), the ray gets no hit at all for that `pixel -> background color -> a hole`

Solution3/Fix3: stop using `tFar` as an upper bound for the triangle test.
`tFar` is not needed for correctness here at all. It was a redundant micro-optimization, since `closestT` already bounds how far one needs to search, and the AABB test that identifies the lead already guarantees these triangles are relevant.
Since `intersectRayTriangle` is only called after passing the Box test, dropping `tFar` costs essentially nothing in extra triangle tests while eliminatiing the crack/error source:
```
// Do NOT further calmp by tFar. tFar is the *box's* exit distance, computed via the slab test, and can differ from a triangle's true intersection distance by a few ULPs due to a different rounding path used (Möller-Trumbore vs. slab-test division).
// Since this leaf's AABB was built to fully contain every triangle in it, closestT alone is sufficient and *safe* upper bound --- clamping by `tFar` only risks FALSE-REJECTING valid hits right at leaf boundaries, producing the PIXEL-SIZED holes ("cracks") in the render.
float maxT = closestT;

vec3 bary;
if (intersectRayTriangle(localOrigin, localDir, v0, v1, v2, maxT, t, bary)) {
    if (t < closestT) {
        closestT = t;
        hitNormalLocal = cross(v1 - v0, v2 - v0);
    }
}
```

Side Note: Remove the `float t = max(tNear, 0.0);` as its dead code because `t` is an out parameter of `intersectRayTriangle`, so whatever is assigned to it beforehand is discarded already. So no need for the `float t = max(tNear, 0.0);`. It should be deleted for clarity.

4) Solution/Fix4: Defense in depth; pad leaf AaBBs at build time:
Even with fix3, one can still get the single-pixel cracks from `box-vs-box` precision at the AABB test level itself due to a ray that should enter a sibling box numerically just missing it by an ULP.
The standard ROBUSTNESS technique is to FATTEN EACH NODE's AABB A TINY EPSILON when building the tree. Hence, the boxes overlap very slightly instead of touching exactly.
This is implemented in BVH5 on the CPU-side.

5) Solution/Fix5: The parallel/degenerate epsilon in Möller–Trumbore:
```
const float EPS = 1e-6;
...
if (abs(det) < EPS) return False; // parallel or degenerate
```
The line of code rejects any triangle whose determinant (roughtly, how edge-on/parallel the triangle is to the ray) falls under a fixed threshold.
For thin/elongated triangles viewed near-grazing, `det` can dip under `1e-6` for individual pixels even when the triangle is legitimately hit, causing a sparse "sparkle" of missing pixels along silhouettes.
This is a much smaller contributor than Fix 3 for the "cracks" symptom (it tends to look like fine dithering along edges rather than solid tiny holes), so this should REALLY ONLY BE IMPLEMENTED IF THE HOLES PERSIST after FIX 3/4:
```
// Only shrink this if there are still sparkle/holes on thin
// or near-grazing triangles after fixing the `tFar` clamp above.
// Shrinking it too far risks numerical instability
// (invDet blowing up) rather than fixing anything; so change it in small steps
```
*/

uniform vec3 camPos, camDir, camUp, camRight;
uniform float camFovTangent;
uniform vec2 screenResolution;

uniform mat4 invModelMatrix;
uniform mat3 normalModelMatrix;


// SSBOs
// Tri/Quad Indices: May use 3 or 4
layout(std430, binding=0) buffer Indices { int indices[]; };
// Tri/Quad data structures (std430 padded to vec4 algnment)
layout(std430, binding=1) buffer Vertices { vec4 vertices[]; };


// BVH data (using SSBOs)
layout(std430, binding=2) buffer BVHMin { vec4 bmin[]; };
layout(std430, binding=3) buffer BVHMax { vec4 bmax[]; };
layout(std430, binding=4) buffer BVHLeft  { int leftChild[]; };
layout(std430, binding=5) buffer BVHRight { int rightChild[]; };
layout(std430, binding=6) buffer BVHFirst { int firstTri[]; };
layout(std430, binding=7) buffer BVHCount { int triCount[]; };
layout(std430, binding=8) buffer TriPermutation { int triPermutation[]; };

const int MAX_STACK=128;

// Fast AABB Ray Intersection Test
// This is the Slab Intersection Method
bool intersectAABB(
	vec3 origin, vec3 invDir,
	vec3 boxMin, vec3 boxMax,
	out float tNear, out float tFar
) {
	vec3 t1 = (boxMin - origin) * invDir;
	vec3 t2 = (boxMax - origin) * invDir;

	vec3 tmin = min(t1, t2);
	vec3 tmax = max(t1, t2);

	tNear = max(max(tmin.x, tmin.y), tmin.z);
	tFar = min(min(tmax.x, tmax.y), tmax.z);
	
	// tFar > 0.0 removes nodes completely behind the ray.
	return (tNear <= tFar) && (tFar > 0.0);
}


// Möller-Trumbore Ray-Triangle Intersection
bool intersectRayTriangle(
	vec3 origin, vec3 dir,
	vec3 v0, vec3 v1, vec3 v2,
	float maxT, out float t,
	out vec3 bary
) {
	const float EPS = 1e-6;
	vec3 e1 = v1 - v0;
	vec3 e2 = v2 - v0;
	vec3 p = cross(dir, e2);
	float det = dot(e1, p);
	if (abs(det) < EPS) return false; // parallel or degenerate
	float invDet = 1.0 / det;
	vec3 tvec = origin - v0;
	float u = dot(tvec, p) * invDet;
	if (u < 0.0 || u > 1.0) return false;
	vec3 q = cross(tvec, e1);
	float v = dot(dir, q) * invDet;
	if (v < 0.0 || u + v > 1.0) return false;
	t = dot(e2, q) * invDet;
	if (t <= EPS) return false;
	if (t > maxT) return false;
	bary = vec3(u, v, 1.0 - u - v);
	return true;
}


void pushNode(
	inout int stack[MAX_STACK],
	inout float stackNear[MAX_STACK],
	inout int sp, int nodeIdx, float near
) {
	if (sp < MAX_STACK) {
		stack[sp] = nodeIdx;
		stackNear[sp] = near;
		sp++;
	}
	// else: dropped -- dropping should not happen if MAX_STACK
	// is sized to the real tree depth; consider tracking
	// max build-time depth on the CPU (self.max_depth) and
	// asserting MAX_STACK > max_depth + 1, or increasing MAX_STACK.
}

/**
The code below uses a fixed-size `stack[MAX_STACK]` of node indices
(size must accommodate depth; 64 is usually enough for ~million triangles).
First, the ray intersects with the node's AABB; if it misses, skip the subtree. If leaf (triCount>0), check each triangle (via intersectRayTriangle).
Otherwise (interior), push its children to the stack.
This drastically cuts down the number of triangle tests. Each node's AABB is stored in bmin(nodeIdx) / bmax[nodeIdx].
*/

void main() {
	// Setup normalized camera coordinates
	vec2 uv = gl_FragCoord.xy / screenResolution;
	vec2 uv_ndc = (uv - 0.5) * 2;
	float aspect = screenResolution.x / screenResolution.y;

	// Generate world space ray
	vec3 rayDir = normalize(camDir +
		uv_ndc.x * aspect * camFovTangent * camRight +
		uv_ndc.y * camFovTangent * camUp
	);

	vec3 rayOrigin = camPos;

	// Transform ray to object;s local space
	vec3 localOrigin = (invModelMatrix * vec4(rayOrigin, 1.0)).xyz;
	vec3 localDir = normalize((invModelMatrix * vec4(rayDir, 0.0)).xyz);

	// Modified by 'intersectAABB'
	float tNear, tFar;

	// Used to determine node traversal
	// based on which node is closer between
	// left or right
	int left;
	int right;
	float leftNear, leftFar;
	float rightNear, rightFar;
	bool hitLeft = false;
	bool hitRight = false;

	// clamp very small directions to prevent
	// cases where invDir could become NaN
	// such as when origin.x == boxMin.x
	const float EPS = 1e-8;
	vec3 safeDir = sign(localDir) * max(abs(localDir), vec3(EPS));
	vec3 invDir = 1.0 / safeDir;

	int stack[MAX_STACK];
	float stackNear[MAX_STACK];

	int sp = 0;
	stack[sp++] = 0; // root idx

	float closestT = 1e30;
	vec3 hitNormalLocal = vec3(0.0);


	while (sp > 0) {
		int nodeIdx = stack[--sp];
		tNear = stackNear[sp];
		if (nodeIdx < 0) continue;

		/**
		AABB test
		Why `|| tNear > closestT`?
		if closest hit = t = 4, and another node has tNear = 9, this other node can never contain a closer triangle.
		So, prevent the shader from continuing by skipping to next iteration
		*/

		if (!intersectAABB(localOrigin, invDir, bmin[nodeIdx].xyz, bmax[nodeIdx].xyz, tNear, tFar) || tNear > closestT) {
			continue;
		}

		// new start t value
		float t = max(tNear, 0.0);

		int cnt = triCount[nodeIdx];

		if (cnt > 0) {
			// Leaf: test each triangle in [firstTri, firstTri + cnt]
			for (int i = 0; i < cnt; ++i) {
				//	Fetch the triangle vertex indices and then positions
				// This is essential for the BVH to properly work.
				int triId = triPermutation[firstTri[nodeIdx] + i];
				int i3 = triId*3;
				int i0 = indices[i3 + 0];
				int i1 = indices[i3 + 1];
				int i2 = indices[i3 + 2];
				vec3 v0 = vertices[i0].xyz;
				vec3 v1 = vertices[i1].xyz;
				vec3 v2 = vertices[i2].xyz;

				// Correct way of implementing maxT
				// so that if already hit triangle at t = 2
				// so closesT = 2. Then,
				// maxT updates to t = 2 thereby rejecting
				// any other triangle much earlier
				//float maxT = min(tFar, closestT);
				// Better way of implementing maxT since node AABBs help make effect of min(tFar, closestT) intrinsic
				float maxT = closestT;

				vec3 bary;

				// begin march at start and ensure to end at maxT
				if (intersectRayTriangle(localOrigin, localDir, v0, v1,v2, maxT, t, bary)) {
					if (t < closestT) {
						closestT = t;
						// do not normalize here. Only normalize once after traversal in '//Shading and Output' below
						hitNormalLocal = cross(v1 - v0, v2 - v0);
					}
				}
			}

		} else {
			//	reset before testing children.
			hitLeft = false;
			hitRight = false;

			left = leftChild[nodeIdx];
			right = rightChild[nodeIdx];

			if (left >= 0) {
				hitLeft = intersectAABB(
					localOrigin, invDir,
					bmin[left].xyz, bmax[left].xyz,
					leftNear, leftFar
				);
				hitLeft = hitLeft && (leftNear <= closestT);
			}

			if (right >= 0) {
				hitRight = intersectAABB(
					localOrigin, invDir,
					bmin[right].xyz, bmax[right].xyz,
					rightNear, rightFar
				);
				hitRight = hitRight && (rightNear <= closestT);
			}

			if (!hitLeft && !hitRight) { continue; }

			// Both Hit

			/**
			Fixing traversal order to visit the nearest child first
			Interior: push children
			Due to being a Last In, First Out (LIFO), the Last Node pushed in using `stack[sp++]` is the first node
			popped and traversed.

			So the below implementation is correct.
			*/
			if (hitLeft && hitRight)
			{
				if (leftNear < rightNear)
				{
					pushNode( stack, stackNear,
						sp, right, rightNear
					);
					
					pushNode( stack, stackNear,
						sp, left, leftNear
					);
				}
				else {
					pushNode( stack, stackNear,
						sp, left, leftNear
					);
					pushNode( stack, stackNear,
						sp, right, rightNear
					);
				}
			}
			else if (hitLeft) {
				pushNode( stack, stackNear,
					sp, left, leftNear
				);
			}
			else if (hitRight) { 
				pushNode( stack, stackNear,
					sp, right, rightNear
				);
			}

		}
	
	}

	// Shading and Output
	if (closestT < 1e29) {
		// Transform normal to World Space using the Normal Matrix
		vec3 hitNormalWorld = normalize(normalModelMatrix * hitNormalLocal);

		// Simple normal visualization shading
		gl_FragColor = vec4(0.5 * (hitNormalWorld + vec3(1.0)), 1.0);
	} else {
		gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
	}
}