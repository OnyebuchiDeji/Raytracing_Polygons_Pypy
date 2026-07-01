#version 430 core

/**
Program4.frag

Improves upon Program3.frag
Implements the Bounding Volume Hierarchy object
for hierarchical spatial partitioning.

It reflects using the BVH for only triangle meshes,
But the BVH algoritm had several issues.

These issues are fixed in Program4b.frag
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
layout(std430, binding=2) buffer BVHMin { vec3 bmin[]; };
layout(std430, binding=3) buffer BVHMax { vec3 bmax[]; };
layout(std430, binding=4) buffer BVHLeft  { int leftChild[]; };
layout(std430, binding=5) buffer BVHRight { int rightChild[]; };
layout(std430, binding=6) buffer BVHFirst { int firstTri[]; };
layout(std430, binding=7) buffer BVHCount { int triCount[]; };


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
	
	return (tNear <= tFar);// && (tFar >= 0.0);
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


/**
The code below uses a fixed-size `stack[64]` of node indices
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

	float tNear, tFar;

	// clamp very small directions to prevent
	// cases where invDir could become NaN
	// such as when origin.x == boxMin.x
	const float EPS = 1e-8;
	vec3 safeDir = sign(localDir) * max(abs(localDir), vec3(EPS));
	vec3 invDir = 1.0 / safeDir;

	int stack[64];
	int sp = 0;
	stack[sp++] = 0; // root idx

	float closestT = 1e30;
	vec3 hitNormalLocal = vec3(0.0);


	while (sp > 0) {
		int nodeIdx = stack[--sp];
		if (nodeIdx < 0) continue;

		//	AABB test
		if (!intersectAABB(localOrigin, invDir, bmin[nodeIdx], bmax[nodeIdx], tNear, tFar)) {
			continue;
		}

		int cnt = triCount[nodeIdx];

		if (cnt > 0) {
			// Leaf: test each triangle in [firstTri, firstTri + cnt]
			for (int i = 0; i < cnt; ++i) {
				//	Fetch the triangle vertex indices and then positions
				int triId = firstTri[nodeIdx] + i;
				int i3 = triId*3;
				int i0 = indices[i3 + 0];
				int i1 = indices[i3 + 1];
				int i2 = indices[i3 + 2];
				vec3 v0 = vertices[i0].xyz;
				vec3 v1 = vertices[i1].xyz;
				vec3 v2 = vertices[i2].xyz;

				// new start t value
				float t = max(tNear, 0.0);
				vec3 bary;
				// begin march at start
				if (intersectRayTriangle(localOrigin, localDir, v0, v1,v2, tFar, t, bary)) {
					if (t < closestT) {
						closestT = t;
						hitNormalLocal = normalize(cross(v1 - v0, v2 - v0));
					}
				}
			}

		} else {
			//	Interior: push children
			stack[sp++] = rightChild[nodeIdx];
			stack[sp++] = leftChild[nodeIdx];
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