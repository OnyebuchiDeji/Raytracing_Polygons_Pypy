#version 430 core

/**
Program3.frag

Improves upon Program2.frag

Implements the Bounding Volume Hierarchy object
for hierarchical spatial partitioning.

But it was scrapped due to complexities using Quads.

This, Program4.frag improves on it by using only Triangles
for all meshes (by splitting meshes with quad faces
into triangles).
*/

uniform vec3 camPos, camDir, camUp, camRight;
uniform float camFovTangent;
uniform vec2 screenResolution;

uniform bool isRayTriIntersect;

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
layout(std430, binding=6) buffer BVHFirst { int firstPrim[]; };
layout(std430, binding=7) buffer BVHCount { int primCount[]; };


// Fast AABB Ray Intersection Test
// This is the Slab Intersection Method
bool intersectAABB(
	vec3 origin, vec3 invDir,
	vec3 boxMin, vec3 boxMax,
	out float tNear, out float tFar
) {
	// find the t value along ray direction for the minimum limits of meshbounds from the origin
	vec3 t0 = (boxMin - origin) * invDir;
	// find the same for the maximum limits
	vec3 t1 = (boxMax - origin) * invDir;

	vec3 tmin = min(t0, t1);
	vec3 tmax = max(t0, t1);

	tNear = max(max(tmin.x, tmin.y), tmin.z);
	tFar = min(min(tmax.x, tmax.y), tmax.z);
	
	return (tNear <= tFar) && (tFar >= 0.0);
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


// Direct Ray-Quad Intersection Test (by Lagae and Dutré)
// Expects vertices ordered counter-clockwise: v0, v1,v2, v3
bool intersectRayQuad(
	vec3 origin, vec3 dir,
	vec3 v0, vec3 v1, vec3 v2, vec3 v3,
	float maxT, out float t,
	out vec2 uvCoords
) {
	const float EPS = 1e-6;

	// Calculate the plane of the first triangle (v0, v1, v3) to find the flat surface
	vec3 e10 = v1 - v0;
	vec3 e30 = v3 - v0;
	vec3 p = cross(dir, e30);
	float det = dot(e10, p);

	// If nearly parallel to the first triangle plane, check the alternate plane (v2)
	if (abs(det) < EPS) {
		vec3 e23 = v2 - v3;
		vec3 e13 = v1 - v3;
		p = cross(dir, e13);
		det = dot(e23, p);
		if (abs(det) < EPS) return false; // parallel to the entire quad
	}

	float invDet = 1.0 / det;
	vec3 tvec = origin - v0;

	// Calculate bilinear parameters to see if the ray hit inside the 4 boundaries
	float u = dot(tvec, p) * invDet;
	if (u < 0.0 || u > 1.0) return false;

	vec3 q = cross(tvec, e10);
	float v = dot(dir, q) * invDet;
	if (v < 0.0 || v > 1.0) return false;

	// Calculate hit distance 't'
	t = dot(e30, q) * invDet;
	if (t <= EPS) return false;

	// For non-planar quads, verify that the point lies inside the second triangle boundary
	if (u + v > 1.0) {
		vec3 e32 = v3 - v2;
		vec3 e12 = v1 - v2;
		vec3 tvec2 = origin - v2;
		p = cross(dir, e12);
		det = dot(e32, p);
		invDet = 1.0 / det;
		float u2 = dot(tvec2, p) * invDet;
		float v2_bary = dot(dir, cross(tvec2, e32)) * invDet;
		if (u2 < 0.0 || v2_bary < 0.0 || u2 + v2_bary > 1.0) return false;
	}

	uvCoords = vec2(u, v); // Out local quad coordinates;
	return true; 
}

/**
The code below uses a fixed-size `stack[64]` of node indices
(size must accommodate depth; 64 is usually enough for ~million triangles).
First, the ray intersects with the node's AABB; if it misses, skip the subtree. If leaf (primCount>0), check each primitive (via intersectRayTriangle or intersectRayQuad).
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

	// Primitive Intersection Loop
	// Important: quadCount = total indices / 4
	//int primitiveCount = primitiveShapeCount;

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

		int cnt = primCount[nodeIdx];

		if (cnt > 0) {
			if (isRayTriIntersect) {
				// Leaf: test each primitive/triangle in [firstPrim, firstPrim + cnt]
				for (int i = 0; i < cnt; ++i) {
					//	Fetch the triangle vertex indices and then positions
					int triId = firstPrim[nodeIdx] + i;
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
				// Leaf: test each primitive/quad in [firstPrim, firstPrim+cnt]
				for(int i = 0; i < cnt; ++i) {
					int quadId = firstPrim[nodeIdx] + i;
					// Fetch 4 indices per primitive shape instead of 3
					int i4 = quadId*4;
					int i0 = indices[i4 + 0];
					int i1 = indices[i4 + 1];
					int i2 = indices[i4 + 2];
					int i3 = indices[i4 + 3];

					vec3 v0 = vertices[i0].xyz;
					vec3 v1 = vertices[i1].xyz;
					vec3 v2 = vertices[i2].xyz;
					vec3 v3 = vertices[i3].xyz;

					// new start t value
					float t = max(tNear, 0.0);
					vec2 quadUV;

					if (intersectRayQuad(localOrigin, localDir, v0, v1, v2, v3, tFar, t, quadUV)) {
						if (t < closestT) {
							closestT = t;
							// Calculate local normal using two edges of the quad
							hitNormalLocal = cross(v1 - v0, v3 - v0);
						}
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