#version 430 core

/**
Program2.frag

Can perform both triangle and quad intersections.

Implements the Global Bounding Box Test (GBBT).

But the GBBT is not very optimized with the ray tracing
algorithm.

Thus, Program2b.frag improves on it. 
*/

uniform vec3 camPos, camDir, camUp, camRight;
uniform float camFovTangent;
uniform vec2 screenResolution;

uniform bool isRayTriIntersect;
uniform int primitiveShapeCount;

uniform vec3 meshAABBMin;
uniform vec3 meshAABBMax;

uniform mat4 invModelMatrix;
uniform mat3 normalModelMatrix;


// SSBOs
// Tri/Quad Indices: May use 3 or 4
layout(std430, binding=0) buffer Indices { int indices[]; };
// Tri/Quad data structures (std430 padded to vec4 algnment)
layout(std430, binding=1) buffer Vertices { vec4 vertices[]; };


// Fast AABB Ray Intersection Test
// This is the Slab Intersection Method
bool intersectAABB(
	vec3 origin, vec3 dir,
	vec3 boxMin, vec3 boxMax
) {
	// clamp very small directions to prevent
	// cases where invDir could become NaN
	// such as when origin.x == boxMin.x
	const float EPS = 1e-8;
	vec3 safeDir = sign(dir) * max(abs(dir), vec3(EPS));
	vec3 invDir = 1.0 / safeDir;

	// find the t value along ray direction for the minimum limits of meshbounds from the origin
	vec3 t0 = (boxMin - origin) * invDir;
	// find the same for the maximum limits
	vec3 t1 = (boxMax - origin) * invDir;
	vec3 tmin = min(t0, t1);
	vec3 tmax = max(t0, t1);
	float tNear = max(max(tmin.x, tmin.y), tmin.z);
	float tFar = min(min(tmax.x, tmax.y), tmax.z);
	// 
	return tNear <= tFar && tFar > 0.0;
}


// Möller-Trumbore Ray-Triangle Intersection
bool intersectRayTriangle(
	vec3 origin, vec3 dir,
	vec3 v0, vec3 v1, vec3 v2,
	out float t, out vec3 bary
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
	bary = vec3(u, v, 1.0 - u - v);
	return true;
}


// Direct Ray-Quad Intersection Test (by Lagae and Dutré)
// Expects vertices ordered counter-clockwise: v0, v1,v2, v3
bool intersectRayQuad(
	vec3 origin, vec3 dir,
	vec3 v0, vec3 v1, vec3 v2, vec3 v3,
	out float t, out vec2 uvCoords
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

	float closestT = 1e30;
	vec3 hitNormalLocal = vec3(0.0);

	// Quad Intersection Loop
	// Important: quadCount = total indices / 4
	int primitiveCount = primitiveShapeCount;

	if (intersectAABB(localOrigin, localDir, meshAABBMin, meshAABBMax)) {
		if (isRayTriIntersect) {
			for (int i = 0; i < primitiveCount; ++i) {
				//	Fetch the triangle vertex indices and then positions
				int i3 = i*3;
				int i0 = indices[i3 + 0];
				int i1 = indices[i3 + 1];
				int i2 = indices[i3 + 2];
				vec3 v0 = vertices[i0].xyz;
				vec3 v1 = vertices[i1].xyz;
				vec3 v2 = vertices[i2].xyz;
				float t;
				vec3 bary;
				if (intersectRayTriangle(localOrigin, localDir, v0, v1,v2, t, bary)) {
					if (t < closestT) {
						closestT = t;
						hitNormalLocal = normalize(cross(v1 - v0, v2 - v0));
					}
				}
			}

		} else {
			for(int i = 0; i < primitiveCount; ++i) {
				// Fetch 4 indices per primitive shape instead of 3
				int i4 = i * 4;
				int i0 = indices[i4 + 0];
				int i1 = indices[i4 + 1];
				int i2 = indices[i4 + 2];
				int i3 = indices[i4 + 3];

				vec3 v0 = vertices[i0].xyz;
				vec3 v1 = vertices[i1].xyz;
				vec3 v2 = vertices[i2].xyz;
				vec3 v3 = vertices[i3].xyz;

				float t;
				vec2 quadUV;

				if (intersectRayQuad(localOrigin, localDir, v0, v1, v2, v3, t, quadUV)) {
					if (t < closestT) {
						closestT = t;
						// Calculate local normal using two edges of the quad
						hitNormalLocal = cross(v1 - v0, v3 - v0);
					}
				}
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