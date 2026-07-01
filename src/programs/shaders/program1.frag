#version 430 core

/**
Program1.frag

The first implementation.
Performs only Ray-Triangle intersections
and cannot perform model transformations.
*/

// Screen and Other uniforms
uniform vec2 screenResolution;
uniform int primitiveShapeCount;
//uniform mat4 viewMat;

// Camera uniforms (set from CPU)
uniform vec3 camPos, camDir, camUp, camRight;
uniform float camFovTangent;


// Scene Data (either from textures or SSBOs)

// Option A: Utiling Shader Storage Buffer Objects
layout(std430, binding=0) buffer Indices { int indices[]; };
layout(std430, binding=1) buffer Vertices { vec4 vertices[]; };

// Option B: using textures (comment out SSBOs above to use this):
//layout(binding=0) uniform sampler2D indexTex1D;
// uniform samplerBuffer vertexTexBuf;
//uniform samplerBuffer indexTexBuf;


// Möller-Trumbore ray-triangle intersection
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


void main()
{
	//	Compute ray direction for this fragment (ensure uv ranges [0, 1])
	//	or pass uv via vertex shader instead of gl_FragCoord if fullscreen
	vec2 uv = gl_FragCoord.xy / screenResolution;
	uv = uv * 2.0 - 1.0; // uv ndx <- map to [-1, 1] range

	float aspect_ratio = screenResolution.x/screenResolution.y;

	// First Version: Assumes that FOV stays the same always
	// Without aspect Scaling, the cube appears as a cuboid
	//vec3 rayDir = normalize(camDir + (uv.x - 0.5) * camRight + (uv.y - 0.5) * camUp);

	// Second Version:
	// Proper scaling
	// Allows zoom-in and zoom-out
	vec3 rayDir = normalize(camDir +
		uv.x * aspect_ratio * camFovTangent * camRight +
		uv.y * camFovTangent * camUp
	);

	vec3 rayOrigin = camPos;

	float closestT = 1e30;//1e-2;
	vec3 hitNormal = vec3(0.0); //	basically, hit point

	int triCount = primitiveShapeCount;

	for (int i = 0; i < triCount; ++i) {
		//	Fetch the triangle vertex indices and then positions
		int i3 = i*3;
		int i0 = indices[i3 + 0];
		int i1 = indices[i3 + 1];
		int i2 = indices[i3 + 2];
		// IT IS NOT NEEDED TO TRANSFORM VERTICES WITH VIEW MATRIX.
		/**
			RayTracing is fast because it relies on transforming 
			the single camera ray INTO the world instead of
			transforming thousands of mesh vertices using the view matrix every single frame
			Because I'm already constructing my rayDir using the
			camera's world-space orientation vectors (camDir,
			camRight, camUp) and setting 'rayOrig=camPos', my rays
			are FIRING DIRECTLY INTO WORLD SPACE.
			Hence, the intersection loop is perfectly balanced
			because the rays and the vertices are already using the same coordinate.
			//vec3 v0 = (viewMat * vec4(vertices[i0].xyz, 1.0)).xyz;
			//vec3 v1 = (viewMat * vec4(vertices[i1].xyz, 1.0)).xyz;
			//vec3 v2 = (viewMat * vec4(vertices[i2].xyz, 1.0)).xyz;
		*/
		vec3 v0 = vertices[i0].xyz;
		vec3 v1 = vertices[i1].xyz;
		vec3 v2 = vertices[i2].xyz;
		float t;
		vec3 bary;
		if (intersectRayTriangle(rayOrigin, rayDir, v0, v1,v2, t, bary)) {
			if (t < closestT) {
				closestT = t;
				hitNormal = normalize(cross(v1 - v0, v2 - v0));
			}
		}
	}

	//if (closestT < 1e29) {
	//gl_FragColor = vec4(closestT*10, 0.0, 0.0, 1.0);

	if (closestT < 1e29) {
		//	Simple shading: normal color
		gl_FragColor = vec4(0.5 * (hitNormal + vec3(1.0)), 1.0);
		//gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);
	}else {
		gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); // background
	}
	

}