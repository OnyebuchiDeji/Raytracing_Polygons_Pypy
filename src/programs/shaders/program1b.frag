#version 430 core

/**
Program1b.frag

Different from `program1.frag` in that it implements
model transformations of the 3D model using the inverse
of its matrix.
*/

// Screen and Other uniforms
uniform vec2 screenResolution;
uniform int primitiveShapeCount;

// Camera uniforms (set from CPU)
uniform vec3 camPos, camDir, camUp, camRight;
uniform float camFovTangent;

// Inverse Model Matrix
uniform mat4 invModelMatrix;
uniform mat3 normalModelMatrix;

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

	// Second Version:
	// Proper scaling
	// Allows zoom-in and zoom-out
	vec3 rayDir = normalize(camDir +
		uv.x * aspect_ratio * camFovTangent * camRight +
		uv.y * camFovTangent * camUp
	);

	/**
		For Object Transformations using the Model Matrix

		Best/Fastest Option: Ray Transformation
		Transform the world-space camera's ray into the 
		OBJECT's LOCAL SPACE before running the intersection loop INSTEAD of transforming each of the mesh vertices into world space (applying the model transformation on each)

		This involves the INVERSE MODEL MATRIX
	*/
	vec3 rayOrigin = camPos;

	//	use `localOrigin`
	vec3 localOrigin = (invModelMatrix * vec4(rayOrigin, 1.0)).xyz;
	vec3 localDir = normalize((invModelMatrix * vec4(rayDir, 0.0)).xyz); // w=0 ignores must be so as translation because it's not needed for the local direction calculation

	float closestT = 1e30;
	vec3 hitNormalLocal = vec3(0.0); //	basically, hit point

	int triCount = primitiveShapeCount;

	for (int i = 0; i < triCount; ++i) {
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


	if (closestT < 1e29) {
		//	Multiply by normal/inverse transpose model matrix and normalize to fix scaling distortions
		vec3 hitNormalWorld = normalize(normalModelMatrix * hitNormalLocal);

		//	Simple shading: normal color
		gl_FragColor = vec4(0.5 * (hitNormalWorld + vec3(1.0)), 1.0);
	}else {
		gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); // background
	}
	

}