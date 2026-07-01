#version 430 core


layout(location=0) in vec2 a_VertexPosition;

void main()
{
	gl_Position = vec4(a_VertexPosition, 0.0f, 1.0f);
}