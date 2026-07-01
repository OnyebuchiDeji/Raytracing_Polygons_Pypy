"""
Program 1.py

Successfully demonstrates brute-force linear scan raytracing,
performing a linear scan (an O(N) operation per pixel).

Implemented Ray-Triangle Intersection Test in shader (by Möller-Trumbore)
Implemented Ray-Quad intersection Test (by Lagai and Dutré)
Though I could split the quads into triangles in the Mesh initialization
and just use Ray-Triangle Intersection throughout, I'd much rather learn
the how to perform ray-quad intersection.


Due to this, for a small cube or smaller models such as those
uncommented in the demonstrations, the shader runs smoothly.

But for a bigger model, such as `deino.obj`, `baryonx.obj`, and `tank.obj`,
the app stalls because a GPU Timeout occurs.

This is known as the GPU Timeout Detection and Recovery (TDR)
It's because for my screen resolution of 1200x674 (810,000 pixels),
multiplied by the number of quads in `deino.obj` (47,712 quads), my GPU
is forced to compute '38,646,720,000' intersection tests per frame.

Now when a sigle draw call takes longer than a few seconds (usually 2 seconds on Windows),
the OS assumes the graphics driver has frozen. It forcefully restarts
the graphics driver to unfreeze the desktop, causing this Program1's app
to stall for a few moments after which the app resumes but the Python-Moderngl Window
stops updating/hangs completely even while background frames appear to loop

SOLUTION:
	1. Program2.py demonstrates solution using a Global Bounding Box Test (a quick fix)
	2. Program3.py demonstrates a solution using Bounding Volume Hierarchy object

"""

from engine import Engine, pg
from programs._program import Program
from utils import (
	data_to_flat_np_array, read_file,
	upload_flat_data_as_ssbo,
	set_uniform, set_complex_uniform
)
from config import SHADERS_PATH
from model_reader import Mesh
from camera import FPSCamera, glm



class Program1(Program):
	def __init__(self):
		super().__init__(
			"Program 1: Rendering Polygon Mesh with SDF Raytracing",
			(	f"Using ordinary raytracing to render a polygon mesh "
				f"on the GPU. No optimizations using BVH."
			),
			engine=Engine(
				program=self, winWidth=1200, winHeight=675
			),
			demonstrations=[
				{
					'subtitle': "Raytracing 3D Model - No Model Control",
					'vert': "program1.vert",
					'frag': "program1.frag",
					'model': "cube.obj"
				},
				{
					'subtitle': "Raytracing 3D Model - Model Control",
					'vert': "program1.vert",
					'frag': "program1b.frag",
					'model': "cube.obj"
				},
				{
					'subtitle': "Raytracing 3D Model of Wall with Quads not Triangles",
					'vert': "program1.vert",
					'frag': "program1c.frag",
					'model': "wall.obj"
				},
				{
					'subtitle': "Raytracing 3D Model of Ground with Quads not Triangles",
					'vert': "program1.vert",
					'frag': "program1c.frag",
					'model': "ground.obj"
				},
				# {
				# 	'subtitle': "Raytracing 3D Model of Deinonychus with Quads not Triangles",
				# 	'vert': "program1.vert",
				# 	'frag': "program1c.frag",
				# 	'model': "deino.obj"
				# },
			]
		)


	def oninit(self):
		# 1. Prepare Full-Screen Quad for Rendering
		surface_vertices = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
		surface_indices = [(0, 1, 2), (2, 3, 0)] #	not needed

		surface_vertices = data_to_flat_np_array(surface_vertices, surface_indices)

		if self.with_texture:
			texture_uv = [(0, 0), (1, 0), (1, 1), (0, 1)]
			texture_uv = data_to_flat_np_array(texture_uv, surface_indices)
			#	Horizontal-Stack --- place side-by-side:
			#	If surface_verts = [-1, -1, 1, -1, 1, 1, ...] -> x0, y0, x1, y1,...
			#	and texutre_uv = [0, 0, 1, 0, 1, 1, 0, 1, ...] -> u0, v0, u1, v1, ...
			#	below: x0, y0, u0, v0, x1, y1, u1, v1...
			#	vertext_data = [-1, -1, 0, 0, 1, -1, 1, 0, 1, 1, 1, 1]
			vertex_data = np.hstack([surface_vertices, texture_uv])
			self.vertex_format = "2f 2f"

		self.vertex_data = surface_vertices

		#	Read and compile shader program
		vertex_file = self.demonstrations[self.current_demonstration]['vert']
		fragment_file = self.demonstrations[self.current_demonstration]['frag']
		model_name = self.demonstrations[self.current_demonstration]['model']
		self.current_subtitle = self.demonstrations[self.current_demonstration]['subtitle']
		print("\n")
		print(self.current_subtitle)

		vertex_shader = read_file(SHADERS_PATH, vertex_file)
		fragment_shader = read_file(SHADERS_PATH, fragment_file)
		self.shader_program = self.engine.ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)

		self.vbo = self.engine.ctx.buffer(self.vertex_data, reserve=0, dynamic=True)
		self.vao = self.engine.ctx.vertex_array(self.shader_program, [(
			self.vbo, self.vertex_format, *self.attributes,
		)])

		# Read and Initialize Mesh
		self.mesh = Mesh() 
		# self.mesh.model.scale *= 4
		self.mesh.initialize(model_name)

		# Triangle and quad indices have no difference
		upload_flat_data_as_ssbo(
			self.engine.ctx, 0,
			self.mesh.vertex_indices,
			dtype='i4', buffertype="int"
		)
		# Upload SSBOs for Triangle or Quad vertices
		upload_flat_data_as_ssbo(
			self.engine.ctx, 1,
			self.mesh.vertices,
			dtype='f4', buffertype="vec3"
		)

		# len(self.vertex_indices)//self.mesh.corner_count because
		# corner count is 3 for a model using triangles which have 3 consecutive indices
		# but corner count is 4 for a model using quads which have 4 consecutive indices
		set_uniform(self.shader_program, 'primitiveShapeCount',
			len(self.mesh.vertex_indices) // self.mesh.corner_count, self.could_throw)



		# Initialize Camera
		self.camera = FPSCamera(self.engine.dimensions, position=(0, 0, 5))
		self.update_camera_uniforms()

		set_uniform(self.shader_program, 'screenResolution', self.engine.dimensions, self.could_throw)


	def update_camera_uniforms(self):
		set_uniform(self.shader_program, 'camPos', self.camera.position, self.could_throw)
		set_uniform(self.shader_program, 'camDir', self.camera.forward, self.could_throw) 
		set_uniform(self.shader_program, 'camUp', self.camera.up, self.could_throw) 
		set_uniform(self.shader_program, 'camRight', self.camera.right, self.could_throw) 
		set_uniform(self.shader_program, 'camFovTangent', self.camera.fov_tangent, self.could_throw)
		# set_complex_uniform(self.shader_program, 'viewMat', self.camera.view_mat, self.could_throw)

	def update_model(self):
		"""
			the transpose of the inverse of the model matrix is the Normal matrix
			not normal as in original but normal as in 90 degrees.
			It PRESERVES the orientation of the normal vector relative to the surface.
			It is engineered solely to transform normal vectors correctly, as a standard
			model matrix or an inverse model matrix will break them.

			Read more about it in _notes/note2-model_space.md


			Why we use mat3 instead of mat4:
			Normals are direction vectors, not positions.
			They do not have a location in space, meaning
		 	they must never be translated.
		 	Passing a mat3 drops the translation column entirely,
		 	ensuring your calculations remain accurate even if your
		 	cube travels miles away from the origin.
		"""
		self.mesh.model.update(self.engine.delta_time)
		model_matrix = self.mesh.model.mat
		# Extract upper-left 3x3 matrix (handles rotation and scaling)
		model_3x3 = glm.mat3(model_matrix)
		inv_model_3x3 = glm.inverse(model_3x3)
		normal_matrix = glm.transpose(inv_model_3x3)
		set_complex_uniform(self.shader_program, 'invModelMatrix', glm.inverse(model_matrix), self.could_throw)
		set_complex_uniform(self.shader_program, 'normalModelMatrix', normal_matrix, self.could_throw)


	def update(self):
		self.update_camera_uniforms()
		self.camera.update(self.engine.delta_time)
		self.update_model()
		self.sense_demonstration_change()

	def render(self):
		if self.vao is not None:
			self.vao.render()

	def onexit(self):
		self.vao.release()
		self.vbo.release()
		self.shader_program.release()
		self.vao = None
		self.vbo = None
		self.shader_program = None
		self.mesh.release()