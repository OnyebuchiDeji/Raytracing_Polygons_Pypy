"""
Program 5

Implements the fixed BVH traversal, BVH5

Program5.frag implements the BVH traversal and improves
rendering speed fast enough to render `tank.obj` (of 6064 polys) at 18.02 fps,
an ~6x increase from the 2.7fps it could only reach with Global AABB.

Moreover, it renders `deino.obj` (95424 polys) at a whopping 60fps constant. It turns out
deinonychus is less heavy than `tank.obj` (no joke intended)

It also renders `baryonx.obj` (411968 polys) at 60fps. `tank.obj` just happens to be much more
weight than the rest.

Despite both borionx and deino having more polygons, `tank.obj` is more expensive
because of the flat plane rendered with it. That flat plane covers a huge
part of the screen, and hence, increases the number of pixel Ray-Triangle intersection
calculations as the BVH certainly identifies them.

However, Program5.frag has issues with artefacts called "cracks" or
watertightness failures, a common bug of BVH ray tracers.
These cracks show up as SPARSE SINGLE-PIXEL or FEW-PIXEL holes
concentrated at:
- Silhouette/grazing angles on a triangle,
- Boundaries between adjacent BVH leaf boxes,
- thin/elongated triangles.

The solution to this is implemented in Program5b.frag

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
from bvh import BVH5



class Program5(Program):
	def __init__(self):
		super().__init__(
			"Program 5: Rendering Polygon Mesh with Raytracing - Proper Optimization: Bounding Volume Hierarchy (BVH)",
			(	f"Using ordinary raytracing to render a polygon mesh "
				f"on the GPU. No optimizations using BVH."
			),
			engine=Engine(
				program=self, winWidth=1200, winHeight=675
			),
			demonstrations=[
				{
					'subtitle': "Raytracing 3D Model of Cube with Triangles Only",
					'vert': "program5.vert",
					'frag': "program5.frag",
					'model': "cube.obj"
				},
				{
					'subtitle': "Raytracing 3D Model of Wall with Triangles Only",
					'vert': "program5.vert",
					'frag': "program5.frag",
					'model': "wall.obj"
				},
				{
					'subtitle': "Raytracing 3D Model of Ground with Triangles Only",
					'vert': "program5.vert",
					'frag': "program5.frag",
					'model': "ground.obj"
				},
				{
					'subtitle': "Raytracing 3D Model of a Tank with Triangles Only",
					'vert': "program5.vert",
					'frag': "program5.frag",
					'model': "tank.obj"
				},
				{
					'subtitle': "Raytracing 3D Model of Deinonychus with Triangles Only",
					'vert': "program5.vert",
					'frag': "program5.frag",
					'model': "deino.obj"
				},
				{
					'subtitle': "Raytracing 3D Model of Baryonx with Triangles Only",
					'vert': "program5.vert",
					'frag': "program5.frag",
					'model': "baryonx.obj"
				},
				{
					'subtitle': "Raytracing 3D Model of Axis with Triangles Only",
					'vert': "program5.vert",
					'frag': "program5.frag",
					'model': "axis.obj"
				},
				{
					'subtitle': "Raytracing 3D Model of Mountains with Triangles Only",
					'vert': "program5.vert",
					'frag': "program5.frag",
					'model': "mountains.obj"
				},
				{
					'subtitle': "Raytracing 3D Model of Teapot with Triangles Only",
					'vert': "program5.vert",
					'frag': "program5.frag",
					'model': "teapot.obj"
				},
				{
					'subtitle': "Raytracing 3D Model of VideoShip with Triangles Only",
					'vert': "program5.vert",
					'frag': "program5.frag",
					'model': "VideoShip.obj"
				},
			]
		)


	def oninit(self):
		pg.event.set_grab(True)
		pg.mouse.set_visible(False)

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


		# Initialize Camera
		self.camera = FPSCamera(self.engine.dimensions, position=(0, 0, 5))
		self.update_camera_uniforms()

		set_uniform(self.shader_program, 'screenResolution', self.engine.dimensions, self.could_throw)


		# Read and Initialize Mesh
		self.mesh = Mesh() 
		# self.mesh.model.scale *= 4
		self.mesh.initialize_v3(model_name)


		#	Program 2 uniforms for backward-compatibility
		set_uniform(self.shader_program, 'isRayTriIntersect',
			True, self.could_throw 
		)

		set_uniform(self.shader_program, 'primitiveShapeCount',
			len(self.mesh.vertex_indices) // 3, self.could_throw)

		# Set Global AABB uniforms
		set_uniform(self.shader_program, 'meshAABBMin',
			self.mesh.aabb.min, self.could_throw
		)
		set_uniform(self.shader_program, 'meshAABBMax',
			self.mesh.aabb.max, self.could_throw
		)

		# Triangle and quad indices have no difference
		upload_flat_data_as_ssbo(
			self.engine.ctx, 0,
			self.mesh.vertex_indices,
			dtype='i4', buffertype="int"
		)
		# Upload SSBOs for Triangle Vertices
		upload_flat_data_as_ssbo(
			self.engine.ctx, 1,
			self.mesh.vertices,
			dtype='f4', buffertype="vec3"
		)

		# Initialize and Set BVH
		vertices = [v for idx, v in enumerate(self.mesh.vertices) if idx == 0 or (idx + 1) % 4 != 0]
		self.bvh = BVH5(vertices, self.mesh.vertex_indices)
		self.bvh.flatten_n_upload_to_ssbo(self.engine.ctx)


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