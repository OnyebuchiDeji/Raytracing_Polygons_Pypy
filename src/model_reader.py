from utils import read_file
from config import MODELS_PATH
import array
import os
from model import Model, glm


class AABB3D:
	def __init__(self):
		self.min = glm.vec3()
		self.max = glm.vec3()
		self.initialized = False

	def assign(self, x:float, y:float, z:float):
		val = glm.vec3(x, y, z)
		if not self.initialized:
			self.min = val
			self.max = val
			self.initialized = True

		self.min = glm.min(self.min, val)
		self.max = glm.max(self.max, val)



def clean_line_parts_array(arr):
	return [val for val in arr.split(" ") if len(val)>=1]

class Mesh:
	def __init__(self):
		"""All flat arrays"""
		self.vertices = array.array('f')
		self.normals = array.array('f')
		self.textures_uvs = array.array('f')
		self.vertex_indices = array.array('f')
		self.texture_indices = array.array('f')
		self.normal_indices = array.array('f')
		self.model = Model()
		self.corner_count = None
		# specify the bounding box AABB of whole Mesh
		self.aabb = AABB3D()

	def release(self):
		array_members = [attr for attr in dir(self) if type(getattr(self, attr)) == array.array]
		for member in array_members:
			# no empty or clear so just reinitialize with empty array
			setattr(self, member, array.array('f'))


	def initialize_v3(self, modelname: str):
		"""
		Final Industry-standard method that splits
		quads into triangles on the CPU during the asset
		loading phase.
		It completely resolves the issue with flat or non-planar
		polygons while allowing reuse of the original, highly
		optimized Möller–Trumbore ray-triangle intersection routine
		on the GPU.

		Loads a .obj file, parses vertices and faces, and splits any 
    	quad primitives into triangles while keeping track of winding order.
		"""
		excluded = ['#', 'm', 'o', 'u', 's']

		# 1. Read Model '.obj'
		with open(os.path.join(MODELS_PATH, modelname), "r") as rfs:
			for line in rfs:
				line = line.strip()
				if len(line) == 0 or line[0] in excluded:
					continue
				output_line = None
				try:
					output_line = line
					if line.startswith('v '):
						# print(line)
						# the + [1] pads the vertices with an extra w component for suitability
						# with GPU SSBO memory layout compliance.
						line_parts = list(map(float, clean_line_parts_array(line)[1:])) + [1]
						self.aabb.assign(*line_parts[:3])
						self.vertices.extend(line_parts)

					# elif line.startswith('vt'):
					# 	line_parts = list(map(float, clean_line_parts_array(line)[1:]))
					# 	self.textures_uvs.extend(line_parts)
					# elif line.startswith('vn'):
					# 	line_parts = list(map(float, clean_line_parts_array(line)[1:]))
					# 	self.normals.extend(line_parts)
					
					elif line.startswith('f'):
						line_parts = list(map(lambda x:x.strip(), clean_line_parts_array(line)[1:]))

						# Some Faces were not Faces but lines having no vt values.
						# they have the format: "f 607//562 608//563 609//563 610//562",
						# only consisting of vertex//normal values
						# At first, I thought to skip these:
						# if len(line_parts[0].split("/")) == 2:
						# 	continue

						corner_count = len(line_parts)

						# the logic would split the line_parts into sub parts
						# for instances with '607//562 608//563 609//563 610//562'
						# where vt indices are missing, it replaces those with None
						# it performs this check for all
						# the below is dependent on the size/length of the split parts array
						# face_list_flat = [(int(val)-1 if len(val) > 0 else None) for face_corner in line_parts for val in face_corner.split("/")]
						face_list_flat = []

						# The below change is to ensure that 3D models with faces of without 1/1/1 2/2/2 3/3/3 but rather just 1 2 3:
						# For the missing face parts, use None 
						# parts = [face_corner.split("/") for face_corner in line_parts]
						for face_corner in line_parts:
							parts = face_corner.split("/")
							for iidx in range(3):
								val = None
								# for faces/face_corner/face_values like: "f 6 5 4", that have no '/'
								# the below makes the missing parts be filled with None
								if iidx == 0 or iidx % len(parts) != 0:
									if len(parts[iidx]) > 0:
										val = int(parts[iidx]) - 1
								face_list_flat.append(val)

						flat_vertex_indices  = face_list_flat[0::3]
						flat_texture_indices = face_list_flat[1::3]
						flat_normal_indices  = face_list_flat[2::3]

						def conditional_extend(target_array, subject_array):
							"""
							Performs contdional extend for face instances with ommitted indices
							such as: '607//562 608//563 609//563 610//562'
							"""
							if subject_array[0] is not None:
								target_array.extend(subject_array)


						# --- TRIANGULATION LOGIC

						if corner_count == 3:
							# the step stride separating consecutive conrner values is always 3
							conditional_extend(self.vertex_indices, flat_vertex_indices)
							conditional_extend(self.texture_indices, flat_texture_indices)
							conditional_extend(self.normal_indices,flat_normal_indices)

						elif corner_count == 4:
							# It;s a Quad! Split into two triangles;
							# Triangle 1: v0 -> v1 -> v2
							# Triangle 2: v0 -> v2 -> v3
							index_list = [(0, 1, 2), (0, 2, 3)]
							for idxs in index_list:
								conditional_extend(self.vertex_indices, [
									flat_vertex_indices[idxs[0]],
									flat_vertex_indices[idxs[1]],
									flat_vertex_indices[idxs[2]]
								])
								conditional_extend(self.texture_indices, [
									flat_texture_indices[idxs[0]],
									flat_texture_indices[idxs[1]],
									flat_texture_indices[idxs[2]]
								])
								conditional_extend(self.normal_indices, [
									flat_normal_indices[idxs[0]],
									flat_normal_indices[idxs[1]],
									flat_normal_indices[idxs[2]]
								])

						else:
							# Optional: Handle n-gons (Fans) if in model
							vert0 = flat_vertex_indices[0]
							for i in range(1, len(flat_vertex_indices) - 1):
								conditional_extend(self.vertex_indices,
									[ vert0, flat_vertex_indices[i], flat_vertex_indices[i+1] ])
							tex0 = flat_texture_indices[0]
							for i in range(1, len(flat_texture_indices) - 1):
								conditional_extend(self.texture_indices,
									[ tex0, flat_texture_indices[i], flat_texture_indices[i+1] ])
							norm0 = flat_normal_indices[0]
							for i in range(1, len(flat_normal_indices) - 1):
								conditional_extend(self.normal_indices,
									[ norm0, flat_normal_indices[i], flat_normal_indices[i+1] ])


				except Exception as e:
					print((
						f"Mesh Initialize Error: ",
						f"line value: {output_line} ",
						f"Error: {str(e)}"
					))

		# print("No. of Polygons: ", len(self.vertex_indices) // 3)
		# // 4 because each vertex has 4 components; x, y, z, w --- w being the extra 
		# print("No. of Vertices: ", len(self.vertices) // 4)



	def initialize_v2(self, modelname: str):
		"""
		My Implementation. It wasn't accurate enough.
		It worked somewhat for faces with 4 corners.
		But even then, it had some issues
		"""
		excluded = ['#', 'm', 'o', 'u', 's']

		# 1. Read Model '.obj'
		with open(os.path.join(MODELS_PATH, modelname), "r") as rfs:
			for line in rfs:
				line = line.strip()
				if len(line) == 0 or line[0] in excluded:
					continue
				output_line = None
				try:
					output_line = line 
					if line.startswith('v '):
						line_parts = list(map(float, clean_line_parts_array(line)[1:])) + [1]
						self.vertices.extend(line_parts)

					# elif line.startswith('vt'):
					# 	line_parts = list(map(float, clean_line_parts_array(line)[1:]))
					# 	self.textures_uvs.extend(line_parts)
					# elif line.startswith('vn'):
					# 	line_parts = list(map(float, clean_line_parts_array(line)[1:]))
					# 	self.normals.extend(line_parts)
					
					elif line.startswith('f'):
						line = line.replace("//", "/")
						line_parts = list(map(lambda x:x.strip(), clean_line_parts_array(line)[1:]))
						corner_count = len(line_parts)

						face_list_flat = [int(val)-1 for face_corner in line_parts for val in face_corner.strip().split("/")]

						flat_vertex_indices  = face_list_flat[0::3]
						flat_texture_indices = face_list_flat[1::3]
						flat_normal_indices  = face_list_flat[2::3]

						# Extract from surplus corners an appropriate number of
						# triangle indices that do not overlap
						if corner_count > 3:
							stack = list(range(corner_count))
							idx = 0 # start idx
							while len(stack) > 3:
								start = stack[idx % len(stack)]
								middle = stack[(idx + 1) % len(stack)]
								end = stack[(idx + 2) % len(stack)]
								self.vertex_indices.extend([
									flat_vertex_indices[start],
									flat_vertex_indices[middle],
									flat_vertex_indices[end]
								])
								self.texture_indices.extend([
									flat_texture_indices[start],
									flat_texture_indices[middle],
									flat_texture_indices[end]
								])
								self.normal_indices.extend([
									flat_normal_indices[start],
									flat_normal_indices[middle],
									flat_normal_indices[end]
								])
								idx += 1
								stack.remove(middle)
							self.vertex_indices.extend(stack)
							self.texture_indices.extend(stack)
							self.normal_indices.extend(stack)

						else:
							# the step stride separating consecutive conrner values is always 3
							self.vertex_indices.extend(flat_vertex_indices)
							self.texture_indices.extend(flat_texture_indices)
							self.normal_indices.extend(flat_normal_indices)

				except Exception as e:
					print((
						f"Mesh Initialize Error: ",
						f"line value: {output_line} ",
						f"Error: {str(e)}"
					))


	def initialize(self, modelname: str):
		"""
		This algorithm is only able to read .obj models
		whose 3D models use triangles.
		Hence why corner count can be 3 or 4
		3 -> triangles
		4 -> quads
		"""
		excluded = ['#', 'm', 'o', 'u', 's']

		# 1. Read Model '.obj'
		with open(os.path.join(MODELS_PATH, modelname), "r") as rfs:
			while True:
				line = rfs.readline()
				if not line:
					break
				line = line.strip()

				if len(line) == 0 or line[0] in excluded:
					continue

				output_line = None
				try:
					output_line = line 
					if line.startswith('v '):
						"""
						could add [1] to the end of each vertex point to change
						from x y z -> x y z 1.
						This is NEEDED for the SSBO as...
						A vec3 array in an SSBO using std430 is still padded to
						vec4 alignment (16 bytes per element, leaving 4 bytes empty at the end).
						So pad the data array with a 1.0, so now its shape will be (-1, 4) (which is
						same as (2, 4)) instead of (-1, 3) (or (2, 3))
						"""
						line_parts = list(map(float, clean_line_parts_array(line)[1:])) + [1]
						self.aabb.assign(*line_parts[:3]) # <- only need x,y,z
						self.vertices.extend(line_parts)

					# elif line.startswith('vt'):
					# 	line_parts = list(map(float, clean_line_parts_array(line)[1:]))
					# 	self.textures_uvs.extend(line_parts)
					# elif line.startswith('vn'):
					# 	line_parts = list(map(float, clean_line_parts_array(line)[1:]))
					# 	self.normals.extend(line_parts)
					
					elif line.startswith('f'):
						line = line.replace("//", "/")
						line_parts = list(map(lambda x:x.strip(), clean_line_parts_array(line)[1:]))
						if self.corner_count is None:
							self.corner_count = len(line_parts)
						# e.g.: ['2/1/1' ,'3/2/1', '4/3/1']
						# each string is a face corner, carrying the index for
						# vertex pos/texture pos/vertex normal -> v/vt/vn	
						# the reason for doing int(val) - 1
						# is to first make the number an int and
						# to make the indexing start from 0 so they can
						# properly identify the corresponding values in their arrays:
						# e.g. vertex_indices indentifies self.vertex_indices
						face_list_flat = [int(val)-1 for face_corner in line_parts for val in face_corner.strip().split("/")]
						# the step stride separating consecutive conrner values is always 3
						self.vertex_indices.extend(face_list_flat[0::3])
						self.texture_indices.extend(face_list_flat[1::3])
						self.normal_indices.extend(face_list_flat[2::3])
				except Exception as e:
					print((
						f"Mesh Initialize Error: ",
						f"line value: {output_line} ",
						f"Error: {str(e)}"
					))

		# print("Length of Vertices: ", len(self.vertices))
		# print("Length of Vertex Indices: ", len(self.vertex_indices))
		# print("No. of Polygons: ", len(self.vertex_indices) // self.corner_count)
		# print("No. of Vertices: ", len(self.vertices) // 3)
		# print("Corner Count: ", self.corner_count)
		# If rendering was normal, all the 
		# vertices, texture_uvs and normals
		# would be reordered according to their indices.
		# Then all used to create a vao.
		# But because I will be rendering using raytracing
		# and not with the ordinary rendering pipeline that
		# involves sending vertex data
		# to the vertex shader before its rasterized, but
		# , I will do the reordering in the GPU --- no need for vao. 