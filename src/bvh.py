"""
The Bounding Volume Hierarchy Structure
"""

from utils import np

PAD = 1e-4

class BVH5:
	def __init__(self, vertices, indices):
		self.verts = np.array(vertices, dtype=np.float32).reshape(-1, 3)

		# Modify indices by removing ending indices
		# If current shape of indices is not reshapable to (-1, 3)
		shape = np.array(indices).shape[0]
		acceptable_shape = shape - (int(shape) % 3)
		indices = indices[:acceptable_shape]
		self.tris = np.array(indices, dtype=np.int32).reshape(-1, 3)

		# triangle count
		self.N = len(self.tris)

		# Compute triangle centroids once
		self.centroids = (self.verts[self.tris[:,0]] +
			self.verts[self.tris[:,1]] + self.verts[self.tris[:,2]]
		) / 3.0


		# Node struct fields storage
		self.nodes = []

		# a numpy type definition of a struct
		# Each node: [minBounds (3), maxBounds(3), left, right, firstTri, triCount]
		self.dtype = np.dtype([
			('min', np.float32, 3), ('max', np.float32, 3),
			('left', np.int32), ('right', np.int32),
			('first', np.int32), ('count', np.int32)
		])

		# Array of triangle indices (permuatation, initialized 0..N-1)
		self.tri_idx = np.arange(self.N, dtype=np.int32)

		self.root = self.build_node(0, self.N)

	def build_node(self, first, count):
		"""
		Recursively build BVH node covering triangles...
		tri_idx[first:first+count].

		The code follows the logic described by Jacco:
		compute axis, splitPos, then do an in-place partition (like quicksort).
		Stop splitting at <=2 triangles.
		After partitioning, two child nodes are created and `leftChild/rightChild`
		indices are set; the parent's count is set to 0 (making it interior).
		A leaf has `count>0`, interior has count=0
		"""
		# Create new node
		node_index = len(self.nodes)

		# Initialize struct
		self.nodes.append(np.zeros(1, dtype=self.dtype)[0])
		node = self.nodes[node_index]

		# assign values
		node['first'], node['count'], node['left'], node['right'] = first, count, -1, -1

		# Compute AABB of all triangles in this node
		# Initialize with extreme values
		mins = np.full(3, np.inf, dtype=np.float32)
		maxs = np.full(3, -np.inf, dtype=np.float32)

		for i in range(first, first+count):
			tri_id = self.tri_idx[i]
			vs = self.verts[self.tris[tri_id]]
			mins = np.minimum(mins, vs.min(axis=0))
			maxs = np.maximum(maxs, vs.max(axis=0))

		"""
		Solution/Fix4: Defense in depth; pad leaf AaBBs at build time:
		Even with fix3, one can still get the single-pixel cracks from `box-vs-box` precision at the AABB test level itself due to a ray that should enter a sibling box numerically just missing it by an ULP.
		The standard ROBUSTNESS technique is to FATTEN EACH NODE's AABB A TINY EPSILON when building the tree. Hence, the boxes overlap very slightly instead of touching exactly.
		This is implemented in BVH5 on the CPU-side.

		Pad the box slightly to guard against floating point "cracks":
		without this, two leaf boxes that are meant to share a boundary face
		can, due to FP rounding in the GPU slab test, appear to have a ray
		pass between them at that exact boundary, producing a miss for both.
		A small epsilon padding removes that razor-thin gap.

		Scale this relative to the scene/object size if the model is not
		roughly unit-scaled -- e.g. pad = 1e-4 * max(extents of the whole mesh).

		This is cheap (constant per-node cost at build time; doesn't affect triangle counts
		or tree shape), and is standard practice in production BVH ray tracers
		specifically to prevent the exact 'cracks' symptom
		"""
		pad = PAD
		node['min'] = mins - pad
		node['max'] = maxs + pad

		# Lead criterion
		if count <= 2:
			return node_index

		# Choose split axis = longest
		extents = maxs - mins
		axis = np.argmax(extents)
		split_pos = mins[axis] + extents[axis] * 0.5

		# Partition triangles around the split plane
		i = first; j = first + count - 1
		while i <= j:
			c = self.centroids[self.tri_idx[i], axis]
			if c < split_pos:
				i += 1
			else:
				# swap tri_idx[i] and tri_idx[j]
				self.tri_idx[i], self.tri_idx[j] = self.tri_idx[j], self.tri_idx[i]
				j -= 1

		left_count = i - first
		if left_count == 0 or left_count == count:
			# Partition failed (all on one side): split in half
			left_count = count // 2
			i = first + left_count

		# Create children nodes
		node['left'] = self.build_node(first, left_count)
		node['right'] = self.build_node(first + left_count, count - left_count)
		node['count'] = 0 # no triangles at interior node
		return node_index


	def flatten_n_upload_to_ssbo(self, ctx):
		"""
		Pad min/max to 4 floats/node so the buffer's real
		stride (16 bytes) matches what std430 forces the GLSL vec3[]
		array stride to be
		"""
		node_count = len(self.nodes)
		bvh_min = np.zeros((node_count, 4), dtype=np.float32)
		bvh_max = np.zeros((node_count, 4), dtype=np.float32)
		lefts = np.zeros(node_count, dtype=np.int32)
		rights = np.zeros(node_count, dtype=np.int32)
		firsts = np.zeros(node_count, dtype=np.int32)
		counts = np.zeros(node_count, dtype=np.int32)

		for i, nd in enumerate(self.nodes):
			bvh_min[i, :3] = nd['min'] # bvh_min[i, 3] stays 0 as padding
			bvh_max[i, :3] = nd['max'] # bvh_max[i, 3] stays 0 as padding
			lefts[i]   = nd['left']
			rights[i]  = nd['right']
			firsts[i]  = nd['first']
			counts[i]  = nd['count']

		# Upload as SSBOs

		# These are an additional 6 SSBOs
		# though they could be packed into a single
		# struct SSBO, separating them simplifies
		# alignment

		bvh_min_buf = ctx.buffer(bvh_min.tobytes())
		bvh_min_buf.bind_to_storage_buffer(2)

		bvh_max_buf = ctx.buffer(bvh_max.tobytes())
		bvh_max_buf.bind_to_storage_buffer(3)

		left_buf = ctx.buffer(lefts.tobytes())
		left_buf.bind_to_storage_buffer(4)

		right_buf = ctx.buffer(rights.tobytes())
		right_buf.bind_to_storage_buffer(5)

		first_buf = ctx.buffer(firsts.tobytes())
		first_buf.bind_to_storage_buffer(6)

		count_buf = ctx.buffer(counts.tobytes())
		count_buf.bind_to_storage_buffer(7)

		"""
		The 'self.tri_idx' reflects the partitioning reordering
		of how the triangles should be iterated over while searching
		for an intersection in the GPU.
		Hence, they out to be sent to the GPU.
		"""
		tri_perm = ctx.buffer(self.tri_idx.astype(np.int32).tobytes())
		tri_perm.bind_to_storage_buffer(8)



class BVH4:
	"""
	The issue with the BVHs thus far:
	1) (root cause): std430 alignment mismatch on the AABB buffers
	The issue is the SSBO vec3 trap.

	In std430 layout, a vec3 still has a base alignment of 16 bytes, rule
	3 of the std140/std430 spec: "a three-component vector has base alignment
	4N", i.e., rounded up to `vec4`.
	std430 only relaces padding for structs/arrays of structs, not for the element
	size of a bare `vec3` array as used below in `flatten_n_upload_to_ssbo`

	Hence, the GPU treats the corresponding shader code:
		```
		layout(std430, binding=2) buffer BVHMin { vec3 bmin[]; };
		layout(std430, binding=2) buffer BVHMax { vec3 bmax[]; };
		```
	used in `program4.frag` and `program4b.frag` as having a 16-byte stride per element, that is,
	12 bytes of data + 4 bytes padding.

	But on the CPU side, in `flatten_n_upload_to_ssbo`:
	```
	bvh_min = np.zeros((node_count, 3), dtype=np.float32) # 12 bytes/node, tightly packed
	...
	bvh_min_buf = ctx.buffer(bvh_min.tobytes())
	```
	This uploads a tightly packed 12-byte-per-node buffer.
	`ctx.buffer()` does a raw byte copy -- it doesn't consider GLSL's alignment rules

	The result

	node 0 happens to read correctly (offset 0 lines up either way).
	But every nde after that is offset by 4 * nodeIndex bytes.
	`bmin[1]` in the shader ends up reading bytes belonging to `{min.y, min.z, max.x}`
	of the CPU data.
	`bmin[2]` is even further out of phase, and so on.
	The AABBs the shader tests against are essentially garbage for any node beyond the rot.

	Garbage AAABBs mean whole subtrees get incorrectly CULLED (they don't appear to intersect
	the ray) OR nodes read past the buffer entirely --- this is what is seen if `Progam4.py` is
	run with either `Program4.frag` or `Program4b.frag`, that is, RANDOMLY MISSING TRIANGLES, and
	it is worse the deeper/larger the tree is.

	This also matches why the Vertices buffer (which was correctly declared as `vec4`) doesn't have
	this problem --- I had solved this but not for bmin/bmax

	It does not affect the flat interger arrays.
	
	That is, the left/right/first/count SSBOs (bindings 4–7)
	are plain int arrays — those are fine as-is,
	since scalar int arrays have a 4-byte stride in both std430 and
 	the NumPy int32 upload, so no padding mismatch there.
	"""
	def __init__(self, vertices, indices):
		self.verts = np.array(vertices, dtype=np.float32).reshape(-1, 3)

		# Modify indices by removing ending indices
		# If current shape of indices is not reshapable to (-1, 3)
		shape = np.array(indices).shape[0]
		acceptable_shape = shape - (int(shape) % 3)
		indices = indices[:acceptable_shape]
		self.tris = np.array(indices, dtype=np.int32).reshape(-1, 3)

		# triangle count
		self.N = len(self.tris)

		# Compute triangle centroids once
		self.centroids = (self.verts[self.tris[:,0]] +
			self.verts[self.tris[:,1]] + self.verts[self.tris[:,2]]
		) / 3.0


		# Node struct fields storage
		self.nodes = []

		# a numpy type definition of a struct
		# Each node: [minBounds (3), maxBounds(3), left, right, firstTri, triCount]
		self.dtype = np.dtype([
			('min', np.float32, 3), ('max', np.float32, 3),
			('left', np.int32), ('right', np.int32),
			('first', np.int32), ('count', np.int32)
		])

		# Array of triangle indices (permuatation, initialized 0..N-1)
		self.tri_idx = np.arange(self.N, dtype=np.int32)

		self.root = self.build_node(0, self.N)

	def build_node(self, first, count):
		"""
		Recursively build BVH node covering triangles...
		tri_idx[first:first+count].

		The code follows the logic described by Jacco:
		compute axis, splitPos, then do an in-place partition (like quicksort).
		Stop splitting at <=2 triangles.
		After partitioning, two child nodes are created and `leftChild/rightChild`
		indices are set; the parent's count is set to 0 (making it interior).
		A leaf has `count>0`, interior has count=0
		"""
		# Create new node
		node_index = len(self.nodes)

		# Initialize struct
		self.nodes.append(np.zeros(1, dtype=self.dtype)[0])
		node = self.nodes[node_index]

		# assign values
		node['first'], node['count'], node['left'], node['right'] = first, count, -1, -1

		# Compute AABB of all triangles in this node
		# Initialize with extreme values
		mins = np.full(3, np.inf, dtype=np.float32)
		maxs = np.full(3, -np.inf, dtype=np.float32)

		for i in range(first, first+count):
			tri_id = self.tri_idx[i]
			vs = self.verts[self.tris[tri_id]]
			mins = np.minimum(mins, vs.min(axis=0))
			maxs = np.maximum(maxs, vs.max(axis=0))

		node['min'] = mins
		node['max'] = maxs

		# Lead criterion
		if count <= 2:
			return node_index

		# Choose split axis = longest
		extents = maxs - mins
		axis = np.argmax(extents)
		split_pos = mins[axis] + extents[axis] * 0.5

		# Partition triangles around the split plane
		i = first; j = first + count - 1
		while i <= j:
			c = self.centroids[self.tri_idx[i], axis]
			if c < split_pos:
				i += 1
			else:
				# swap tri_idx[i] and tri_idx[j]
				self.tri_idx[i], self.tri_idx[j] = self.tri_idx[j], self.tri_idx[i]
				j -= 1

		left_count = i - first
		if left_count == 0 or left_count == count:
			# Partition failed (all on one side): split in half
			left_count = count // 2
			i = first + left_count

		# Create children nodes
		node['left'] = self.build_node(first, left_count)
		node['right'] = self.build_node(first + left_count, count - left_count)
		node['count'] = 0 # no triangles at interior node
		return node_index


	def flatten_n_upload_to_ssbo(self, ctx):
		node_count = len(self.nodes)
		bvh_min = np.zeros((node_count, 3), dtype=np.float32)
		bvh_max = np.zeros((node_count, 3), dtype=np.float32)
		lefts = np.zeros(node_count, dtype=np.int32)
		rights = np.zeros(node_count, dtype=np.int32)
		firsts = np.zeros(node_count, dtype=np.int32)
		counts = np.zeros(node_count, dtype=np.int32)

		for i, nd in enumerate(self.nodes):
			bvh_min[i] = nd['min']
			bvh_max[i] = nd['max']
			lefts[i]   = nd['left']
			rights[i]  = nd['right']
			firsts[i]  = nd['first']
			counts[i]  = nd['count']

		# Upload as SSBOs

		# These are an additional 6 SSBOs
		# though they could be packed into a single
		# struct SSBO, separating them simplifies
		# alignment

		bvh_min_buf = ctx.buffer(bvh_min.tobytes())
		bvh_min_buf.bind_to_storage_buffer(2)

		bvh_max_buf = ctx.buffer(bvh_max.tobytes())
		bvh_max_buf.bind_to_storage_buffer(3)

		left_buf = ctx.buffer(lefts.tobytes())
		left_buf.bind_to_storage_buffer(4)

		right_buf = ctx.buffer(rights.tobytes())
		right_buf.bind_to_storage_buffer(5)

		first_buf = ctx.buffer(firsts.tobytes())
		first_buf.bind_to_storage_buffer(6)

		count_buf = ctx.buffer(counts.tobytes())
		count_buf.bind_to_storage_buffer(7)

		"""
		The 'self.tri_idx' reflects the partitioning reordering
		of how the triangles should be iterated over while searching
		for an intersection in the GPU.
		Hence, they out to be sent to the GPU.
		"""
		tri_perm = ctx.buffer(self.tri_idx.astype(np.int32).tobytes())
		tri_perm.bind_to_storage_buffer(8)


class BVH3:
	def __init__(self, vertices, indices, cornerCount):
		self.verts = np.array(vertices, dtype=np.float32).reshape(-1, 3)
		self.corner_count = cornerCount
		# primitives, referring to unit shapes like triangles or quads
		# in this case it's called `primitives` because it can be either
		self.primitives = np.array(indices, dtype=np.int32).reshape(-1, cornerCount)

		# primitives count
		self.N = len(self.primitives)

		# Compute primitives centroids once
		if cornerCount  == 3:
			self.centroids = (self.verts[self.primitives[:,0]] +
				self.verts[self.primitives[:,1]] + self.verts[self.primitives[:,2]]
			) / float(cornerCount)
		elif cornerCount == 4:
			self.centroids = (self.verts[self.primitives[:,0]] +
				self.verts[self.primitives[:,1]] + self.verts[self.primitives[:,2]] +
				self.verts[self.primitives[:,3]]
			) / float(cornerCount)


		# Node struct fields storage
		self.nodes = []

		# a numpy type definition of a struct
		# Each node: [minBounds (3), maxBounds(3), left, right, firstTri, triCount]
		self.dtype = np.dtype([
			('min', np.float32, 3), ('max', np.float32, 3),
			('left', np.int32), ('right', np.int32),
			('first', np.int32), ('count', np.int32)
		])

		# Array of triangle indices (permuatation, initialized 0..N-1)
		self.prim_idx = np.arange(self.N, dtype=np.int32)

		self.root = self.build_node(0, self.N)

	def build_node(self, first, count):
		"""
		Recursively build BVH node covering triangles...
		tri_idx[first:first+count].

		The code follows the logic described by Jacco:
		compute axis, splitPos, then do an in-place partition (like quicksort).
		Stop splitting at <=2 triangles.
		After partitioning, two child nodes are created and `leftChild/rightChild`
		indices are set; the parent's count is set to 0 (making it interior).
		A leaf has `count>0`, interior has count=0
		"""
		# Create new node
		node_index = len(self.nodes)

		# Initialize struct
		self.nodes.append(np.zeros(1, dtype=self.dtype)[0])
		node = self.nodes[node_index]

		# assign values
		node['first'], node['count'], node['left'], node['right'] = first, count, -1, -1

		# Compute AABB of all triangles in this node
		# Initialize with extreme values
		mins = np.full(3, np.inf, dtype=np.float32)
		maxs = np.full(3, -np.inf, dtype=np.float32)

		for i in range(first, first+count):
			prim_id = self.prim_idx[i]
			vs = self.verts[self.primitives[prim_id]]
			mins = np.minimum(mins, vs.min(axis=0))
			maxs = np.maximum(maxs, vs.max(axis=0))

		node['min'] = mins
		node['max'] = maxs

		# Lead criterion
		if count <= 2:
			return node_index

		# Choose split axis = longest
		extents = maxs - mins
		axis = np.argmax(extents)
		split_pos = mins[axis] + extents[axis] * 0.5

		# Partition triangles around the split plane
		i = first; j = first + count - 1
		while i <= j:
			c = self.centroids[self.prim_idx[i], axis]
			if c < split_pos:
				i += 1
			else:
				# swap prim_idx[i] and prim_idx[j]
				self.prim_idx[i], self.prim_idx[j] = self.prim_idx[j], self.prim_idx[i]
				j -= 1

		left_count = i - first
		if left_count == 0 or left_count == count:
			# Partition failed (all on one side): split in half
			left_count = count // 2
			i = first + left_count

		# Create children nodes
		node['left'] = self.build_node(first, left_count)
		node['right'] = self.build_node(first + left_count, count - left_count)
		node['count'] = 0 # no primitives at interior node
		return node_index


	def flatten_n_upload_to_ssbo(self, ctx):
		node_count = len(self.nodes)
		bvh_min = np.zeros((node_count, 3), dtype=np.float32)
		bvh_max = np.zeros((node_count, 3), dtype=np.float32)
		lefts = np.zeros(node_count, dtype=np.int32)
		rights = np.zeros(node_count, dtype=np.int32)
		firsts = np.zeros(node_count, dtype=np.int32)
		counts = np.zeros(node_count, dtype=np.int32)

		for i, nd in enumerate(self.nodes):
			bvh_min[i] = nd['min']
			bvh_max[i] = nd['max']
			lefts[i]   = nd['left']
			rights[i]  = nd['right']
			firsts[i]  = nd['first']
			counts[i]  = nd['count']

		# Upload as SSBOs

		# These are an additional 6 SSBOs
		# though they could be packed into a single
		# struct SSBO, separating them simplifies
		# alignment

		bvh_min_buf = ctx.buffer(bvh_min.tobytes())
		bvh_min_buf.bind_to_storage_buffer(2)

		bvh_max_buf = ctx.buffer(bvh_max.tobytes())
		bvh_max_buf.bind_to_storage_buffer(3)

		left_buf = ctx.buffer(lefts.tobytes())
		left_buf.bind_to_storage_buffer(4)

		right_buf = ctx.buffer(rights.tobytes())
		right_buf.bind_to_storage_buffer(5)

		first_buf = ctx.buffer(firsts.tobytes())
		first_buf.bind_to_storage_buffer(6)

		count_buf = ctx.buffer(counts.tobytes())
		count_buf.bind_to_storage_buffer(7)