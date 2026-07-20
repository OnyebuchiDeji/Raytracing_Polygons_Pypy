"""
CPU-side BVH: object-tree build (with proper depth tracking) + flatten to
GPU-ready flat arrays, verified against both a Python re-implementation of
the GLSL traversal and a brute-force reference.

Design choices, and why:

  - Build produces an actual BVHNode TREE (not arrays directly) so the build
    algorithm can be inspected/debugged like any other tree algorithm, and so
    depth is a first-class, tracked quantity (self.max_depth), not an
    afterthought.
  - Split selection uses BINNED SAH (surface-area heuristic), the same
    approach production CPU BVH builders (e.g. Embree, PBRT) use -- not a
    naive median split, which produces much worse traversal for irregular
    triangle density.
  - Flatten uses the classic "left child implicit" layout: a node's left
    child is ALWAYS placed at (index + 1) by construction order, so only the
    right child's index needs to be stored explicitly. This halves the
    per-node integer storage compared to storing both child indices.
  - GPU node struct is (vec4, vec4, ivec4) -- no vec3 anywhere. This sidesteps
    the std430 vec3-array padding trap entirely, since vec4/ivec4 already sit
    on 16-byte boundaries.
"""

import numpy as np


# ---------------------------------------------------------------------------
# 1. AABB + BVHNode (CPU tree representation)
# ---------------------------------------------------------------------------

class AABB:
    __slots__ = ("bmin", "bmax")

    def __init__(self, bmin=None, bmax=None):
        self.bmin = np.array(bmin, dtype=np.float64) if bmin is not None else np.full(3, np.inf)
        self.bmax = np.array(bmax, dtype=np.float64) if bmax is not None else np.full(3, -np.inf)

    def surface_area(self):
        d = np.maximum(self.bmax - self.bmin, 0.0)
        return 2.0 * (d[0] * d[1] + d[1] * d[2] + d[2] * d[0])


class BVHNode:
    __slots__ = ("bounds", "left", "right", "first_prim", "prim_count", "depth")

    def __init__(self):
        self.bounds = AABB()
        self.left = None
        self.right = None
        self.first_prim = 0
        self.prim_count = 0     # > 0  => leaf
        self.depth = 0

    @property
    def is_leaf(self):
        return self.prim_count > 0


# ---------------------------------------------------------------------------
# 2. Builder: binned-SAH top-down split, in-place index partitioning
# ---------------------------------------------------------------------------

class BVHBuilder:
    def __init__(self, triangles, max_leaf_size=4, sah_bins=12):
        """triangles: (N,3,3) float array -- N triangles, 3 verts, 3 coords."""
        self.tris = np.asarray(triangles, dtype=np.float64)
        self.centroids = self.tris.mean(axis=1)                 # (N,3)
        self.prim_bmin = self.tris.min(axis=1)                  # (N,3)
        self.prim_bmax = self.tris.max(axis=1)                  # (N,3)
        self.indices = np.arange(len(self.tris))                # permuted during build
        self.max_leaf_size = max_leaf_size
        self.sah_bins = sah_bins

        # stats the earlier implementation was missing:
        self.max_depth = 0
        self.node_count = 0
        self.leaf_count = 0

    # -- bounds helpers ------------------------------------------------

    def _bounds_for_range(self, start, end):
        idx = self.indices[start:end]
        return AABB(self.prim_bmin[idx].min(axis=0), self.prim_bmax[idx].max(axis=0))

    def _centroid_bounds_for_range(self, start, end):
        c = self.centroids[self.indices[start:end]]
        return c.min(axis=0), c.max(axis=0)

    # -- public entry point ---------------------------------------------

    def build(self):
        root = BVHNode()
        self._build_recursive(root, 0, len(self.indices), depth=0)
        return root

    # -- recursive split --------------------------------------------------

    def _build_recursive(self, node, start, end, depth):
        self.node_count += 1
        self.max_depth = max(self.max_depth, depth)
        node.depth = depth
        node.bounds = self._bounds_for_range(start, end)
        count = end - start

        if count <= self.max_leaf_size or depth >= 32:
            node.first_prim, node.prim_count = start, count
            self.leaf_count += 1
            return

        split = self._find_sah_split(start, end)
        if split is None:                       # SAH says splitting doesn't pay off
            node.first_prim, node.prim_count = start, count
            self.leaf_count += 1
            return

        _, _, mid = split
        node.left, node.right = BVHNode(), BVHNode()
        self._build_recursive(node.left, start, mid, depth + 1)
        self._build_recursive(node.right, mid, end, depth + 1)

    # -- binned SAH split search -----------------------------------------

    def _find_sah_split(self, start, end):
        idx = self.indices[start:end]
        cmin, cmax = self._centroid_bounds_for_range(start, end)
        extent = cmax - cmin
        axis = int(np.argmax(extent))
        if extent[axis] < 1e-12:
            return None

        bins = self.sah_bins
        bin_ids = np.clip(
            ((self.centroids[idx, axis] - cmin[axis]) / extent[axis] * bins).astype(int),
            0, bins - 1,
        )

        bin_bmin = np.full((bins, 3), np.inf)
        bin_bmax = np.full((bins, 3), -np.inf)
        bin_count = np.zeros(bins, dtype=int)
        for b in range(bins):
            mask = bin_ids == b
            if not mask.any():
                continue
            prims = idx[mask]
            bin_bmin[b] = self.prim_bmin[prims].min(axis=0)
            bin_bmax[b] = self.prim_bmax[prims].max(axis=0)
            bin_count[b] = mask.sum()

        def area(bmin, bmax):
            d = np.maximum(bmax - bmin, 0.0)
            return 2.0 * (d[0] * d[1] + d[1] * d[2] + d[2] * d[0])

        left_area = np.zeros(bins); left_count = np.zeros(bins, dtype=int)
        rmin, rmax, rcount = np.full(3, np.inf), np.full(3, -np.inf), 0
        for b in range(bins):
            if bin_count[b] > 0:
                rmin = np.minimum(rmin, bin_bmin[b]); rmax = np.maximum(rmax, bin_bmax[b])
                rcount += bin_count[b]
            left_area[b] = area(rmin, rmax) if rcount else 0.0
            left_count[b] = rcount

        right_area = np.zeros(bins); right_count = np.zeros(bins, dtype=int)
        rmin, rmax, rcount = np.full(3, np.inf), np.full(3, -np.inf), 0
        for b in range(bins - 1, -1, -1):
            if bin_count[b] > 0:
                rmin = np.minimum(rmin, bin_bmin[b]); rmax = np.maximum(rmax, bin_bmax[b])
                rcount += bin_count[b]
            right_area[b] = area(rmin, rmax) if rcount else 0.0
            right_count[b] = rcount

        best_cost, best_bin = np.inf, None
        for b in range(bins - 1):
            lc, la = left_count[b], left_area[b]
            rc, ra = right_count[b + 1], right_area[b + 1]
            if lc == 0 or rc == 0:
                continue
            cost = la * lc + ra * rc
            if cost < best_cost:
                best_cost, best_bin = cost, b

        parent_cost = self._bounds_for_range(start, end).surface_area() * (end - start)
        if best_bin is None or best_cost >= parent_cost:
            return None   # leaf is cheaper than any split -> don't split

        threshold = cmin[axis] + (best_bin + 1) * (extent[axis] / bins)

        # Hoare-style in-place partition of self.indices[start:end] -- this
        # is exactly the array-partitioning step a C++ builder does with
        # std::partition; here it's explicit so it's easy to follow.
        arr = self.indices
        lo, hi = start, end - 1
        while lo <= hi:
            if self.centroids[arr[lo], axis] < threshold:
                lo += 1
            else:
                arr[lo], arr[hi] = arr[hi], arr[lo]
                hi -= 1
        mid = lo
        if mid == start or mid == end:
            return None   # degenerate partition -> leaf
        return axis, threshold, mid


# ---------------------------------------------------------------------------
# 3. Flatten: left child implicit (idx+1), right child stored explicitly
# ---------------------------------------------------------------------------

def flatten_bvh(root, node_count):
    bmin = np.zeros((node_count, 4), dtype=np.float32)
    bmax = np.zeros((node_count, 4), dtype=np.float32)
    meta = np.zeros((node_count, 4), dtype=np.int32)   # x=right/firstPrim y=triCount z,w unused

    counter = [0]

    def emit(node):
        idx = counter[0]
        counter[0] += 1
        bmin[idx, :3] = node.bounds.bmin
        bmax[idx, :3] = node.bounds.bmax
        if node.is_leaf:
            meta[idx] = [node.first_prim, node.prim_count, 0, 0]
        else:
            slot = idx
            emit(node.left)                 # always lands at idx+1
            right_idx = counter[0]
            emit(node.right)
            meta[slot] = [right_idx, 0, 0, 0]
        return idx

    emit(root)
    return bmin, bmax, meta


# ---------------------------------------------------------------------------
# 4. Reference traversal (pure Python) -- mirrors the GLSL algorithm exactly,
#    so it doubles as a correctness spec before any GLSL gets written.
# ---------------------------------------------------------------------------

def ray_aabb(ro, inv_rd, bmin, bmax):
    t0 = (bmin - ro) * inv_rd
    t1 = (bmax - ro) * inv_rd
    tsm, tbg = np.minimum(t0, t1), np.maximum(t0, t1)
    tnear, tfar = tsm.max(), tbg.min()
    return (tfar >= max(tnear, 0.0)), tnear


def intersect_tri(ro, rd, v0, v1, v2):
    e1, e2 = v1 - v0, v2 - v0
    p = np.cross(rd, e2)
    det = np.dot(e1, p)
    if abs(det) < 1e-10:
        return None
    inv_det = 1.0 / det
    tv = ro - v0
    u = np.dot(tv, p) * inv_det
    if u < 0.0 or u > 1.0:
        return None
    q = np.cross(tv, e1)
    v = np.dot(rd, q) * inv_det
    if v < 0.0 or u + v > 1.0:
        return None
    t = np.dot(e2, q) * inv_det
    return t if t > 1e-6 else None


def flat_bvh_trace(bmin, bmax, meta, tri_indices, tris, ro, rd):
    """Iterative stack traversal over the FLAT arrays -- this is the exact
    logic the GLSL shader implements, kept in Python so it can be checked
    before it ever touches ModernGL."""
    inv_rd = 1.0 / rd
    stack = [0]
    closest_t, hit_tri = np.inf, -1
    nodes_visited = 0

    while stack:
        node_idx = stack.pop()
        nodes_visited += 1
        hit, tnear = ray_aabb(ro, inv_rd, bmin[node_idx, :3], bmax[node_idx, :3])
        if not hit or tnear > closest_t:
            continue

        tri_count = meta[node_idx, 1]
        if tri_count > 0:
            first = meta[node_idx, 0]
            for i in range(tri_count):
                ti = tri_indices[first + i]
                t = intersect_tri(ro, rd, *tris[ti])
                if t is not None and t < closest_t:
                    closest_t, hit_tri = t, ti
        else:
            left_idx = node_idx + 1
            right_idx = meta[node_idx, 0]
            stack.append(left_idx)
            stack.append(right_idx)

    return (hit_tri, closest_t, nodes_visited) if hit_tri >= 0 else (-1, np.inf, nodes_visited)


def brute_force_trace(tris, ro, rd):
    closest_t, hit_tri = np.inf, -1
    for i, (v0, v1, v2) in enumerate(tris):
        t = intersect_tri(ro, rd, v0, v1, v2)
        if t is not None and t < closest_t:
            closest_t, hit_tri = t, i
    return hit_tri, closest_t


# ---------------------------------------------------------------------------
# 5. Test mesh: procedural bumpy grid (no external mesh dependency needed)
# ---------------------------------------------------------------------------

def make_bumpy_grid(n=24, extent=10.0, seed=0):
    rng = np.random.default_rng(seed)
    xs = np.linspace(-extent, extent, n)
    zs = np.linspace(-extent, extent, n)
    heights = rng.normal(0, 0.6, size=(n, n))
    # smooth it a bit so it looks like terrain, not noise
    for _ in range(2):
        heights = (heights + np.roll(heights, 1, 0) + np.roll(heights, -1, 0)
                   + np.roll(heights, 1, 1) + np.roll(heights, -1, 1)) / 5.0

    verts = np.zeros((n, n, 3))
    for i in range(n):
        for j in range(n):
            verts[i, j] = [xs[i], heights[i, j], zs[j]]

    tris = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = verts[i, j], verts[i + 1, j], verts[i, j + 1], verts[i + 1, j + 1]
            tris.append((a, b, c))
            tris.append((b, d, c))
    return np.array(tris, dtype=np.float64)


# ---------------------------------------------------------------------------
# 6. Demo / verification
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tris = make_bumpy_grid(n=24)
    print(f"Test mesh: {len(tris)} triangles\n")

    builder = BVHBuilder(tris, max_leaf_size=4, sah_bins=12)
    root = builder.build()
    print("Build stats:")
    print(f"  node_count : {builder.node_count}")
    print(f"  leaf_count : {builder.leaf_count}")
    print(f"  max_depth  : {builder.max_depth}")
    print(f"  avg leaf size: {len(tris) / builder.leaf_count:.2f}\n")

    bmin, bmax, meta = flatten_bvh(root, builder.node_count)
    tri_indices = builder.indices.copy()   # already permuted in place by build()

    rng = np.random.default_rng(1)
    n_rays = 500
    mismatches = 0
    total_nodes_visited = 0

    for _ in range(n_rays):
        ro = rng.uniform(-15, 15, size=3)
        ro[1] = 15.0
        target = rng.uniform(-8, 8, size=3)
        target[1] = 0.0
        rd = target - ro
        rd /= np.linalg.norm(rd)

        bvh_tri, bvh_t, nodes_visited = flat_bvh_trace(bmin, bmax, meta, tri_indices, tris, ro, rd)
        ref_tri, ref_t = brute_force_trace(tris, ro, rd)
        total_nodes_visited += nodes_visited

        same_hit = (bvh_tri == -1 and ref_tri == -1) or (
            bvh_tri != -1 and ref_tri != -1 and abs(bvh_t - ref_t) < 1e-6
        )
        if not same_hit:
            mismatches += 1
            print(f"  MISMATCH: bvh=({bvh_tri},{bvh_t:.4f}) ref=({ref_tri},{ref_t:.4f})")

    print(f"Traced {n_rays} rays against {len(tris)} triangles:")
    print(f"  mismatches vs brute force : {mismatches}")
    print(f"  avg BVH nodes visited/ray : {total_nodes_visited / n_rays:.1f}")
    print(f"  brute-force tris tested/ray: {len(tris)} (always)")
    print(f"  -> BVH visits ~{100 * (total_nodes_visited / n_rays) / len(tris):.1f}% "
          f"as many nodes as brute force has triangles")

    assert mismatches == 0, "flat BVH traversal disagrees with brute-force reference!"
    print("\nAll rays agree with brute-force reference. Flattened BVH is correct.")
