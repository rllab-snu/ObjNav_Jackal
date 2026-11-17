"""Binary occupancy map built from depth point clouds (occ_map.py-style).

A clean reimplementation of the MapBuilder idea collected in ``occ_map.py``: as
the agent explores, back-project each depth frame to a world point cloud, drop
points whose height above the floor falls in the obstacle band, and accumulate
two top-down boolean grids:

  * ``occ``  (obstacle map): a cell saw geometry in the obstacle height band
    [obstacle_height_min, obstacle_height_max] (above the floor band, below the
    head -> excludes floor and ceiling/overhead).
  * ``seen`` (explored map): a cell was observed at all (floor included). So
    free-but-explored = ``seen & ~occ``.

Both are stored as 0/1 (uint8). The save format
(``{'obstacle_map', 'seen_map', 'map_params'}``) matches what
``navigation_module.py`` loads.

Depth validity: the sensor clamps no-return pixels to ``min_depth`` / ``max_depth``,
so only ``min_depth + eps < depth < max_depth - eps`` is treated as valid.

World->grid (global frame sized to the navmesh bounds): col follows world +x,
row follows world +z, origin at the navmesh lower bound.
"""
import numpy as np

from navigation.geometry import backproject_depth_image


class OccupancyMap:
    def __init__(
        self,
        resolution_m=0.05,
        obstacle_height_min=0.2,
        obstacle_height_max=0.88,
        min_depth=0.0,
        max_depth=10.0,
        depth_eps=0.05,
        depth_subsample=2,
    ):
        self.res = resolution_m
        self.obs_hmin = obstacle_height_min
        self.obs_hmax = obstacle_height_max
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.eps = depth_eps
        self.subsample = depth_subsample
        # bounds / grid set in reset()
        self.min_x = self.min_z = 0.0
        self.max_x = self.max_z = 0.0
        self.H = self.W = 0
        self.floor_height = None   # world y of this floor (for 3D alignment)
        self.occ = None
        self.seen = None

    # ------------------------------------------------------------------ #
    def reset(self, sim, floor_height=None):
        lo, hi = sim.pathfinder.get_bounds()
        self.min_x = float(lo[0])
        self.min_z = float(lo[2])
        self.max_x = float(hi[0])
        self.max_z = float(hi[2])
        self.floor_height = None if floor_height is None else float(floor_height)
        self.W = int(np.ceil((self.max_x - self.min_x) / self.res)) + 1
        self.H = int(np.ceil((self.max_z - self.min_z) / self.res)) + 1
        self.occ = np.zeros((self.H, self.W), dtype=bool)
        self.seen = np.zeros((self.H, self.W), dtype=bool)

    # ------------------------------------------------------------------ #
    def update(self, depth_m, camera_pos, cam_rotvec, fx, fy, cx, cy, floor_y):
        """Accumulate one depth frame. depth_m in meters; floor_y = agent base y."""
        d = np.asarray(depth_m, dtype=np.float64).copy()
        # Drop clamped/invalid returns: keep only strictly-between min and max.
        invalid = (d <= self.min_depth + self.eps) | (d >= self.max_depth - self.eps)
        d[invalid] = 0.0  # backproject drops z <= 1e-4

        pts, _, _ = backproject_depth_image(
            d, fx, fy, cx, cy, camera_pos, cam_rotvec,
            subsample=self.subsample, z_max=self.max_depth,
        )
        if len(pts) == 0:
            return

        h = pts[:, 1] - float(floor_y)
        cols = np.round((pts[:, 0] - self.min_x) / self.res).astype(np.int64)
        rows = np.round((pts[:, 2] - self.min_z) / self.res).astype(np.int64)
        inb = (rows >= 0) & (rows < self.H) & (cols >= 0) & (cols < self.W)

        # Explored: anything observed up to head height (exclude high ceiling).
        seen_m = inb & (h < self.obs_hmax)
        self.seen[rows[seen_m], cols[seen_m]] = True

        # Obstacle: geometry in the obstacle height band.
        occ_m = inb & (h > self.obs_hmin) & (h < self.obs_hmax)
        self.occ[rows[occ_m], cols[occ_m]] = True

    # ------------------------------------------------------------------ #
    # Grid <-> world helpers (row follows world +z, col follows world +x).
    def point_to_grid(self, world_pt):
        col = (float(world_pt[0]) - self.min_x) / self.res
        row = (float(world_pt[2]) - self.min_z) / self.res
        return np.array([row, col])

    def grid_to_world_xz(self, row, col):
        x = col * self.res + self.min_x
        z = row * self.res + self.min_z
        return x, z

    def to_dict(self):
        """occ/seen grids + all metadata needed to reload and align to the world.

        Alignment (cell centers): world (x, z) of grid cell (row, col) is
            x = col * resolution_m + origin_world_xz[0]
            z = row * resolution_m + origin_world_xz[1]
        and the inverse:
            col = (x - origin_world_xz[0]) / resolution_m
            row = (z - origin_world_xz[1]) / resolution_m
        """
        return {
            "obstacle_map": self.occ.astype(np.uint8),
            "seen_map": self.seen.astype(np.uint8),
            "map_params": {
                "resolution_m": self.res,                 # meters per cell
                "height": self.H,                         # grid rows (world +z)
                "width": self.W,                          # grid cols (world +x)
                # World coord of grid origin (row=0, col=0):
                "origin_world_xz": [self.min_x, self.min_z],
                # Real-world map extent (x_size, z_size) in meters:
                "world_size_m": [self.W * self.res, self.H * self.res],
                # Full world bounds covered by the map:
                "bounds_world": {
                    "min_x": self.min_x, "max_x": self.max_x,
                    "min_z": self.min_z, "max_z": self.max_z,
                },
                # World y (height) of this floor, for 3D alignment (may be None):
                "floor_height": self.floor_height,
                "grid_convention": "row=+z, col=+x; "
                                   "x=col*res+origin_x, z=row*res+origin_z",
                "obstacle_height_min": self.obs_hmin,
                "obstacle_height_max": self.obs_hmax,
            },
        }

    # ------------------------------------------------------------------ #
    @classmethod
    def from_dict(cls, d):
        """Rebuild a (read-only-usable) map from a saved dict, with grid<->world
        helpers wired up. Use for loading a map back later."""
        mp = d["map_params"]
        m = cls(resolution_m=mp["resolution_m"],
                obstacle_height_min=mp.get("obstacle_height_min", 0.2),
                obstacle_height_max=mp.get("obstacle_height_max", 0.88))
        m.min_x, m.min_z = mp["origin_world_xz"]
        m.max_x = mp.get("bounds_world", {}).get("max_x", m.min_x + mp["width"] * m.res)
        m.max_z = mp.get("bounds_world", {}).get("max_z", m.min_z + mp["height"] * m.res)
        m.H, m.W = int(mp["height"]), int(mp["width"])
        m.floor_height = mp.get("floor_height", None)
        m.occ = np.asarray(d["obstacle_map"]).astype(bool)
        m.seen = np.asarray(d["seen_map"]).astype(bool)
        return m

    def visualize(self):
        """RGB top-down: white=unseen, grey=free(explored), black=obstacle."""
        img = np.full((self.H, self.W, 3), 255, dtype=np.uint8)
        img[self.seen] = (180, 180, 180)
        img[self.occ] = (0, 0, 0)
        return img
