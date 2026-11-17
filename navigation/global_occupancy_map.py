"""Global occupancy map from the sim navmesh top-down view.

The v2 controller follows navigation_module.py's design of planning over a known
*global* occupancy map (there the map is precomputed offline and loaded from
disk). We don't have those files, so we build the equivalent at reset from the
habitat navmesh: ``pathfinder.get_topdown_view(mpp, floor_height)`` returns a
complete boolean navigable grid for the agent's floor. Obstacle = not navigable.

Grid convention (verified against this sim build): the top-down array is indexed
``[row, col]`` with ``row`` spanning world +z and ``col`` spanning world +x, and
the grid origin is the navmesh lower bound. So:
    col = (x - min_x) / res ,  row = (z - min_z) / res .

The map also precomputes, once: the obstacle dilated by the agent radius (the
planner's free space) and a per-cell clearance (distance to the nearest blocked
cell, in cells) used for medial-axis / wall-avoidance biasing.
"""
import numpy as np
import cv2
import skimage.morphology


class GlobalOccupancyMap:
    def __init__(self, resolution_m=0.05, agent_radius_m=0.18):
        self.res = resolution_m
        self.radius_cells = max(1, int(round(agent_radius_m / resolution_m)))
        self._selem = skimage.morphology.disk(self.radius_cells)

    # ------------------------------------------------------------------ #
    def reset(self, sim, floor_height):
        lo, hi = sim.pathfinder.get_bounds()
        self.min_x = float(lo[0])
        self.min_z = float(lo[2])

        td = np.asarray(
            sim.pathfinder.get_topdown_view(self.res, float(floor_height))
        ).astype(bool)                      # [row=z, col=x], True = navigable
        self.navigable = td
        self.H, self.W = td.shape

        # Free space the agent center can occupy: navigable, eroded by the agent
        # radius (== obstacle dilated by the radius). Then clearance = distance
        # to the nearest blocked cell (cells), for center-seeking path bias.
        blocked = cv2.dilate((~td).astype(np.uint8), self._selem).astype(bool)
        self.free = ~blocked
        # navmesh view is fully known -> everything is "seen"
        self.seen = np.ones_like(self.free)
        self.explorable = self.navigable & self.seen    # navigable & seen
        # Planner traversable space: radius-eroded free AND observed. Unseen cells
        # are only optimistically free, so the FMM field / clearance / skeleton must
        # not leak into never-mapped exterior (here seen is all-True, so == free).
        self.free_seen = self.free & self.seen
        self.clearance = cv2.distanceTransform(
            self.free_seen.astype(np.uint8), cv2.DIST_L2, 5
        )

    # ------------------------------------------------------------------ #
    @classmethod
    def from_occupancy_dict(cls, d, agent_radius_m=0.18):
        """Build from a saved occupancy map (collect_occupancy_map.py format:
        {'obstacle_map','seen_map','map_params'}) instead of the navmesh.

        Obstacle = obstacle_map; free = NOT (obstacle dilated by the agent
        radius); unseen cells are treated as free (optimistic). Grid convention
        matches OccupancyMap (row=+z, col=+x; origin = origin_world_xz)."""
        mp = d["map_params"]
        m = cls(resolution_m=mp["resolution_m"], agent_radius_m=agent_radius_m)
        m.min_x, m.min_z = mp["origin_world_xz"]
        m.H, m.W = int(mp["height"]), int(mp["width"])
        obstacle = np.asarray(d["obstacle_map"]).astype(bool)
        m.navigable = ~obstacle
        blocked = cv2.dilate(obstacle.astype(np.uint8), m._selem).astype(bool)
        m.free = ~blocked
        # Cells actually observed while building the map (the rest is unmapped and
        # only optimistically assumed free). Used as a localization sampling mask so
        # particles / likelihood are only evaluated inside the explored region.
        seen = d.get("seen_map")
        m.seen = (np.asarray(seen).astype(bool) if seen is not None
                  else np.ones_like(m.free))
        # explorable = navigable & seen: every observed non-obstacle cell (NOT
        # eroded by the agent radius), used as the sampling mask for particle
        # init / resample and to restrict the candidate cells scored.
        m.explorable = m.navigable & m.seen
        # Planner traversable space: radius-eroded free AND observed. Unseen cells
        # are only optimistically free, so the FMM field / clearance / skeleton must
        # not leak into never-mapped exterior. Clearance is measured on this mask so
        # the wall-avoidance bias also treats the seen/unseen frontier as a wall.
        m.free_seen = m.free & m.seen
        # Optional PLANNING-ONLY keep-out: cells the robot must not enter even though they
        # are physically free/observed (e.g. a doorway to seal off). Subtracted from the
        # planner's traversable space (free_seen + clearance) ONLY -- navigable / seen /
        # explorable (the localizer's masks) are left untouched, so a keep-out never affects
        # localization. Absent -> all-False (existing maps behave exactly as before).
        keepout = d.get("keepout_map")
        m.keepout = (np.asarray(keepout).astype(bool) if keepout is not None
                     else np.zeros_like(m.free))
        m.free_seen = m.free_seen & ~m.keepout
        # Also drop the keep-out from `explorable`: that mask is the localizer's set of
        # CANDIDATE agent positions (particle seeding / re-globalize / global-sweep cells),
        # and the robot can never be inside a keep-out, so a particle there is a bogus
        # hypothesis. This does NOT touch the scan-comparison likelihood (built from
        # navigable / seen / free), so a keep-out cell is still scored as the open floor it
        # physically is -- only the pose PRIOR excludes it.
        m.explorable = m.explorable & ~m.keepout
        m.clearance = cv2.distanceTransform(m.free_seen.astype(np.uint8),
                                            cv2.DIST_L2, 5)
        # Obstacle height band the map was built with -> localizers read this so
        # their depth scan uses the SAME band (else scan points from a different
        # height won't match the mapped geometry).
        m.obstacle_height_min = float(mp.get("obstacle_height_min", 0.2))
        m.obstacle_height_max = float(mp.get("obstacle_height_max", 1.0))
        return m

    # ------------------------------------------------------------------ #
    def point_to_grid(self, world_pt):
        col = (float(world_pt[0]) - self.min_x) / self.res
        row = (float(world_pt[2]) - self.min_z) / self.res
        return np.array([row, col])

    def grid_to_world_xz(self, row, col):
        x = col * self.res + self.min_x
        z = row * self.res + self.min_z
        return x, z

    def in_bounds(self, r, c):
        return 0 <= r < self.H and 0 <= c < self.W

    # -------------------- online (dynamic) occupancy update ------------------ #
    def enable_dynamic(self, occ_thr=0.6, l_occ=0.8, l_free=0.2, l_clamp=4.0,
                       prior=0.5, near_m=0.25, end_margin_m=0.12, max_range_m=5.0,
                       open_margin_m=0.25):
        """Turn on online occupancy updates. Obstacles observed after the localizer
        has converged are fused into the PLANNING layers (free / free_seen /
        clearance) via a log-odds grid, WITHOUT touching navigable / explorable /
        seen -- those stay the static base map the localizer matches against, so
        localization is unaffected ("controller map only"). Base structure is never
        removed; only NEW obstacles (in base-free space) are ever added."""
        self.dynamic = True
        self._base_obstacle = (~self.navigable).copy()
        self.logodds = np.where(self._base_obstacle, prior, -prior).astype(np.float32)
        self.dyn_occ = np.zeros_like(self._base_obstacle)      # NEW obstacles only
        # OPEN space = base-free cells well away from any base wall. Only here does
        # occupancy dominate free (endpoints protected from same-frame carving), so
        # free-standing floor objects get promoted while wall-edge cells (base-map
        # registration offset) keep normal carving and stay thin.
        wm = max(1, int(round(open_margin_m / self.res)))
        near_wall = cv2.dilate(self._base_obstacle.astype(np.uint8),
                               skimage.morphology.disk(wm)).astype(bool)
        self._open = (~self._base_obstacle) & (~near_wall)
        self._dyn = dict(occ_thr=occ_thr, l_occ=l_occ, l_free=l_free, l_clamp=l_clamp,
                         near=near_m, end_margin=end_margin_m, max_range=max_range_m)

    def _xz_to_rc(self, x, z):
        c = np.round((x - self.min_x) / self.res).astype(np.int64)
        r = np.round((z - self.min_z) / self.res).astype(np.int64)
        ok = (r >= 0) & (r < self.H) & (c >= 0) & (c < self.W)
        return r[ok], c[ok]

    def update_dynamic(self, scan_robot_xz, px, pz, pyaw):
        """Fuse one frame's scan (robot-frame [x, z] obstacle points) taken at world
        pose (px, pz, pyaw) into the planning layers. Each ray carves free space up
        to just before its endpoint; endpoints deposit occupied evidence. Returns
        True if the traversable mask changed (-> the controller should replan)."""
        if not getattr(self, "dynamic", False) or scan_robot_xz is None \
                or len(scan_robot_xz) == 0:
            return False
        p = self._dyn
        ca, sa = np.cos(pyaw), np.sin(pyaw)
        sx, sz = scan_robot_xz[:, 0], scan_robot_xz[:, 1]
        ex = px + sx * ca - sz * sa                    # world endpoints (rot2d + origin)
        ez = pz + sx * sa + sz * ca
        rng = np.hypot(ex - px, ez - pz)
        keep = (rng > 1e-3) & (rng <= p["max_range"])
        ex, ez, rng = ex[keep], ez[keep], rng[keep]
        if len(ex) == 0:
            return False
        ux, uz = (ex - px) / rng, (ez - pz) / rng
        lo = self.logodds
        er, ec = self._xz_to_rc(ex, ez)                # occupied endpoints (this frame)
        # free carving: sample each ray from `near` up to just before its endpoint.
        # In OPEN space only, protect this frame's endpoints so rays grazing OVER a
        # free-standing floor object don't carve its footprint free (near walls we
        # keep normal carving so wall-edge cells stay thin).
        S = np.arange(p["near"], float(rng.max()), self.res)
        if S.size:
            valid = S[None, :] < (rng[:, None] - p["end_margin"])
            fr, fc = self._xz_to_rc((px + ux[:, None] * S[None, :])[valid],
                                    (pz + uz[:, None] * S[None, :])[valid])
            if len(fr):
                keep = ~self._open[fr, fc]             # carve ONLY near-wall cells;
                fr, fc = fr[keep], fc[keep]            # open space is never carved, so a
            np.add.at(lo, (fr, fc), -p["l_free"])      # floor object's footprint survives
        np.add.at(lo, (er, ec), p["l_occ"])            # occupied evidence at endpoints
        np.clip(lo, -p["l_clamp"], p["l_clamp"], out=lo)
        # NEW obstacles = base-free cells whose log-odds crossed the threshold.
        new_occ = (~self._base_obstacle) & (lo > p["occ_thr"])
        if np.array_equal(new_occ, self.dyn_occ):
            return False                               # nothing new -> no replan
        self.dyn_occ = new_occ
        plan_obstacle = self._base_obstacle | new_occ  # base structure never removed
        blocked = cv2.dilate(plan_obstacle.astype(np.uint8), self._selem).astype(bool)
        self.free = ~blocked
        self.free_seen = self.free & self.seen
        ko = getattr(self, "keepout", None)               # keep the planning-only keep-out
        if ko is not None:
            self.free_seen = self.free_seen & ~ko
        self.clearance = cv2.distanceTransform(self.free_seen.astype(np.uint8),
                                               cv2.DIST_L2, 5)
        return True

    def nearest_free(self, r, c):
        """Nearest traversable (free & seen) cell to (r, c) (L1), or (r, c) if
        there is none. Snaps a goal/agent that lands on an obstacle or in unseen
        space back onto the observed navigable region the planner uses."""
        fr, fc = np.where(getattr(self, "free_seen", self.free))
        if len(fr) == 0:
            return r, c
        d = np.abs(fr - r) + np.abs(fc - c)
        i = int(np.argmin(d))
        return int(fr[i]), int(fc[i])
