"""FMM low-level controller -- navigation_module.py-style, global-map planner.

A drop-in replacement for ``ShortestPathFollower`` (same
``get_next_action(goal_world, goal_radius=None) -> action`` interface), built after
``navigation_module.py`` (``RealNavigatorModule``):

  * plans over a *global* occupancy map (here from the navmesh top-down view --
    see ``GlobalOccupancyMap``),
  * computes the skfmm geodesic distance field from the goal over free space,
  * snaps a goal/agent that lands on an obstacle to the nearest free cell,
  * and -- the key navigation_module idea -- biases the path toward the medial
    axis (room/corridor centers, away from walls) so the agent doesn't hug
    obstacles and wedge.

Improvements over the naive original: the medial-axis bias is a clean,
continuous clearance term blended into the short-term-goal score (instead of the
skeleton-distance hack), and the "stuck" escape is deterministic (report status
so the agent reroutes / scans) rather than emitting random actions.

The global map is static, so the field is recomputed only when the goal changes.
"""
import numpy as np
from numpy import ma
import skfmm

from .global_occupancy_map import GlobalOccupancyMap
from .geometry import heading_from_rotvec

# Action-id map. Prefer Habitat's ids so they match the sim when habitat IS present;
# otherwise fall back to the canonical {stop, forward, left, right} = {0,1,2,3} that
# the localizer and the real-robot wrapper use. This lets the controller be imported
# WITHOUT habitat (real-robot deployment).
try:
    from habitat.sims.habitat_simulator.actions import HabitatSimActions as _HSA
    _ACTIONS = {k: int(_HSA[k]) for k in ("stop", "move_forward", "turn_left", "turn_right")}
except Exception:
    _ACTIONS = {"stop": 0, "move_forward": 1, "turn_left": 2, "turn_right": 3}


def _wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def _disk_mask(radius):
    size = 2 * radius + 1
    yy, xx = np.mgrid[0:size, 0:size]
    return ((yy - radius) ** 2 + (xx - radius) ** 2 <= radius ** 2)


class FMMController:
    def __init__(
        self,
        sim,
        goal_radius,
        *,
        resolution_m=0.05,
        agent_radius=0.18,
        turn_angle=30.0,
        forward_step_size=0.25,
        step_cells=7,
        clearance_weight=1.25,
        clearance_cap_m=0.75,
        clearance_relax_m=1.0,
        turn_tolerance_deg=None,
    ):
        self.sim = sim
        self.goal_radius = float(goal_radius)
        self.forward_step_size = float(forward_step_size)
        # Half a discrete turn is the nearest-action boundary; add 5 deg of slack
        # so small target/pose jitter around that boundary cannot flip left/right.
        self.turn_tolerance_deg = float(
            turn_angle / 2.0 + 5.0 if turn_tolerance_deg is None
            else turn_tolerance_deg
        )
        self.step_cells = int(step_cells)
        self.clearance_weight = float(clearance_weight)
        self.clearance_cap = float(clearance_cap_m) / float(resolution_m)  # in cells
        self.clearance_relax = float(clearance_relax_m) / float(resolution_m)  # in cells

        self.map = GlobalOccupancyMap(resolution_m, agent_radius)
        self.A = dict(_ACTIONS)
        self._goal_key = None
        self.goal_mask = None
        self.fmm_dist = None
        self.last_status = "moving"
        self._active_stg_cell = None
        self._last_planner_action = None
        # Visualization-only record of the short-term goal selected on the most
        # recent control step.  It never feeds back into planning or control.
        self.last_stg_world = None
        # Optional pose source (observation-based localizer). None -> oracle sim pose.
        self.pose_source = None

    # ------------------------------------------------------------------ #
    def set_pose_source(self, pose_source):
        """Attach an object exposing pose() -> (position_xyz, heading_rad)."""
        self.pose_source = pose_source

    # ------------------------------------------------------------------ #
    def reset(self, start_position, occ_map=None):
        # Use a preloaded global occupancy map (e.g. built from a v4 collection)
        # if given; otherwise build one from the navmesh for the current floor.
        if occ_map is not None:
            self.map = occ_map
        else:
            self.map.reset(self.sim, float(start_position[1]))
        self._goal_key = None
        self.goal_mask = None
        self.fmm_dist = None
        self.last_status = "moving"
        self.last_stg_world = None
        self._active_stg_cell = None
        self._last_planner_action = None

    # ------------------------------------------------------------------ #
    def _goal_cell(self, goal_world):
        # Every other world->planner-cell conversion uses nearest-cell rounding.
        # Truncation here could map a ray-validated goal into a different blocked
        # cell, then snap the FMM seed elsewhere in response to centimetre jitter.
        gr, gc = np.rint(self.map.point_to_grid(goal_world)).astype(int)
        gr = int(np.clip(gr, 0, self.map.H - 1))
        gc = int(np.clip(gc, 0, self.map.W - 1))
        trav_mask = getattr(self.map, "free_seen", self.map.free)
        if not trav_mask[gr, gc]:
            gr, gc = self.map.nearest_free(gr, gc)
        return gr, gc

    def _set_goal(self, goal_world, goal_cell=None):
        gr, gc = self._goal_cell(goal_world) if goal_cell is None else goal_cell

        # traversable = radius-eroded free AND observed; unseen exterior stays masked
        trav_mask = getattr(self.map, "free_seen", self.map.free)
        trav = ma.masked_values(trav_mask.astype(float), 0.0)  # blocked -> masked
        trav[gr, gc] = 0.0
        dd = skfmm.distance(trav, dx=1)
        self.fmm_dist = ma.filled(dd, np.max(dd) + 1.0)
        self._goal_key = (gr, gc)
        self.goal_mask = None
        self._active_stg_cell = None
        self._last_planner_action = None

    def _set_goal_mask(self, goal_mask):
        """Build one distance field whose zero set is every safe goal cell."""
        trav_mask = getattr(self.map, "free_seen", self.map.free)
        sources = np.asarray(goal_mask, dtype=bool)
        if sources.shape != trav_mask.shape:
            raise ValueError(
                f"goal_mask shape {sources.shape} != map shape {trav_mask.shape}")
        sources = sources & trav_mask
        if not sources.any():
            self.goal_mask = sources
            self.fmm_dist = None
            self._goal_key = ("mask",)
            self._active_stg_cell = None
            self._last_planner_action = None
            return False

        trav = ma.masked_values(trav_mask.astype(float), 0.0)
        trav[sources] = 0.0
        dd = skfmm.distance(trav, dx=1)
        self.fmm_dist = ma.filled(dd, np.max(dd) + 1.0)
        self.goal_mask = sources.copy()
        self._goal_key = ("mask",)
        self._active_stg_cell = None
        self._last_planner_action = None
        return True

    @staticmethod
    def _key(goal_world):
        return (round(float(goal_world[0]), 2), round(float(goal_world[2]), 2))

    # ------------------------------------------------------------------ #
    def get_next_action(self, goal_world, goal_radius=None, goal_mask=None):
        """Return the next discrete action toward ``goal_world``.

        ``goal_radius`` overrides the controller default for this call.  This lets
        topology navigation use a tighter node-arrival boundary without changing
        the last-mile stopping distance owned by the same controller instance.

        ``goal_mask`` switches to multi-source planning: every True cell is a
        terminal goal, and arrival means entering that mask. ``goal_radius`` is
        then not applied again around the individual source cells.
        """
        mask_mode = goal_mask is not None
        active_goal_radius = (0.0 if mask_mode else
                              self.goal_radius if goal_radius is None
                              else float(goal_radius))
        # A stopped/failed planning step has no active STG.  Set this again only
        # after _short_term_goal successfully selects one below.
        self.last_stg_world = None
        # Pose comes from the oracle sim state unless a pose source (e.g. an
        # observation-based localizer) was attached via set_pose_source().
        if self.pose_source is not None:
            pos, heading_rad = self.pose_source.pose()
            pos = np.asarray(pos, dtype=np.float64)
        else:
            import quaternion as q          # lazy: only the sim-oracle path needs it
            state = self.sim.get_agent_state()
            pos = np.asarray(state.position, dtype=np.float64)
            heading_rad = heading_from_rotvec(q.as_rotation_vector(state.rotation))

        state_rc = self.map.point_to_grid(pos)
        r = int(np.clip(round(state_rc[0]), 0, self.map.H - 1))
        c = int(np.clip(round(state_rc[1]), 0, self.map.W - 1))

        if not mask_mode:
            gx, gz = float(goal_world[0]), float(goal_world[2])
            if np.hypot(gx - pos[0], gz - pos[2]) <= active_goal_radius:
                self.last_status = "arrived"
                self._active_stg_cell = None
                self._last_planner_action = self.A["stop"]
                return self.A["stop"]

        if mask_mode:
            trav_mask = getattr(self.map, "free_seen", self.map.free)
            sources = np.asarray(goal_mask, dtype=bool)
            if sources.shape != trav_mask.shape:
                raise ValueError(
                    f"goal_mask shape {sources.shape} != map shape {trav_mask.shape}")
            sources = sources & trav_mask
            if sources[r, c]:
                self.goal_mask = sources.copy()
                self.last_status = "arrived"
                self._active_stg_cell = None
                self._last_planner_action = self.A["stop"]
                return self.A["stop"]
            if not sources.any():
                self.goal_mask = sources
                self.fmm_dist = None
                self.last_status = "stuck"
                self._active_stg_cell = None
                self._last_planner_action = self.A["stop"]
                return self.A["stop"]

            changed = (self.goal_mask is None or
                       not np.array_equal(sources, self.goal_mask))
            if changed or self.fmm_dist is None:
                previous_stg = self._active_stg_cell
                previous_action = self._last_planner_action
                was_turning = previous_action in (
                    self.A["turn_left"], self.A["turn_right"])
                self._set_goal_mask(sources)
                if was_turning and self._stg_is_valid_from(r, c, previous_stg):
                    self._active_stg_cell = previous_stg
                    self._last_planner_action = previous_action

        # Recompute only when the RESOLVED goal cell changes. Last-mile target EMA
        # often moves by centimetres while remaining in one map cell; rebuilding the
        # same field for that sub-cell jitter only destabilizes STG selection.
        #
        # If a field update happens while the robot is still turning toward an STG,
        # keep that STG when it remains safe and descends the *new* field. Otherwise
        # equally good clearance candidates on opposite sides can alternate with
        # every depth/segmentation update, producing a left-right oscillation even
        # though the target is directly reachable.
        key = None if mask_mode else self._goal_cell(goal_world)
        if not mask_mode and (key != self._goal_key or self.fmm_dist is None):
            previous_stg = self._active_stg_cell
            previous_action = self._last_planner_action
            was_turning = previous_action in (
                self.A["turn_left"], self.A["turn_right"])
            self._set_goal(goal_world, goal_cell=key)
            if was_turning and self._stg_is_valid_from(r, c, previous_stg):
                self._active_stg_cell = previous_stg
                self._last_planner_action = previous_action

        geo_m = float(self.fmm_dist[r, c]) * self.map.res
        if not mask_mode and geo_m <= active_goal_radius:
            self.last_status = "arrived"
            self._active_stg_cell = None
            self._last_planner_action = self.A["stop"]
            return self.A["stop"]

        turning = self._last_planner_action in (
            self.A["turn_left"], self.A["turn_right"])
        if turning and self._stg_is_valid_from(r, c, self._active_stg_cell):
            stg, replan = self._active_stg_cell, False
        else:
            if mask_mode:
                stg, replan = self._short_term_goal(
                    r, c, active_goal_radius, goal_mask=self.goal_mask)
            else:
                stg, replan = self._short_term_goal(r, c, active_goal_radius)
            self._active_stg_cell = None if replan else stg
        if replan:
            self.last_status = "stuck"
            self._last_planner_action = self.A["stop"]
            return self.A["stop"]

        tx, tz = self.map.grid_to_world_xz(stg[0], stg[1])
        self.last_stg_world = np.array([tx, float(pos[1]), tz], dtype=np.float64)
        self.last_status = "moving"
        action = self._action_toward(pos, heading_rad, tx, tz)
        self._last_planner_action = action
        return action

    def _stg_is_valid_from(self, r, c, stg):
        """A turn-phase STG may be reused while it stays safe and descends the field."""
        if stg is None or self.fmm_dist is None:
            return False
        sr, sc = int(stg[0]), int(stg[1])
        if not (0 <= sr < self.map.H and 0 <= sc < self.map.W):
            return False
        if not float(self.fmm_dist[sr, sc]) < float(self.fmm_dist[r, c]):
            return False
        return self._segment_is_traversable(r, c, sr, sc)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _supercover_cells(r0, c0, r1, c1):
        """Grid cells touched by the centre-to-centre segment, including both
        neighbours when it passes exactly through a cell corner."""
        r, c = int(r0), int(c0)
        r1, c1 = int(r1), int(c1)
        dr, dc = r1 - r, c1 - c
        nr, nc = abs(dr), abs(dc)
        sr = 0 if dr == 0 else (1 if dr > 0 else -1)
        sc = 0 if dc == 0 else (1 if dc > 0 else -1)
        ir = ic = 0
        cells = [(r, c)]

        while ir < nr or ic < nc:
            # Compare the next horizontal/vertical grid-boundary crossings
            # without floating-point division.
            decision = (1 + 2 * ic) * nr - (1 + 2 * ir) * nc
            if decision == 0:
                # Exact corner crossing: a radius-padded body must be clear on
                # both sides of the corner, not only in the diagonal cell.
                if ic < nc:
                    cells.append((r, c + sc))
                if ir < nr:
                    cells.append((r + sr, c))
                c += sc
                r += sr
                ic += 1
                ir += 1
            elif decision < 0:
                c += sc
                ic += 1
            else:
                r += sr
                ir += 1
            cells.append((r, c))
        return cells

    def _segment_is_traversable(self, r0, c0, r1, c1):
        """Whether the full agent-centre segment stays in radius-padded free space."""
        cells = self._supercover_cells(r0, c0, r1, c1)
        rr, cc = zip(*cells)
        traversable = getattr(self.map, "free_seen", self.map.free)
        return bool(np.all(traversable[np.asarray(rr), np.asarray(cc)]))

    # ------------------------------------------------------------------ #
    def _short_term_goal(self, r, c, goal_radius=None, goal_mask=None):
        """Within a step-radius disk, pick the cell that reduces the geodesic
        distance to the goal while preferring clearance (medial-axis bias)."""
        active_goal_radius = (
            self.goal_radius if goal_radius is None else float(goal_radius)
        )
        du = self.step_cells
        big = float(self.fmm_dist.shape[0] ** 2)

        dist = np.pad(self.fmm_dist, du, "constant", constant_values=big)
        clr = np.pad(self.map.clearance, du, "constant", constant_values=0.0)
        sub = dist[r:r + 2 * du + 1, c:c + 2 * du + 1]
        subc = clr[r:r + 2 * du + 1, c:c + 2 * du + 1]

        disk = _disk_mask(du)
        cur = float(self.fmm_dist[r, c])
        improve = (sub < cur) & disk          # must make progress toward goal
        # The FMM/clearance score describes only each candidate endpoint. The
        # low-level controller drives straight toward that endpoint, so reject a
        # candidate when the segment cuts across radius-padded blocked space.
        for sr, sc in np.argwhere(improve):
            tr, tc = int(sr + r - du), int(sc + c - du)
            if not self._segment_is_traversable(r, c, tr, tc):
                improve[sr, sc] = False
        # Centerline (clearance) bias, faded over the ACTIVE approach interval:
        # zero at the controller's stop boundary and full at clearance_relax. The
        # former cur/clearance_relax formula still retained half the clearance
        # reward at a 0.5 m stop boundary, so skeleton and goal candidates could
        # keep trading places until the instant STOP fired.
        stop_cells = active_goal_radius / float(self.map.res)
        fade_span = self.clearance_relax - stop_cells
        if fade_span > 0.0:
            relax = float(np.clip((cur - stop_cells) / fade_span, 0.0, 1.0))
        else:
            relax = 1.0 if cur > stop_cells else 0.0
        w = self.clearance_weight * relax
        # Lower score = better: closer to goal, but rewarded for staying clear of
        # walls (clearance capped so wide-open rooms don't over-dominate).
        score = sub - w * np.minimum(subc, self.clearance_cap)
        score = np.where(improve, score, big)

        # A multi-source field is flat (zero) throughout its terminal region.
        # Once that region enters the local horizon, stop at its first safely
        # reachable cell instead of letting clearance pull the STG deeper into it.
        if goal_mask is not None:
            terminal = np.pad(np.asarray(goal_mask, dtype=bool), du,
                              "constant", constant_values=False)
            terminal = terminal[r:r + 2 * du + 1, c:c + 2 * du + 1]
            entries = improve & terminal
            if entries.any():
                yy, xx = np.mgrid[-du:du + 1, -du:du + 1]
                d2 = yy * yy + xx * xx
                nearest = float(np.min(d2[entries]))
                entry_score = np.where(entries & (d2 == nearest), score, big)
                sr, sc = np.unravel_index(int(np.argmin(entry_score)), score.shape)
                return (sr + r - du, sc + c - du), False

        sr, sc = np.unravel_index(int(np.argmin(score)), score.shape)
        replan = not improve.any()
        return (sr + r - du, sc + c - du), replan

    # ------------------------------------------------------------------ #
    def _action_toward(self, pos, heading_rad, target_x, target_z):
        desired = np.degrees(np.arctan2(target_x - pos[0], -(target_z - pos[2])))
        current = np.degrees(heading_rad)
        delta = _wrap_deg(desired - current)
        if abs(delta) <= self.turn_tolerance_deg:
            return self.A["move_forward"]
        return self.A["turn_right"] if delta > 0 else self.A["turn_left"]
