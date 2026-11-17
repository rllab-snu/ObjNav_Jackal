"""Early-localization navigation (pre-global-convergence).

OPT-IN and REVERSIBLE: LastMileFMMAgent only builds an EarlyNav when ``--early_nav`` is on
(the default); with it off the agent is byte-for-byte its old self (no EarlyNav instance,
no behavior change).

Premise (user's): the robot does NOT know its absolute pose while the global MCL is
still multi-modal, but it DOES know its delta-pose (odometry). So we integrate odometry
from the start (origin frame) and accumulate the depth scans into a LOCAL occupancy map.
Before the global filter converges, we pick SAFE actions from that local map instead of
driving blindly on a bogus global estimate.

Policy: SAFETY (hard) + infogain (unknown area a heading reveals) + a persistent frontier
subgoal, plus a LOCAL last-mile (approach + STOP) once the goal is detected pre-convergence.
(Earlier reactive/localmap/infogain-only variants were removed; combo won the ablation.)

step(obs, collided, last_action) -> action id.
"""
import numpy as np


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class _OdoPose:
    """Pose source for FMMController: hands it EarlyNav's integrated ODOMETRY pose
    (origin frame), so the FMM last-mile plans on the same frame the local map lives in
    -- no global convergence needed. heading = oyaw (forward = -z), matching
    FMMController._action_toward's atan2(dx, -dz) convention."""
    def __init__(self, en):
        self.en = en

    def pose(self):
        return np.array([self.en.ox, 0.0, self.en.oz], np.float64), self.en.oyaw


class _LocalMapView:
    """Duck-typed occupancy map over EarlyNav's odometry grids so FMMController can plan
    the local last-mile. Traversable = the radius-eroded complement of the mapped-obstacle
    halo; unobserved cells are OPTIMISTICALLY passable so the planner can route toward a
    goal it has only partly seen. refresh() rebuilds it from the current local map each
    step (the odo map grows/changes, unlike the static global map FMMController assumes)."""
    def __init__(self, en):
        self.en = en
        self.res = en.res
        self.H, self.W = en.H, en.W
        self.free = np.zeros((self.H, self.W), bool)
        self.free_seen = self.free
        self.clearance = np.zeros((self.H, self.W), np.float32)

    def refresh(self):
        from scipy import ndimage
        en = self.en
        obst = en._obst_mask()                            # confirmed obstacles only (noise-gated)
        rad_cells = max(1, int(round(en.rad / en.res)))
        blocked = ndimage.binary_dilation(obst, iterations=rad_cells)
        trav = ~blocked                                   # unknown + free, minus obstacle halo
        self.free = trav
        self.free_seen = trav
        self.clearance = ndimage.distance_transform_edt(trav).astype(np.float32)

    def point_to_grid(self, pos):
        return np.array([round(float(pos[2]) / self.res) + self.en.c0,
                         round(float(pos[0]) / self.res) + self.en.c0], np.float64)

    def grid_to_world_xz(self, r, c):
        return ((c - self.en.c0) * self.res, (r - self.en.c0) * self.res)

    def nearest_free(self, r, c):
        fr, fc = np.where(self.free_seen)
        if len(fr) == 0:
            return r, c
        i = int(np.argmin(np.abs(fr - r) + np.abs(fc - c)))
        return int(fr[i]), int(fc[i])


class EarlyNav:
    def __init__(self, actions, extract_scan, forward_step, turn_angle_deg,
                 res=0.05, half_extent_m=8.0, max_range=5.0, agent_radius=0.18,
                 stop_distance=0.5, fmm_lastmile=False,
                 w_info=1.0, w_sub=1.0, obst_min_hits=2, fwd_clear_margin=0.05,
                 last_mile_veto_off=1.2, goal_ema=0.9, goal_jump_reject=1.5,
                 goal_jump_confirmations=2, goal_jump_consistency=0.5,
                 goal_count_threshold=1, lost_detection_steps=500):
        # local last-mile: once the goal is confidently seen pre-convergence,
        # approach it in the LOCAL map and STOP -- success is by physical pose, so this
        # needs no global convergence.
        self.stop_distance = float(stop_distance)
        # Keep pre-convergence target handling and close-approach semantics aligned with
        # the regular last-mile path.  The planner still differs intentionally: this path
        # has only an odometry-local map and therefore keeps greedy homing by default.
        self.last_mile_veto_off = float(last_mile_veto_off)
        self.goal_ema = float(goal_ema)
        self.goal_jump_reject = float(goal_jump_reject)
        self.goal_jump_confirmations = int(goal_jump_confirmations)
        self.goal_jump_consistency = float(goal_jump_consistency)
        self.goal_count_threshold = int(goal_count_threshold)
        self.lost_detection_steps = int(lost_detection_steps)
        self.fmm_lastmile = bool(fmm_lastmile)  # combo: plan the local last-mile with FMM
        self._fmm = None                        # lazily-built FMMController over the local map
        self._lmv = None                        # local-map view (duck-typed for FMMController)
        self._det_count = 0
        self._lost_detection_count = 0
        self._committed = False               # committed to local last-mile
        self._avoid_turn = None               # committed avoidance turn dir (anti-oscillation)
        self._avoid_count = 0                 # steps spent in the current avoidance maneuver
        self._lm_stall = 0                    # near-goal steps unable to advance (arrival failsafe)
        self._lm_subgoal_odo = None           # last-mile ray-navigable standpoint (odo frame); viz
        self.A = actions
        self.extract_scan = extract_scan          # obs -> (N,2) robot-frame [x=right, z=back]
        self.fwd = float(forward_step)
        self.turn = np.deg2rad(float(turn_angle_deg))
        self.res = float(res)
        self.max_range = float(max_range)
        self.rad = float(agent_radius)
        # odometry pose, integrated from actions (origin frame); matches the localizer's
        # convention: yaw += dyaw; x += d*sin(yaw); z += -d*cos(yaw); forward = -z.
        self.ox = self.oz = self.oyaw = 0.0
        self.goal_odo = None              # DETECTED goal anchored in odo frame (x,z) -> last-mile
        self._goal_jump_candidate = None  # uncommitted large correction in the odometry frame
        self._goal_jump_count = 0
        self._subgoal_yaw = None          # EXPLORE subgoal = a persistent frontier heading
        self._subgoal_age = 0
        self._viz_combo = None            # LAST combo evaluation (per-candidate scores); viz-only
        self.w_info, self.w_sub = float(w_info), float(w_sub)  # combo weights (safety is a hard filter)
        # local occupancy grid (odometry frame), origin at the grid centre
        self.H = self.W = int(round(2 * half_extent_m / res))
        self.c0 = self.H // 2
        self.occ = np.zeros((self.H, self.W), np.int32)   # obstacle hit count
        self.free = np.zeros((self.H, self.W), np.int32)  # free observation count
        # a cell is a CONFIRMED obstacle only after >= obst_min_hits endpoint hits AND
        # more obstacle- than free-through evidence (occ > free); gates single-frame /
        # transient depth noise so one bad range reading never becomes a permanent wall.
        self.obst_min_hits = int(obst_min_hits)
        # forward-clear cone half-width beyond the body radius (m): the live-scan block test
        # vetoes only obstacles within ~body width, so the robot can thread a narrow gap its
        # heading actually fits (the map-based _swept_obstacle still guards the true body).
        self.fwd_clear_margin = float(fwd_clear_margin)
        self._steps = 0
        self._seed_translation_pending = True
        self._seed_forward_issued = False

    def reset_map(self):
        """Wipe odometry + local map + goal/subgoal state (called on re-globalize)."""
        self.ox = self.oz = self.oyaw = 0.0
        self.occ[:] = 0; self.free[:] = 0
        self.goal_odo = None; self._subgoal_yaw = None; self._subgoal_age = 0
        self._goal_jump_candidate = None; self._goal_jump_count = 0
        self._viz_combo = None
        self._det_count = 0; self._lost_detection_count = 0
        self._committed = False; self._steps = 0
        self._seed_translation_pending = True; self._seed_forward_issued = False
        self._avoid_turn = None; self._avoid_count = 0; self._lm_stall = 0
        self._lm_subgoal_odo = None

    # -------------------- odometry + local map -------------------- #
    def _integrate(self, last_action, collided, delta_pose=None):
        # delta_pose=(d_fwd, d_lat, dyaw) = the MEASURED odometry delta (real robot); otherwise
        # the discrete action maps to the nominal motion model (sim / robot-without-odom).
        if delta_pose is not None:
            d_fwd, d_lat, dyaw = (float(v) for v in delta_pose)
        else:
            d_lat = 0.0
            if last_action == self.A["move_forward"]:
                d_fwd, dyaw = (0.0 if collided else self.fwd), 0.0
            elif last_action == self.A["turn_left"]:
                d_fwd, dyaw = 0.0, -self.turn
            elif last_action == self.A["turn_right"]:
                d_fwd, dyaw = 0.0, self.turn
            else:
                d_fwd, dyaw = 0.0, 0.0
        self.oyaw = _wrap(self.oyaw + dyaw)
        self.ox += d_fwd * np.sin(self.oyaw) + d_lat * np.cos(self.oyaw)
        self.oz += -d_fwd * np.cos(self.oyaw) + d_lat * np.sin(self.oyaw)

    def integrate_pending_motion(self, last_action, collided, delta_pose=None):
        """Bring the odometry pose to the current observation timestamp.

        Normally :meth:`step` performs this integration.  On the frame where the
        particle filter first converges, control skips ``step`` and hands the local
        goal to the global planner, so that one pending motion must be applied here.
        """
        if last_action is not None or delta_pose is not None:
            self._integrate(last_action, collided, delta_pose)

    def _to_cell(self, x, z):
        return (np.round(z / self.res).astype(int) + self.c0,
                np.round(x / self.res).astype(int) + self.c0)

    def _update_map(self, scan):
        if scan is None or len(scan) == 0:
            return
        ca, sa = np.cos(self.oyaw), np.sin(self.oyaw)
        wx = scan[:, 0] * ca - scan[:, 1] * sa + self.ox     # odo-frame endpoints
        wz = scan[:, 0] * sa + scan[:, 1] * ca + self.oz
        r, c = self._to_cell(wx, wz)
        ok = (r >= 0) & (r < self.H) & (c >= 0) & (c < self.W)
        np.add.at(self.occ, (r[ok], c[ok]), 1)
        # free along each ray: sample points between the robot and the endpoint
        rng = np.hypot(scan[:, 0], scan[:, 1])
        n = max(2, int(self.max_range / self.res))
        t = np.linspace(0.0, 1.0, n)[None, :]               # (1,n)
        fx = (scan[:, 0][:, None] * t)                       # robot-frame free samples
        fz = (scan[:, 1][:, None] * t)
        # keep only samples strictly before the hit (0.9*range) so we don't erase walls
        keep = (t * rng[:, None]) < (rng[:, None] - self.res)
        wfx = fx * ca - fz * sa + self.ox
        wfz = fx * sa + fz * ca + self.oz
        fr, fc = self._to_cell(wfx[keep], wfz[keep])
        ok2 = (fr >= 0) & (fr < self.H) & (fc >= 0) & (fc < self.W)
        np.add.at(self.free, (fr[ok2], fc[ok2]), 1)

    def _obst_mask(self):
        """Full-grid CONFIRMED-obstacle mask: a cell is an obstacle only after
        >= obst_min_hits endpoint hits AND more obstacle- than free-through evidence
        (occ > free). Rejects single-frame / transient depth noise that later rays
        passed through, so one bad range reading never becomes a permanent wall."""
        return (self.occ >= self.obst_min_hits) & (self.occ > self.free)

    def _is_obstacle(self, r, c):
        return (0 <= r < self.H and 0 <= c < self.W
                and self.occ[r, c] >= self.obst_min_hits
                and self.occ[r, c] > self.free[r, c])

    def _swept_obstacle(self, yaw):
        """Any MAPPED obstacle in the robot's swept forward footprint (radius-wide,
        one step long) at heading `yaw`? Stronger than a 1-cell check -- catches walls
        the narrow forward cone misses but the body would still hit."""
        for d in np.linspace(0.0, self.fwd, 4):
            cx = self.ox + d * np.sin(yaw); cz = self.oz - d * np.cos(yaw)
            for off in np.linspace(-self.rad, self.rad, 5):       # across the body width
                px = cx + off * np.cos(yaw); pz = cz + off * np.sin(yaw)
                r, c = self._to_cell(np.array([px]), np.array([pz]))
                if self._is_obstacle(int(r[0]), int(c[0])):
                    return True
        return False

    # -------------------- clearance from the current scan -------------------- #
    def _forward_clear(self, scan):
        """min forward distance in a ~body-width corridor directly ahead (forward = -z).
        Half-width = rad + fwd_clear_margin (just over the body), so obstacles beside the
        path -- not in it -- don't veto a narrow gap the robot actually fits through."""
        if scan is None or len(scan) == 0:
            return np.inf
        ahead = scan[(np.abs(scan[:, 0]) < self.rad + self.fwd_clear_margin) & (scan[:, 1] < -0.05)]
        return float(np.abs(ahead[:, 1]).min()) if len(ahead) else np.inf

    def _open_turn(self, scan):
        """turn toward the more open side (higher mean range); default left."""
        if scan is None or len(scan) < 3:
            return self.A["turn_left"]
        x, rng = scan[:, 0], np.hypot(scan[:, 0], scan[:, 1])
        left = rng[x < 0].mean() if (x < 0).any() else 0.0
        right = rng[x > 0].mean() if (x > 0).any() else 0.0
        return self.A["turn_left"] if left >= right else self.A["turn_right"]

    def _blocked(self, scan):
        return self._forward_clear(scan) < (self.fwd + self.rad + 0.15)

    # -------------------- info gain (unknown area a heading would reveal) -------------------- #
    def _unknown_ahead(self, yaw):
        """count UNKNOWN (never-observed) cells in a forward cone at heading `yaw`."""
        rs = np.arange(self.res, self.max_range, self.res)
        bs = np.linspace(-0.6, 0.6, 9)                       # ~+-34 deg cone
        cnt = 0
        for b in bs:
            hy = yaw + b
            xs = self.ox + rs * np.sin(hy)
            zs = self.oz - rs * np.cos(hy)
            r, c = self._to_cell(xs, zs)
            ok = (r >= 0) & (r < self.H) & (c >= 0) & (c < self.W)
            r, c = r[ok], c[ok]
            o, fcnt = self.occ[r, c], self.free[r, c]
            unknown = (o == 0) & (fcnt == 0)
            hit = (o >= self.obst_min_hits) & (o > fcnt)   # confirmed obstacle (noise-gated)
            # stop the ray at the first obstacle
            stop = np.argmax(hit) if hit.any() else len(hit)
            cnt += int(unknown[:stop].sum())
        return cnt

    # -------------------- goal anchored in the local (odometry) map -------------------- #
    def _set_goal(self, goal_robot_xz):
        """Anchor/refine a detected goal in the odometry frame.

        Before local last-mile commitment, keep the latest detection just as regular
        last-mile does while counting detections.  Once committed, reject implausible
        jumps and EMA-smooth accepted updates so per-frame mask/depth jitter cannot flip
        the greedy bearing across the discrete turn boundary.
        """
        ca, sa = np.cos(self.oyaw), np.sin(self.oyaw)
        gx = goal_robot_xz[0] * ca - goal_robot_xz[1] * sa + self.ox
        gz = goal_robot_xz[0] * sa + goal_robot_xz[1] * ca + self.oz
        candidate = (float(gx), float(gz))
        if self.goal_odo is None or not self._committed:
            self.goal_odo = candidate
            self._goal_jump_candidate = None; self._goal_jump_count = 0
            return
        jump = float(np.hypot(candidate[0] - self.goal_odo[0],
                              candidate[1] - self.goal_odo[1]))
        if jump <= self.goal_jump_reject:
            a = self.goal_ema
            self.goal_odo = ((1.0 - a) * self.goal_odo[0] + a * candidate[0],
                             (1.0 - a) * self.goal_odo[1] + a * candidate[1])
            self._goal_jump_candidate = None; self._goal_jump_count = 0
            return

        pending = self._goal_jump_candidate
        consistent = (pending is not None and
                      float(np.hypot(candidate[0] - pending[0], candidate[1] - pending[1]))
                      <= self.goal_jump_consistency)
        self._goal_jump_count = self._goal_jump_count + 1 if consistent else 1
        self._goal_jump_candidate = candidate
        if self._goal_jump_count >= self.goal_jump_confirmations:
            self.goal_odo = candidate
            self._goal_jump_candidate = None; self._goal_jump_count = 0

    def _bearing_to(self, gx, gz):
        """robot-frame bearing (rad, 0=forward, +=right) to odo-frame point (gx, gz)."""
        dx, dz = gx - self.ox, gz - self.oz
        ca, sa = np.cos(self.oyaw), np.sin(self.oyaw)
        rx = dx * ca + dz * sa                # world delta -> robot frame (inverse rot)
        rz = -dx * sa + dz * ca
        return float(np.arctan2(rx, -rz))     # forward = -z

    def _goal_bearing(self):
        """robot-frame bearing (rad, 0=forward, +=right) to the anchored goal, or None."""
        return None if self.goal_odo is None else self._bearing_to(self.goal_odo[0], self.goal_odo[1])

    def _ray_navigable_goal(self, gx, gz):
        """Choose a navigable local-map point on the robot-to-object ray.

        This helper is specific to pre-convergence greedy navigation.  The
        post-convergence FMM controller instead plans to a multi-source stop
        region around the raw object target.
        """
        ax, az = self.ox, self.oz
        L = float(np.hypot(gx - ax, gz - az))
        if L < 1e-6:
            return (gx, gz)
        ux, uz = (gx - ax) / L, (gz - az) / L
        for t in np.linspace(L, 0.0, int(L / (self.res * 0.5)) + 1):
            px, pz = ax + ux * t, az + uz * t
            r, c = self._to_cell(np.array([px]), np.array([pz]))
            if not self._is_obstacle(int(r[0]), int(c[0])):
                return (float(px), float(pz))     # first non-obstacle cell = standpoint
        return (gx, gz)

    def _last_mile_action(self, scan):
        """LOCAL-map last-mile: approach the anchored goal, STOP when within stop_distance
        (success is judged on physical pose, so no global convergence is needed).

        Arrival-when-blocked: near the detected goal the object itself shows up as an
        obstacle in the scan/local map, so waiting for dist <= stop_distance can never be
        met -- the robot would orbit it forever. So once near, we FACE the goal and STOP as
        soon as the way ahead is blocked (the blocker IS the object we homed on; detection
        implies line of sight), with a stall failsafe if it can't even face it.

        Anti-oscillation (far from the goal): once an obstacle forces an avoidance turn we
        COMMIT that turn direction (never immediately reverse it) until the way ahead opens,
        then take one forced forward step before resuming greedy homing."""
        # Pre-convergence greedy navigation needs one explicit local-map
        # standpoint because it cannot plan to the post-convergence FMM source
        # region.
        gx, gz = self._ray_navigable_goal(self.goal_odo[0], self.goal_odo[1])
        self._lm_subgoal_odo = (gx, gz)
        dist = float(np.hypot(gx - self.ox, gz - self.oz))
        goal_dist = float(np.hypot(self.goal_odo[0] - self.ox,
                                   self.goal_odo[1] - self.oz))
        if dist <= self.stop_distance:
            self._avoid_turn = None; self._avoid_count = 0; self._lm_stall = 0
            return self.A["stop"]
        fwd_ok = (not self._blocked(scan)) and (not self._swept_obstacle(self.oyaw))
        bear = self._bearing_to(gx, gz)
        # Half a turn step is the nearest-action boundary; 5 deg of extra slack prevents
        # small target/pose jitter there from flipping the turn direction every frame while
        # still correcting the 25--30 deg errors that a full-step deadband accepted.
        facing_tolerance = self.turn / 2.0 + np.deg2rad(5.0)
        facing = abs(bear) <= facing_tolerance
        full_turn = int(round(2 * np.pi / self.turn))

        # NEAR the detected goal: face it and STOP -- don't orbit. Approach while clear; the
        # moment forward is blocked (or we've spun a full turn without ever facing it) we are
        # as close as we can physically get -> STOP.
        if goal_dist <= self.last_mile_veto_off:
            self._avoid_turn = None; self._avoid_count = 0
            if not fwd_ok and (facing or self._lm_stall >= full_turn):
                self._lm_stall = 0
                return self.A["stop"]
            self._lm_stall += 1
            if not facing:
                return self.A["turn_left"] if bear < 0 else self.A["turn_right"]
            return self.A["move_forward"]            # facing + clear -> close the last bit
        self._lm_stall = 0

        # FAR from the goal: greedy homing with the anti-oscillation avoidance commit.
        if self._avoid_turn is not None:
            if fwd_ok:
                self._avoid_turn = None; self._avoid_count = 0
                return self.A["move_forward"]
            if self._avoid_count >= full_turn:
                self._avoid_turn = self._open_turn(scan)   # a full turn without opening -> re-pick side
                self._avoid_count = 0
            self._avoid_count += 1
            return self._avoid_turn
        if not facing:                                # not facing the goal -> turn to it
            return self.A["turn_left"] if bear < 0 else self.A["turn_right"]
        if fwd_ok:
            return self.A["move_forward"]             # facing + clear -> approach
        self._avoid_turn = self._open_turn(scan)      # blocked -> COMMIT an avoidance side
        self._avoid_count = 1
        return self._avoid_turn

    def _ensure_fmm(self):
        """Lazily build one FMMController that plans over the LOCAL odo map. Reuses the
        exact post-convergence planner (skfmm geodesic field + medial-axis clearance bias
        + deterministic stuck), just fed the odometry map/pose instead of the global one."""
        if self._fmm is not None:
            return
        from navigation.fmm_controller import FMMController
        self._lmv = _LocalMapView(self)
        fmm = FMMController(None, goal_radius=self.stop_distance,
                            resolution_m=self.res, agent_radius=self.rad,
                            turn_angle=np.rad2deg(self.turn),
                            forward_step_size=self.fwd)
        fmm.map = self._lmv                       # swap the throwaway global map for ours
        fmm.A = dict(self.A)                       # use EarlyNav's action ids (avoid int mismatch)
        fmm.set_pose_source(_OdoPose(self))
        self._fmm = fmm

    def _fmm_last_mile_action(self, scan):
        """FMM-planned local last-mile: route AROUND mapped obstacles to the anchored goal
        (safer than greedy homing, which wedges on a wall between robot and goal). Falls
        back to the greedy last-mile if the planner is stuck/goal unreachable, and always
        vetoes a forward that the coarse cell missed but the swept body would hit."""
        dist = float(np.hypot(self.goal_odo[0] - self.ox, self.goal_odo[1] - self.oz))
        if dist <= self.stop_distance:
            return self.A["stop"]
        self._ensure_fmm()
        self._lmv.refresh()                       # rebuild trav/clearance from the current map
        self._fmm.fmm_dist = None                 # force a replan on the (grown) local map
        self._fmm._goal_key = None
        try:
            a = self._fmm.get_next_action([self.goal_odo[0], 0.0, self.goal_odo[1]])
        except Exception:
            return self._last_mile_action(scan)   # any planner failure -> greedy homing
        if a == self.A["stop"] and dist > self.stop_distance:
            return self._last_mile_action(scan)   # stuck / unreachable -> greedy homing
        if a == self.A["move_forward"] and (self._blocked(scan) or self._swept_obstacle(self.oyaw)):
            return self._open_turn(scan)          # safety veto (coarse-cell miss)
        return a

    def _subgoal_bearing(self):
        """EXPLORE subgoal (NOT the detected object): a PERSISTENT frontier heading -- the
        odo-frame direction revealing the most unknown area -- re-picked periodically.
        Returns its bearing relative to the current facing (0 = already headed at it)."""
        self._subgoal_age -= 1
        if self._subgoal_yaw is None or self._subgoal_age <= 0:
            best_h, best_u = self.oyaw, -1
            for k in range(12):                              # scan headings around 360 deg
                h = _wrap(-np.pi + k * (2 * np.pi / 12))
                u = self._unknown_ahead(h)
                if u > best_u:
                    best_u, best_h = u, h
            self._subgoal_yaw, self._subgoal_age = best_h, 15
        return _wrap(self._subgoal_yaw - self.oyaw)

    def _combo_action(self, scan):
        """Exploration policy: SAFETY (hard) -> w_info*local-infogain + w_sub*heading
        toward the persistent EXPLORE subgoal (frontier). (The detected object is handled
        SEPARATELY by the committed local last-mile, not here.)"""
        sb = self._subgoal_bearing()
        fwd_safe = (not self._blocked(scan)) and (not self._swept_obstacle(self.oyaw))
        cands = {self.A["move_forward"]: 0.0,
                 self.A["turn_left"]: -self.turn, self.A["turn_right"]: self.turn}
        infos = {a: self._unknown_ahead(self.oyaw + dh) for a, dh in cands.items()}
        maxinfo = max(infos.values()) or 1
        best, bs, viz = None, -1e9, []
        for a, dh in cands.items():
            if a == self.A["move_forward"] and not fwd_safe:
                viz.append((float(dh), None))                # unsafe forward: not considered
                continue                                     # never drive into an obstacle
            score = (self.w_info * (infos[a] / maxinfo)
                     + self.w_sub * (np.cos(_wrap(sb - dh)) + 1) / 2)   # toward subgoal
            viz.append((float(dh), float(score)))
            if score > bs:
                bs, best = score, a
        # viz-only: remember this step's combo evaluation for the renderer (no control effect)
        self._viz_combo = {"sb": float(sb), "cands": viz,
                           "best_dh": (None if best is None else float(cands[best]))}
        if best is None:                                     # no safe forward -> rotate to subgoal
            return self.A["turn_left"] if sb < 0 else self.A["turn_right"]
        return best

    # -------------------- main -------------------- #
    def step(self, obs, collided, last_action, goal_robot_xz=None, delta_pose=None):
        if last_action is not None or delta_pose is not None:
            self._integrate(last_action, collided, delta_pose)
        scan = self.extract_scan(obs)
        self._update_map(scan)
        if goal_robot_xz is not None:
            self._set_goal(goal_robot_xz)
            self._det_count += 1
            self._lost_detection_count = 0
            if self._det_count >= self.goal_count_threshold:
                self._committed = True
        else:
            self._det_count = 0
            self._goal_jump_candidate = None; self._goal_jump_count = 0
            if self._committed:
                self._lost_detection_count += 1
                if self._lost_detection_count >= self.lost_detection_steps:
                    self._committed = False
                    self.goal_odo = None
                    self._lost_detection_count = 0
                    self._avoid_turn = None; self._avoid_count = 0
                    self._lm_stall = 0; self._lm_subgoal_odo = None
                    print("    [early-last-mile] target lost; resuming early exploration")
        self._steps += 1

        # recovery: a forward that just bumped -> turn away instead of retrying into it
        # (a narrow forward cone can miss a wall the robot BODY still hits; this caps
        # consecutive collisions, which dominated loc_coll for the map variants).
        if collided and last_action == self.A["move_forward"]:
            return self._open_turn(scan)

        # once the goal is confidently seen: LOCAL last-mile (approach + STOP), bypassing
        # global convergence. Otherwise explore (seed spin, then safety + infogain + subgoal).
        if self._committed and self.goal_odo is not None:
            return (self._fmm_last_mile_action(scan) if self.fmm_lastmile
                    else self._last_mile_action(scan))
        full_turn = int(round(2 * np.pi / self.turn))
        if self._steps <= full_turn:
            return self.A["turn_left"]

        # After the initial 360-degree scan, force one SAFE translation before
        # returning to the normal combo policy. This gives the localizer a new
        # viewpoint for its post-spin hold frame instead of blindly rotating to
        # 390 degrees. If forward is blocked, keep the existing combo policy's
        # safe turn choice and retain the pending translation until the path opens.
        if self._seed_translation_pending:
            if self._seed_forward_issued:
                if last_action == self.A["move_forward"] and not collided:
                    self._seed_translation_pending = False
                self._seed_forward_issued = False
            fwd_safe = (not self._blocked(scan)) and (not self._swept_obstacle(self.oyaw))
            if self._seed_translation_pending and fwd_safe:
                self._seed_forward_issued = True
                return self.A["move_forward"]
        return self._combo_action(scan)
