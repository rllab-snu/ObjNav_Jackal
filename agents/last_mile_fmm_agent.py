"""Last-mile navigation agent (GT-semantic detector + depth -> FMM approach).

Inspired by OVG-Nav / complex's last-mile logic, simplified:

  1. EXPLORE: random policy (ShortestPathFollower to random navigable points),
     never STOP.
  2. DETECT: every step a "detector" (here the GT semantic sensor) checks whether
     the goal object is visible. The goal object's pixels are back-projected with
     depth into world coordinates; their per-axis MEDIAN is the target point.
  3. LAST-MILE: once detected (>= min_goal_pixels within last_mile_range for a few
     frames), switch to last-mile and drive to that target with the FMM
     controller (FMMController over a loaded occupancy map -- the low-level control
     is used as-is). While the object stays visible the target is refined (EMA). STOP when
     within stop_distance of the target. No stuck handling (per request).

Dataset-agnostic; needs a `semantic_id_mapping.json` next to the scene-dataset
config (daejeon provides one) to map the goal category -> semantic id(s), and a
saved occupancy map (``--map_dir``, from collect_occupancy_map.py) per scene.
"""
import os
import sys
import time
import glob
import gzip
import json
import pickle
import threading
import numpy as np

from navigation.fmm_controller import FMMController
from navigation.global_occupancy_map import GlobalOccupancyMap
from navigation.localizer import make_localizer
from planning.topology import Topology
from qwen_priors.common import combine_prior_scores
from navigation.geometry import (intrinsics_from_hfov, backproject_pixels,
                                 heading_from_rotvec, scan_from_depth)
from utils import habitat_cfg
from utils.sam3_prompt import prompt as SAM3_PROMPT   # category -> natural-language SAM 3 prompt


class NoReachableLastMileSource(RuntimeError):
    """The detected object has no safe, visible, reachable FMM terminal cell."""


class LastMileFMMAgent:
    def __init__(self, args, config):
        self.map_dir = args.map_dir
        self.success_distance = habitat_cfg.success_distance(config)
        sim = config.habitat.simulator
        self.agent_radius = float(sim.agents.main_agent.radius)
        self.turn_angle = float(sim.turn_angle)
        self.forward_step = habitat_cfg.forward_step_size(config)

        # Localization: "oracle" (sim state) or an observation-based localizer
        # ("scan_match" / "particle") that matches accumulated depth scans against
        # the loaded occupancy map.
        self.localization = getattr(args, "localization", "oracle")
        # The start pose is ALWAYS unknown: the particle filter seeds map-wide and the
        # agent spins in place until it converges (global MCL is the only mode).
        # dynamic_map: after the pose is trusted (localizer converged, or oracle),
        # fuse observed NEW obstacles into the PLANNING map so the controller avoids
        # them (the base map has none). Localization stays on the static base map.
        self.dynamic_map = bool(getattr(args, "dynamic_map", False))
        # dynamic-map tuning knobs (passed to GlobalOccupancyMap.enable_dynamic);
        # dynmap_hmin overrides the scan height-band lower bound for the map update
        # only (so floor objects below the base map's band can be captured).
        self.dynmap_kwargs = dict(getattr(args, "dynmap_kwargs", {}) or {})
        self.dynmap_hmin = getattr(args, "dynmap_hmin", None)
        # gate for trusting the pose to update the dynamic map: "settled" (stricter,
        # sustained-tight convergence) or "converged" (looser/earlier).
        self.dynmap_gate = str(getattr(args, "dynmap_gate", "settled"))
        self.max_localize_steps = 80   # spin + active motion needs a larger budget
        # early_nav / dguard / safe_veto flags -- SHARED with the robot (RobotAgent calls the
        # same helper with its cfg), so a new knob is added once here and both inherit it.
        self._init_nav_extensions(args)
        # optional localizer overrides for ablations (e.g. {"active_seek": False})
        self.loc_overrides = dict(getattr(args, "loc_overrides", {}) or {})

        # Optional FMM controller overrides (else use FMMController defaults).
        self.ctrl_overrides = {}
        for k in ("clearance_weight", "clearance_cap_m", "step_cells"):
            if getattr(args, k, None) is not None:
                self.ctrl_overrides[k] = getattr(args, k)

        # Detection cameras. Each camera's mounting (height + down-tilt) is known, so
        # its goal-object pixels back-project to the SAME world frame -> we can fuse
        # detections from the main camera AND the higher/tilted sub-camera (which
        # catches low objects the main misses up close). --detect_camera forces a
        # single one ("main"/"top"); "auto" (default) uses every camera present.
        sim_sensors = config.habitat.simulator.agents[habitat_cfg.MAIN_AGENT].sim_sensors
        has_top = "depth_sensor_top" in sim_sensors

        def _cam_spec(suffix):
            dep = habitat_cfg.sim_sensor(config, "depth_sensor" + suffix)
            W, H = int(dep.width), int(dep.height)
            fx, fy, cx, cy = intrinsics_from_hfov(W, H, float(dep.hfov))
            return {"name": "top" if suffix else "main",
                    "rgb": "rgb" + ("_top" if suffix else ""),
                    "depth": "depth" + ("_top" if suffix else ""),
                    "sem": "semantic" + ("_top" if suffix else ""),
                    "W": W, "H": H, "max_depth": float(dep.max_depth),
                    "height": float(dep.position[1]),
                    "position": np.asarray(dep.position, dtype=np.float64),
                    "tilt": float(getattr(dep, "orientation", [0.0, 0.0, 0.0])[0]),
                    "fx": fx, "fy": fy, "cx": cx, "cy": cy}

        main_cam = _cam_spec("")
        want = getattr(args, "detect_camera", "auto")
        if want == "top" and not has_top:
            print("    [last_mile_fmm] --detect_camera top but no sub-camera; using main")
            want = "main"
        if want == "main":
            self.det_cams = [main_cam]
        elif want == "top":
            self.det_cams = [_cam_spec("_top")]
        else:                                   # "auto"/"both": fuse all present
            self.det_cams = [main_cam] + ([_cam_spec("_top")] if has_top else [])
        self.detect_camera = want
        print(f"    [last_mile_fmm] detection camera(s): {[c['name'] for c in self.det_cams]}")

        # The MAIN camera drives the localizer / scan (its height band matches how the
        # occupancy map was built); detection fuses self.det_cams.
        self.W, self.H = main_cam["W"], main_cam["H"]
        self.max_depth = main_cam["max_depth"]
        self.sensor_height = main_cam["height"]
        self.sensor_tilt = -main_cam["tilt"]  # positive value means downward pitch
        self.sensor_position = np.asarray(main_cam["position"], dtype=np.float64)
        self.fx, self.fy, self.cx, self.cy = (main_cam["fx"], main_cam["fy"],
                                              main_cam["cx"], main_cam["cy"])
        self.rgb_key, self.depth_key, self.sem_key = main_cam["rgb"], main_cam["depth"], main_cam["sem"]

        # Detector for the goal object: "oracle" (GT semantic sensor) or "sam3"
        # (open-vocab SAM 3 server, prompted with the goal category; runs in the
        # separate sam3_complex env, reached via sam3_bridge.Sam3Client).
        self.detector = getattr(args, "detector", "oracle")
        self.sam3 = None
        if self.detector == "sam3":
            from sam3_bridge import Sam3Client
            self.sam3 = Sam3Client(host=getattr(args, "sam3_host", "127.0.0.1"),
                                   port=int(getattr(args, "sam3_port", 5555)))
            print(f"    [last_mile_fmm] detector = SAM 3 (server {self.sam3.host}:{self.sam3.port})")

        # Detection / last-mile params (simple defaults).
        self.min_goal_pixels = int(getattr(args, "min_goal_pixels", 50))
        self.last_mile_range = 5.0       # only trust object pixels within this depth
        self.goal_count_threshold = 1    # first valid detection starts last-mile
        self.lost_detection_steps = int(getattr(args, "lost_detection_steps", 500))
        self.goal_ema = 0.9              # favor the latest accepted target detection
        self.goal_jump_reject = 1.5      # single-frame jump enters confirmation above this (m)
        self.goal_jump_confirmations = 2 # consistent large detections required to rebase
        self.goal_jump_consistency = 0.5 # max separation between large-jump confirmations (m)
        # Keep the planner/early-nav arrival radius at 0.5 m.  The direct
        # PF-pose-to-object arrival check has a separate, slightly wider margin:
        # a 20--30 cm global-localization error must not make an already-close
        # robot keep driving into the object.
        self.stop_distance = 0.5
        self.object_stop_distance = float(getattr(args, "object_stop_distance", 0.7))
        # Before STOPping, turn to face the goal object so it is roughly centered in view.
        # Half a turn step is the nearest-action boundary; 5 deg of slack absorbs the small
        # per-frame target jitter that otherwise causes left/right oscillation at that boundary.
        self.face_tolerance_deg = self.turn_angle / 2.0 + 5.0
        self.max_face_turns = int(round(360.0 / self.turn_angle)) + 1
        # Search safe source cells up to 1 m from the object, then use only the
        # nearest cells plus a 20 cm band. This avoids 0.75/1.0 radius switching
        # while retaining several nearby alternatives for multi-source FMM.
        self.source_max_distance = 1.0
        self.source_band_width = 0.20

        # Search phase before last-mile: "random" = wander to random navigable
        # points; "gt_goal" = head toward the episode's GT goal position. Either
        # way the detector runs every step and last-mile takes over on detection.
        # (search_mode_cfg is the configured mode; search_mode is the *runtime* one,
        # reset per episode -- gt_goal may switch to random mid-episode.)
        self.search_mode_cfg = getattr(args, "lastmile_search", "random")

        # --lastmile_search topo: simulation episodes may first visit one target-annotated view
        # from their scenario topology, then use the highest-scoring VIEW for the goal from the
        # per-view object priors. The prior for the goal is LOADED from the stored
        # object-first scores (prior_scores_path -> scores[goal] = {node:{dir}}) into self._prior,
        # then UPDATED IN PLACE as the robot visits views (update_view_prior): on arrival at a
        # subgoal view, that one view is re-scored from the live image (dynamic scenes) -- fast, no
        # full recompute. topo_update='view' enables the live update (needs Qwen); 'off' = stored only.
        self.topo_path = getattr(
            args, "topo_path", "topo_data/topo_graph/base_topology.pkl"
        )
        self.scenario_topologies = dict(getattr(args, "scenario_topologies", {}) or {})
        self.prior_scores_path = getattr(
            args, "prior_scores", os.path.join("outputs", "priors", "ensemble", "scores.json"))
        self.topo_update = getattr(args, "topo_update", "off")   # 'view' = live per-view update
        self.qwen_vlm = getattr(args, "qwen_vlm", "Qwen/Qwen3-VL-8B-Instruct")
        self.qwen_vlm_load = getattr(args, "qwen_vlm_load", "4bit")
        self._upd = None                 # cached (dir_scorer, room_scorer) resident VLM for updates
        self._room_aff = {}              # cached room-affinity per node id (for live updates)
        self._prior = None               # working {node:{dir:score}} for the current goal
        self._prior_obj = None
        self._visited = None             # visited-graph {node:{dir:bool}} (built on first select)
        self._cur_node = None            # node id of the current subgoal (for update_view_prior)
        # Topologies are large image-carrying pickles. Cache each scenario by path across
        # episodes, while `_topo_cached` remains the active episode topology for existing
        # planners and visualization code.
        self._topo_cache = {}
        self._topo_cached = None
        self._active_topo_path = None
        self._topo_phase = None          # episode-local: prior_hint -> score_search
        self._prior_hint = None
        # On arrival at a topo subgoal, turn to FACE its highest-scoring view. Views are
        # captured dir_00 = world yaw 0, +30 deg per index (turn_right = +yaw), so the target
        # world yaw of view k is k * VIEW_STEP_DEG.
        self.VIEW_STEP_DEG = 30.0
        self._face_dir = None            # best view index to face at the current subgoal
        self._face_view_turns = 0

        # What to do in gt_goal mode when the GT viewpoint is reached but the goal
        # object was never detected:
        #   "explore"     -> switch to random exploration and keep searching until
        #                    detection (last-mile/STOP only ever follow a detection)
        #   "gt_fallback" -> last-mile toward the known GT object position
        self.gt_goal_on_miss = getattr(args, "gt_goal_on_miss", "explore")

        # Map goal category -> semantic id(s) from the dataset's mapping file.
        sd = str(sim.scene_dataset)
        self._mapping_path = os.path.join(os.path.dirname(sd), "semantic_id_mapping.json")

    # ------------------------------------------------------------------ #
    def _goal_ids(self, category):
        with open(self._mapping_path) as f:
            m = json.load(f)
        cat = str(category).lower()
        ids = {int(sid) for sid, c in m.get("semantic_id_to_category", {}).items()
               if str(c).lower() == cat}
        if not ids:  # fall back to matching the asset name
            ids = {int(sid) for a, sid in m.get("asset_to_semantic_id", {}).items()
                   if str(a).lower() == cat or str(a).lower().startswith(cat)}
        return ids

    # ------------------------------------------------------------------ #
    # ---- nav-extension setup: SHARED by the sim agent and the robot (RobotAgent) so both
    #      pick up every flag/feature with identical defaults; add a knob once, here. --------
    def _init_nav_extensions(self, src):
        """Read the early_nav / dguard / safe_veto knobs from `src` (run_eval args OR the robot
        cfg) via getattr with defaults. Called from __init__ (once)."""
        g = lambda k, d: getattr(src, k, d)
        self.use_early_nav = bool(g("early_nav", True))               # pre-conv odometry local-map nav
        self.fmm_lastmile = bool(g("fmm_lastmile", False))            # combo local last-mile via FMM
        self.combo_w_info = float(g("combo_w_info", 1.0))            # infogain weight
        self.combo_w_sub = float(g("combo_w_sub", 1.0))             # frontier-subgoal weight
        self.early_obst_min_hits = int(g("early_obst_min_hits", 2))   # local-map obstacle confirm count
        self.early_fwd_clear_margin = float(g("early_fwd_clear_margin", 0.05))  # fwd-clear cone margin (m)
        self.diverge_guard = bool(g("diverge_guard", True))          # bg: re-globalize on ram
        self.diverge_bad_hold = int(g("diverge_bad_hold", 8))
        self.safe_veto = bool(g("safe_veto", True))                  # collision shield
        self.safe_veto_hold = int(g("safe_veto_hold", 20))
        self.safe_veto_use_map = bool(g("safe_veto_use_map", True))   # shield uses odo local map
        self.last_mile_veto_off = float(g("last_mile_veto_off", 1.2))  # near-goal veto/dg exception (m)
        # A topology node is a waypoint whose yaw still has to be inspected, so
        # arrive more tightly than the 0.5 m last-mile object stopping boundary.
        self.topo_arrival_radius = float(g("topo_arrival_radius", 0.3))

    def _build_early_nav(self):
        """Build early_nav + the collision-shield local map and reset their per-episode state.
        Called from reset() (per episode), AFTER self.controller and self.localizer exist."""
        self._badstep = 0          # consecutive post-conv "policy drove into obstacle" steps
        self._avoid_mode = False   # safe_veto: in a wall-following avoidance maneuver
        self._avoid_dir = None     # committed turn action for the current maneuver
        self._shield_persist = 0   # steps the shield fought without free FMM progress
        self._shield_raw_action = None   # per-step FMM/early-nav action before shield override
        self._shield_action = None       # per-step action emitted by an active shield override
        self._odom_delta = None    # robot: measured (d_fwd,d_lat,dyaw) odom delta; None -> nominal
        self.early_nav = None
        if self.localizer is not None and self.use_early_nav:
            from navigation.early_nav import EarlyNav
            self.early_nav = EarlyNav(
                self.controller.A, lambda o: self.localizer.extract_scan(o, "depth"),
                self.forward_step, self.turn_angle,
                max_range=float(getattr(self.localizer, "max_range", 5.0)),
                agent_radius=self.agent_radius, stop_distance=self.stop_distance,
                fmm_lastmile=self.fmm_lastmile, w_info=self.combo_w_info, w_sub=self.combo_w_sub,
                obst_min_hits=self.early_obst_min_hits, fwd_clear_margin=self.early_fwd_clear_margin,
                last_mile_veto_off=self.last_mile_veto_off,
                goal_ema=self.goal_ema, goal_jump_reject=self.goal_jump_reject,
                goal_jump_confirmations=self.goal_jump_confirmations,
                goal_jump_consistency=self.goal_jump_consistency,
                goal_count_threshold=self.goal_count_threshold,
                lost_detection_steps=self.lost_detection_steps)
        self._shield_map = None
        if self.safe_veto and self.safe_veto_use_map and self.localizer is not None:
            from navigation.shield_local_map import ShieldLocalMap
            self._shield_map = ShieldLocalMap(
                self.controller.A, lambda o: self.localizer.extract_scan(o, "depth"),
                self.forward_step, self.turn_angle, agent_radius=self.agent_radius)

    def set_odom_delta(self, delta_pose):
        """Robot driver hook: supply the MEASURED odometry delta (d_fwd, d_lat, dyaw) in the
        ROBOT frame for the NEXT get_action, so the localizer / early_nav / shield integrate the
        real motion instead of the nominal action model. None (the reset default) -> nominal."""
        self._odom_delta = None if delta_pose is None else tuple(float(v) for v in delta_pose)

    def _handoff_early_last_mile(self):
        """Transfer a committed odometry-frame EarlyNav goal into the converged map frame."""
        en = getattr(self, "early_nav", None)
        if (self.mode != "explore" or en is None
                or not getattr(en, "_committed", False)
                or getattr(en, "goal_odo", None) is None):
            return False

        # EarlyNav stores the object in its odometry-origin frame. First recover the
        # object vector in the robot frame at convergence, then apply the converged
        # PF pose to express that same vector in the global planning frame.
        dx = float(en.goal_odo[0]) - float(en.ox)
        dz = float(en.goal_odo[1]) - float(en.oz)
        co, so = np.cos(float(en.oyaw)), np.sin(float(en.oyaw))
        robot_x = dx * co + dz * so
        robot_z = -dx * so + dz * co

        pos, yaw = self._pose()
        cw, sw = np.cos(float(yaw)), np.sin(float(yaw))
        target = np.array([
            float(pos[0]) + robot_x * cw - robot_z * sw,
            float(pos[1]),
            float(pos[2]) + robot_x * sw + robot_z * cw,
        ], dtype=np.float64)

        self.mode = "last_mile"
        self._last_mile_detection_backed = True
        self._topo_arrival_latch = None
        self.last_mile_goal = target
        self.subgoal = target
        self._lost_detection_count = int(getattr(en, "_lost_detection_count", 0))
        self._clear_goal_jump_candidate()
        self._last_mile_source_mask = None
        self._prev_xz = None
        self._stuck = 0
        self._face_turns = 0
        en._committed = False
        en.goal_odo = None
        en._lost_detection_count = 0
        print(f"    >>> LAST-MILE HANDOFF: early odometry goal -> {np.round(target, 2)}")
        return True

    # ------------------------------------------------------------------ #
    def reset(self, env, state_info=None):
        self.sim = env.sim
        ep = env.current_episode
        scene = ep.scene_id.split('/')[-1].split('.')[0]
        self.goal_category = getattr(ep, "object_category", "?")
        self.goal_ids = self._goal_ids(self.goal_category)
        # SAM 3 text prompt: map the category to a natural-language phrase (open-vocab
        # SAM 3 fails on concatenated names, e.g. 'tissuebox' -> 0 px but 'tissue box' works).
        self.sam3_prompt = SAM3_PROMPT.get(self.goal_category, self.goal_category)
        self.search_mode = self.search_mode_cfg   # runtime mode (may switch mid-episode)
        self._prior = None                        # reloaded for this goal on first subgoal select
        self._prior_obj = None
        self._visited = None                      # visited-graph rebuilt when the prior (re)loads
        self._cur_node = None
        self._face_dir = None
        self._topo_arrival_latch = None
        self._topo_phase = None
        self._prior_hint = None
        if self.search_mode == "topo":
            self._prepare_topology_search(scene)

        # Load the saved occupancy map (collect_occupancy_map.py output) for this
        # scene; the FMM controller plans over it for BOTH exploration (drive to
        # random navigable points) and last-mile.
        cands = sorted(glob.glob(os.path.join(self.map_dir, f"{scene}_*_occ_map.dat.gz")))
        if not cands:  # base scene is identical across variants -> any base map works
            cands = sorted(glob.glob(os.path.join(self.map_dir, "*_occ_map.dat.gz")))
        if not cands:
            raise FileNotFoundError(
                f"no occ map for scene '{scene}' in {self.map_dir} "
                f"(run collect_occupancy_map.py first)")
        with gzip.open(cands[0], "rb") as f:
            d = pickle.load(f)
        res = float(d["map_params"]["resolution_m"])
        gmap = GlobalOccupancyMap.from_occupancy_dict(d, agent_radius_m=self.agent_radius)

        self.controller = FMMController(
            self.sim, goal_radius=self.stop_distance,
            resolution_m=res, agent_radius=self.agent_radius,
            turn_angle=self.turn_angle, forward_step_size=self.forward_step,
            **self.ctrl_overrides)
        st0 = self.sim.get_agent_state()
        self.controller.reset(np.array(st0.position), occ_map=gmap)
        if self.dynamic_map:
            gmap.enable_dynamic(**self.dynmap_kwargs)   # fresh log-odds layer this episode

        # Observation-based localizer (replaces the oracle pose everywhere below).
        # The start pose is UNKNOWN: global MCL seeds particles map-wide and everything
        # comes from actions + depth (no oracle/start pose is used).
        self.localizer = None
        if self.localization != "oracle":
            self.localizer = make_localizer(
                self.localization,
                fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy, img_height=self.H,
                max_depth=self.max_depth, sensor_height=self.sensor_height,
                sensor_position=self.sensor_position,
                tilt_deg=np.degrees(self.sensor_tilt),
                forward_step=self.forward_step, turn_angle=self.turn_angle,
                actions=self.controller.A,
                **getattr(self, "loc_overrides", {}),   # ablation flags (e.g. active_seek=False)
            )
            self.localizer.reset(gmap)   # unknown start (global MCL); no start pose used
            self.controller.set_pose_source(self.localizer)
        # early_nav + collision-shield local map + their per-episode state -- SHARED with the
        # robot reset (RobotAgent calls the same builder), so both get identical behavior.
        self._build_early_nav()
        self._last_action = None
        self._ate = []      # localization error vs oracle, for reporting
        self._loc_steps = 0
        self._loc_reported = False
        self._grid_reported = False

        # Search target, navigated with FMM: in gt_goal mode the GT goal's best
        # (navigable) viewpoint -- the raw object position sits on furniture, so on
        # an occupancy map its nearest-free cell can be an isolated pocket the FMM
        # can't reach; the viewpoint is a proper navigable point. In random mode a
        # random navigable point (resampled on arrival).
        if self.search_mode == "gt_goal":
            g = ep.goals[0]
            ious = [(vp.iou if vp.iou is not None else -1.0) for vp in g.view_points]
            self.explore_goal = np.array(
                g.view_points[int(np.argmax(ious))].agent_state.position, dtype=np.float32)
        else:
            # 'random' -> a random navigable point; 'topo' -> high-level planning (dispatch).
            self.explore_goal = self._next_search_goal()

        self.goal_pos = np.array(ep.goals[0].position, dtype=np.float32)  # GT object pos (nav_viz)
        self.mode = "explore"
        self.last_mile_goal = None
        self._last_mile_detection_backed = False
        self._goal_jump_candidate = None
        self._goal_jump_count = 0
        self.subgoal = self.explore_goal      # for nav_viz
        self._det_count = 0
        self._lost_detection_count = 0
        self._prev_xz = None
        self._stuck = 0
        self._face_turns = 0
        self._free_lab = None       # connected-component cache (per episode map)
        self._last_mile_source_mask = None  # safe multi-source FMM terminal region
        self._episode_failure_reason = None
        print(f"    [last_mile_fmm search={self.search_mode} on occ map: {os.path.basename(cands[0])}]"
              f" goal '{self.goal_category}' -> semantic ids {sorted(self.goal_ids)}"
              + (f" | sam3 prompt '{self.sam3_prompt}'" if self.detector == "sam3" else ""))

    # ------------------------------------------------------------------ #
    # --- sim I/O boundary (overridden by the habitat-free robot subclass) -------
    def _sample_navigable(self):
        """A random navigable world point (explore target). Habitat: the sim
        pathfinder; robot: a random explorable occupancy-map cell."""
        return np.array(self.sim.sample_navigable_point(), dtype=np.float32)

    # --- exploration subgoal selection (search phase, before last-mile) ----------
    def _next_search_goal(self):
        """The next EXPLORE subgoal (world x,y,z) to navigate toward before a detection
        triggers last-mile. Dispatches on the search mode:
          * 'topo'  -> one scenario annotation hint, then _high_level_subgoal() score planning
          * else    -> a random navigable point (the current default)
        ('gt_goal' targets the GT viewpoint and is handled directly in reset/get_action.)
        Returns a world point, or None if the planner is exhausted (caller falls back)."""
        if self.search_mode == "topo":
            if getattr(self, "_topo_phase", None) == "prior_hint":
                hint = self._prior_hint
                self._cur_node = hint["node_id"]
                self._face_dir = hint["dir"]
                self._face_view_turns = 0
                pos = self._pose()[0]
                print(f"    [topo-hint] target '{self.goal_category}' -> "
                      f"n{hint['node_id']}d{hint['dir']} "
                      f"({hint['pixels']} px, confidence {hint['confidence']:.3f})", flush=True)
                return np.array([hint["x"], float(pos[1]), hint["z"]], dtype=np.float32)
            return self._high_level_subgoal()
        return self._sample_navigable()

    TOPK_SUBGOAL = 3        # among the top-K scoring unvisited views, go to the NEAREST

    def _high_level_subgoal(self):
        """--lastmile_search topo: SELECT the next subgoal view each call from the CURRENT prior --
        take the top-K (=TOPK_SUBGOAL) highest-scoring UNVISITED views, then pick the one NEAREST to
        the current pose; go there and face it. Equal scores are ordered by node distance and then
        required yaw rotation, so a tie at one node is inspected in a continuous yaw sweep rather
        than topology/JSON insertion order. On arrival the view is re-scored live
        (update_view_prior) and marked visited, so the next call re-selects from the updated prior
        over the remaining views -- repeating until a detection starts last-mile. Returns the world
        (x,y,z) subgoal, or None when no unvisited view has a positive score (-> random fallback).

        The prior is LOADED once from the stored object-first scores (self._prior = scores[goal]);
        `self._visited` is a visited-graph mirroring the topology (per node, per direction)."""
        if getattr(self, "_prior", None) is None or getattr(self, "_prior_obj", None) != self.goal_category:
            self._prior = self._load_prior(self.goal_category)
            self._prior_obj = self.goal_category
            if self._visited is None:
                self._visited = self._init_visited_graph()
        if not self._prior:
            print(f"    [topo] no prior for goal '{self.goal_category}'; falling back to random")
            return None
        topo = self._topo_for_positions()
        if topo is None:
            return None
        pos, heading = self._pose()
        px, pz = float(pos[0]), float(pos[2])
        cands = []                                  # (score, node_id, dir, x, z) over UNVISITED views
        for node in topo.nodes:
            nid = str(node.id)
            for dd, s in self._prior.get(nid, {}).items():
                if self._visited.get(nid, {}).get(str(dd)):
                    continue
                cands.append((float(s), node.id, int(dd), float(node.x), float(node.z)))
        cands = [c for c in cands if c[0] > 0.0]
        if not cands:
            print(f"    [topo] all views visited / zero prior for '{self.goal_category}'; random")
            return None
        def distance_sq(c):
            return (c[3] - px) ** 2 + (c[4] - pz) ** 2

        def yaw_delta(c):
            desired = np.deg2rad(c[2] * getattr(self, "VIEW_STEP_DEG", 30.0))
            delta = abs((desired - heading + np.pi) % (2.0 * np.pi) - np.pi)
            # Numerically symmetric turns (e.g. dir1/dir11 from yaw 0) must reach the
            # deterministic node/dir tie-break instead of differing at ~1e-16 radians.
            return round(float(delta), 12)

        # Score remains the primary top-K criterion. Distance/yaw only resolve exact score ties,
        # and IDs make the result independent of topology and JSON insertion order.
        ranked = sorted(cands, key=lambda c: (
            -c[0], distance_sq(c), yaw_delta(c), int(c[1]), c[2]))
        topk = ranked[:self.TOPK_SUBGOAL]
        # Preserve the existing "nearest among top-K" policy. For the same node/distance, retain
        # the higher score first and use the smallest turn only when those scores are tied.
        best = min(topk, key=lambda c: (
            distance_sq(c), -c[0], yaw_delta(c), int(c[1]), c[2]))
        _, nid_i, dir_i, x, z = best
        # NOTE: not marked visited here -- only on ARRIVAL (see _explore_action), so the in-progress
        # subgoal keeps its score colour while travelling; the next selection excludes it once reached.
        self._cur_node = nid_i
        self._face_dir = dir_i
        self._face_view_turns = 0
        print("    [topo] top%d=%s -> nearest n%sd%s (score %.2f)" % (
            self.TOPK_SUBGOAL, [(f"n{c[1]}d{c[2]}", round(c[0], 2)) for c in topk],
            nid_i, dir_i, best[0]), flush=True)
        return np.array([x, float(pos[1]), z], dtype=np.float32)

    def _init_visited_graph(self):
        """Visited-graph mirroring the topology: {node_id: {dir: False}} for every view. Target
        views and intermediate yaw views observed while turning at a node are set True and excluded
        from future subgoal selection."""
        topo = self._topo_for_positions()
        if topo is None:
            return {}
        return {str(n.id): {str(v.index): False for v in n.views} for n in topo.nodes}

    def _load_prior(self, goal):
        """Stored per-view prior for ONE object: object-first scores.json[goal] -> {node:{dir:score}}
        (floats). {} if the file/object is unavailable."""
        path = getattr(self, "prior_scores_path", os.path.join("outputs", "priors", "ensemble", "scores.json"))
        try:
            with open(path) as f:
                allobj = json.load(f)
        except Exception as e:
            print(f"    [topo-plan] cannot load prior scores ({e})")
            return {}
        per = allobj.get(goal, {})
        return {str(n): {str(d): float(s) for d, s in dirs.items()} for n, dirs in per.items()}

    def _resolve_topology_path(self, scene):
        """Return the scenario topology mapped to ``scene``, or the configured base path."""
        for prefix, path in getattr(self, "scenario_topologies", {}).items():
            if str(scene).startswith(str(prefix)):
                return path
        return self.topo_path

    def _load_topology_path(self, path):
        """Load a topology once per path and reuse it across shuffled episodes."""
        cache = getattr(self, "_topo_cache", None)
        if cache is None:
            cache = self._topo_cache = {}
        if path not in cache:
            cache[path] = Topology.load(path)
        return cache[path]

    def _activate_topology(self, scene):
        """Activate the scene topology, falling back to ``topo_path`` on load failure."""
        requested = self._resolve_topology_path(scene)
        candidates = [requested]
        if requested != self.topo_path:
            candidates.append(self.topo_path)
        self._topo_cached = None
        self._active_topo_path = None
        for path in candidates:
            try:
                topo = self._load_topology_path(path)
            except Exception as e:
                print(f"    [topo-plan] cannot load topo '{path}' ({e})")
                continue
            self._topo_cached = topo
            self._active_topo_path = path
            if path != requested:
                print(f"    [topo-plan] scenario topo unavailable; using base '{path}'", flush=True)
            else:
                print(f"    [topo-plan] active topology '{path}'", flush=True)
            return topo
        return None

    def _prepare_topology_search(self, scene):
        """Activate this episode's topology and initialize its one-shot hint phase."""
        topo = self._activate_topology(scene)
        self._visited = self._init_visited_graph() if topo is not None else {}
        self._prior_hint = self._select_prior_hint(self.goal_category) if topo is not None else None
        if self._prior_hint is None:
            self._topo_phase = "score_search"
            print(f"    [topo-hint] no stored annotation for '{self.goal_category}'; "
                  "starting score search", flush=True)
        else:
            self._topo_phase = "prior_hint"

    def _complete_topology_view(self, obs=None):
        """Mark the faced topo view visited and advance a one-shot hint to score search."""
        node_id = getattr(self, "_cur_node", None)
        dir_idx = getattr(self, "_face_dir", None)
        self._topo_arrival_latch = None
        if node_id is None or dir_idx is None:
            return
        if self._visited is None:
            self._visited = self._init_visited_graph()
        self._visited.setdefault(str(node_id), {})[str(dir_idx)] = True
        if obs is not None:
            self.update_view_prior(node_id, dir_idx, obs)
        if getattr(self, "_topo_phase", None) == "prior_hint":
            self._topo_phase = "score_search"
            print(f"    [topo-hint] n{node_id}d{dir_idx} faced without detection; "
                  "continuing with score search", flush=True)
        self._face_dir = None

    def _mark_observed_topology_yaw(self, heading_rad):
        """Mark the nearest view direction seen while turning at the current node."""
        node_id = getattr(self, "_cur_node", None)
        if node_id is None:
            return None
        if self._visited is None:
            self._visited = self._init_visited_graph()
        step = float(getattr(self, "VIEW_STEP_DEG", 30.0))
        direction_count = max(1, int(round(360.0 / step)))
        yaw_deg = float(np.degrees(heading_rad)) % 360.0
        dir_idx = int(np.floor((yaw_deg + step / 2.0) / step)) % direction_count
        self._visited.setdefault(str(node_id), {})[str(dir_idx)] = True
        return dir_idx

    def _select_prior_hint(self, goal):
        """Best stored target annotation in the active topology, or ``None``.

        Mask pixels from the main and sub cameras are added. The largest combined
        object view wins, with confidence, node ID, and direction as deterministic
        tie-breakers.
        """
        topo = self._topo_for_positions()
        if topo is None:
            return None
        candidates = []
        for node in topo.nodes:
            for view in node.views:
                per_camera = view.meta.get("objects", {})
                infos = [cats.get(goal) for cats in per_camera.values()
                         if isinstance(cats, dict) and cats.get(goal) is not None]
                if not infos:
                    continue
                pixels = sum(int(info.get("pixels", 0) or 0) for info in infos)
                if pixels <= 0:
                    continue
                confidence = max(float(info.get("score", 0.0) or 0.0) for info in infos)
                candidates.append((
                    -pixels, -confidence, int(node.id), int(view.index),
                    {
                        "node_id": node.id,
                        "dir": int(view.index),
                        "pixels": pixels,
                        "confidence": confidence,
                        "x": float(node.x),
                        "z": float(node.z),
                    },
                ))
        return min(candidates, key=lambda item: item[:4])[4] if candidates else None

    def _topo_for_positions(self):
        if getattr(self, "_topo_cached", None) is None:
            try:
                path = getattr(self, "_active_topo_path", None) or self.topo_path
                self._topo_cached = self._load_topology_path(path)
                self._active_topo_path = path
            except Exception as e:
                print(f"    [topo-plan] cannot load topo ({e})")
                return None
        return self._topo_cached

    def _node_meta(self, node_id):
        """(room_type, furniture_all) for a node id, from the topology (stored context)."""
        topo = self._topo_for_positions()
        n = topo.node(int(node_id)) if topo is not None else None
        if n is None:
            return "", []
        return n.room_type, n.meta.get("furniture", {}).get("all", [])

    def _obs_images(self, obs):
        """LIVE main(+sub) RGB frames from the current observation as PIL images."""
        from PIL import Image
        out = []
        for cam in self.det_cams:
            k = cam.get("rgb")
            if k and k in obs:
                arr = np.asarray(obs[k])
                if arr.ndim == 3:
                    out.append(Image.fromarray(np.ascontiguousarray(arr[..., :3]).astype(np.uint8)))
        return out

    def _qwen_updater(self):
        """Lazily load the resident Qwen VLM used for live per-view updates (kept in memory across
        updates; VLM only, so it coexists with SAM 3 in ~10 GB). None if Qwen is unavailable."""
        if getattr(self, "_upd", None) is not None:
            return self._upd
        if getattr(self, "_upd_failed", False):
            return None
        try:
            here = os.path.dirname(os.path.abspath(__file__))
            qp = os.path.join(os.path.dirname(here), "qwen_priors")
            if qp not in sys.path:
                sys.path.insert(0, qp)
            from qwen_score import load_model
            from scorers import SharpenedScorer, RoomAffinityScorer
            model, proc = load_model(self.qwen_vlm, "vlm", self.qwen_vlm_load)
            self._upd = (SharpenedScorer(model, proc, "vlm"), RoomAffinityScorer(model, proc, "vlm"))
            print(f"    [topo-update] loaded {self.qwen_vlm} (resident VLM for live per-view updates)",
                  flush=True)
        except Exception as e:
            print(f"    [topo-update] Qwen unavailable ({type(e).__name__}: {str(e)[:80]}); "
                  "no live updates (stored prior kept)", flush=True)
            self._upd_failed = True
            self._upd = None
        return self._upd

    def update_view_prior(self, node_id, dir_idx, obs):
        """Re-score ONE view from the LIVE image (obs) for the goal and update the working prior in
        place -- the cheap per-view alternative to a full recompute, called on arrival at a subgoal
        view so a dynamic scene is reflected. Live updates have no caption-LLM branch, so they use
        the shared fusion rule with caption=0: score = VLM(live image) x soft-room-gate. The remaining
        plan is re-ranked from the updated prior. No-op unless topo_update=='view' and Qwen is
        available. Returns the new score, or None."""
        if getattr(self, "topo_update", "off") != "view":
            return None
        upd = self._qwen_updater()
        if upd is None:
            return None
        imgs = self._obs_images(obs)
        if not imgs:
            return None
        dir_scorer, room_scorer = upd
        obj = SAM3_PROMPT.get(self.goal_category, self.goal_category)
        room, fur = self._node_meta(node_id)
        if node_id not in self._room_aff:                 # room-affinity gate: once per node
            self._room_aff[node_id] = float(room_scorer.score(obj, room, fur))
        ds = float(dir_scorer.score(obj, room, fur, images=imgs))
        new = combine_prior_scores(ds, 0.0, self._room_aff[node_id])
        if self._prior is None:
            self._prior = {}
        self._prior.setdefault(str(node_id), {})[str(dir_idx)] = new   # next _high_level_subgoal re-selects
        print(f"    [topo-update] n{node_id} dir{dir_idx} live re-score -> {new:.2f} "
              f"(live VLM {ds:.2f}, room affinity {self._room_aff[node_id]:.2f}, soft gate)",
              flush=True)
        return new

    def _collided(self):
        """Did the last forward actually fail (bump)? Habitat: the sim flag."""
        return bool(self.sim.previous_step_collided)

    def _gt_position(self):
        """Oracle position for ATE logging only; None when unavailable (robot)."""
        return np.array(self.sim.get_agent_state().position, np.float64)

    def _pose(self):
        """(position xyz, heading rad) -- from the localizer if active, else oracle."""
        if self.localizer is not None:
            return self.localizer.pose()
        import quaternion as q
        st = self.sim.get_agent_state()
        return (np.array(st.position, np.float64),
                heading_from_rotvec(q.as_rotation_vector(st.rotation)))

    def _camera_pose(self, cam):
        """(cam_pos xyz, cam_rotvec) of camera `cam` (a det_cams spec), for
        back-projection. Exact from the sim (oracle) or composed from the estimated
        agent pose + this camera's known mounting translation + down-tilt)."""
        if self.localizer is None:
            import quaternion as q                     # oracle/sim only (the robot has a localizer)
            st = self.sim.get_agent_state()
            sensor = st.sensor_states.get(cam["depth"])
            if sensor is not None:
                return (np.array(sensor.position, np.float64),
                        q.as_rotation_vector(sensor.rotation))
            mount = np.asarray(cam.get(
                "position", [0.0, cam["height"], 0.0]), dtype=np.float64)
            cam_pos = (np.array(st.position, np.float64)
                       + q.as_rotation_matrix(st.rotation).dot(mount))
            return cam_pos, q.as_rotation_vector(st.rotation)
        # estimated: agent pose + this camera's mounting translation + pitch
        from scipy.spatial.transform import Rotation as R
        pos, yaw = self.localizer.pose()
        yaw_rot = R.from_euler("y", -yaw)
        mount = np.asarray(cam.get(
            "position", [0.0, cam["height"], 0.0]), dtype=np.float64)
        cam_pos = np.array(pos, np.float64) + yaw_rot.apply(mount)
        rot = yaw_rot * R.from_euler("x", cam["tilt"])
        return cam_pos, rot.as_rotvec()

    # ------------------------------------------------------------------ #
    def _goal_mask(self, cam, obs):
        """Boolean HxW mask of the goal object in camera `cam`, BEFORE depth gating.
        Source depends on --detector: 'oracle' = GT semantic sensor; 'sam3' = SAM 3
        server on the RGB frame (open-vocab, prompted with the goal category). On a
        SAM 3 failure it falls back to the semantic sensor if available. Returns None
        if the needed observation is missing."""
        if self.detector == "sam3":
            if cam["rgb"] in obs:
                r = self.sam3.segment(np.asarray(obs[cam["rgb"]]),
                                      getattr(self, "sam3_prompt", None) or self.goal_category)
                if r is not None:
                    return r["mask"]
            # SAM 3 unavailable -> fall through to the oracle semantic sensor
        if cam["sem"] not in obs:
            return None
        return np.isin(np.asarray(obs[cam["sem"]]).squeeze(), list(self.goal_ids))

    # ------------------------------------------------------------------ #
    def _split_instances(self, mask):
        """Split a boolean mask into connected-component instances (8-connectivity).
        Used for the ORACLE detector, whose semantic sensor carries no instance id --
        spatially-disjoint blobs of the goal category are taken as separate instances."""
        if mask is None or not mask.any():
            return []
        from scipy.ndimage import label
        lab, n = label(mask, structure=np.ones((3, 3), int))
        return [lab == k for k in range(1, n + 1)]

    # ------------------------------------------------------------------ #
    # Parallel detection: SAM 3 (GPU / server socket) only needs the RGB frame, and the
    # particle-filter predict+update (CPU FFT) touch no SAM 3 state, so we run them
    # concurrently -- get_action fires _start_segmentation() up front, the localizer
    # update runs, then the detection helpers read the pre-computed masks via _goal_masks
    # (which joins the worker first). Output is IDENTICAL to the sequential path (same RGB,
    # same prompt, and the pose used for back-projection is still this step's updated pose);
    # only the wall-clock overlaps. No-op when the detector is not SAM 3.
    _parallel_detection = True            # class default; both sim + robot inherit it

    def _join_segmentation(self):
        """Block until the background SAM 3 worker (if any) has filled self._seg_cache."""
        t = getattr(self, "_seg_thread", None)
        if t is not None:
            t.join()
            self._seg_thread = None

    def _start_segmentation(self, obs):
        """Kick off SAM 3 segmentation for every detection camera in a daemon thread so the
        GPU/socket detection overlaps the CPU particle-filter update. Results land in
        self._seg_cache keyed by camera name. No-op unless SAM 3 is the active detector."""
        self._join_segmentation()         # never leak a straggler from a step that skipped detection
        self._seg_cache = {}
        self._seg_obs_id = id(obs)
        self._seg_thread = None
        if (not getattr(self, "_parallel_detection", True)
                or self.detector != "sam3" or self.sam3 is None):
            return
        prompt = getattr(self, "sam3_prompt", None) or self.goal_category
        cams = [c for c in self.det_cams if c["rgb"] in obs]
        if not cams:
            return
        cache, sam3 = self._seg_cache, self.sam3

        def _work():
            for cam in cams:
                try:
                    cache[cam["name"]] = sam3.segment(np.asarray(obs[cam["rgb"]]), prompt)
                except Exception:         # match inline behaviour: a failure -> oracle fallback
                    cache[cam["name"]] = None
        self._seg_thread = threading.Thread(target=_work, name="sam3-seg", daemon=True)
        self._seg_thread.start()

    def _sam3_result(self, cam, obs):
        """SAM 3 segmentation dict for `cam` on `obs`, or None. Reuses the result the
        background worker computed for this obs (_start_segmentation) -- so the detection
        overlapped the localizer update -- otherwise segments inline. Same result either
        way; also lets detection_viz reuse the frame's masks instead of re-running SAM 3."""
        if self.sam3 is None:
            return None
        if getattr(self, "_seg_obs_id", None) == id(obs):
            self._join_segmentation()
            if cam["name"] in self._seg_cache:
                return self._seg_cache[cam["name"]]
        if cam["rgb"] not in obs:
            return None
        result = self.sam3.segment(np.asarray(obs[cam["rgb"]]),
                                   getattr(self, "sam3_prompt", None) or self.goal_category)
        # Sequential detection still belongs to this observation. Cache it just like the
        # parallel worker so decision, renderer, and recorder all reuse the exact same masks.
        if getattr(self, "_seg_obs_id", None) == id(obs):
            self._seg_cache[cam["name"]] = result
        return result

    def _goal_masks(self, cam, obs):
        """List of PER-INSTANCE boolean masks (each H,W) of the goal in camera `cam`.
        SAM 3: the detector's OWN per-instance masks (r['masks']) -- SAM 3 already
        separates the instances, so several objects of the category (e.g. many chairs)
        come back individually instead of as one union blob. Oracle: connected components
        of the semantic-sensor goal mask (no instance id in the sensor). Returns a list
        (possibly empty); None if the needed observation is missing. Shared by the sim and
        the robot (both inherit this), and needs no distance/eps parameter -- the instance
        split comes straight from the detector. The SAM 3 call may be served from the
        background pre-segmentation (see _start_segmentation); the result is identical."""
        if self.detector == "sam3":
            r = self._sam3_result(cam, obs)
            if r is not None and r.get("masks") is not None:
                return [np.asarray(m, bool) for m in r["masks"]]
            # SAM 3 unavailable -> fall through to the oracle semantic sensor
        if cam["sem"] not in obs:
            return None
        union = np.isin(np.asarray(obs[cam["sem"]]).squeeze(), list(self.goal_ids))
        return self._split_instances(union)

    def _goal_robot_xz(self, obs):
        """(x, z) of the detected goal in the ROBOT frame (no world pose needed) -- the
        same last-mile detection, back-projected with a robot-frame camera pose so it can
        be anchored in the pre-convergence local map. Each detected INSTANCE (SAM 3 mask /
        oracle component) is back-projected separately and the one NEAREST the robot wins;
        None if too few goal pixels total."""
        from scipy.spatial.transform import Rotation as R
        cands, n = [], 0                       # cands: per-instance median (x, y, z)
        for cam in self.det_cams:
            if cam["depth"] not in obs:
                continue
            masks = self._goal_masks(cam, obs)
            if not masks:
                continue
            depth_m = np.asarray(obs[cam["depth"]]).squeeze() * cam["max_depth"]
            rotvec = R.from_euler("x", cam["tilt"]).as_rotvec()      # robot frame (yaw=0)
            for im in masks:
                if im.shape != depth_m.shape:
                    import cv2
                    im = cv2.resize(im.astype(np.uint8), (depth_m.shape[1], depth_m.shape[0]),
                                    interpolation=cv2.INTER_NEAREST).astype(bool)
                valid = im & (depth_m > 1e-4) & (depth_m < self.last_mile_range)
                k = int(valid.sum())
                if k == 0:
                    continue
                ys, xs = np.nonzero(valid)
                pts = backproject_pixels(xs, ys, depth_m[ys, xs], cam["fx"], cam["fy"],
                                         cam["cx"], cam["cy"],
                                         np.asarray(cam.get(
                                             "position", [0.0, cam["height"], 0.0])),
                                         rotvec)
                cands.append(np.median(pts, axis=0)); n += k
        if n < self.min_goal_pixels or not cands:
            return None
        # robot frame: the agent is at the origin -> nearest instance = smallest hypot(x, z)
        m = min(cands, key=lambda p: float(np.hypot(p[0], p[2])))
        return (float(m[0]), float(m[2]))

    def _detect_target(self, obs):
        """World (x, y, z) of the goal to home on, or None. Every detection camera's every
        INSTANCE (SAM 3 per-instance mask / oracle connected component) is back-projected
        into the world with that camera's own pose, and the instance whose median is
        NEAREST the agent wins -- so several objects of the category (e.g. many chairs) no
        longer blend into a between-objects point. Cameras fuse implicitly: every camera's
        every instance is a candidate, so the sub-camera can still supply a low object the
        main misses. None if too few goal pixels total (across all cameras/instances)."""
        cands, n_px = [], 0                    # cands: per-instance world median (x, y, z)
        for cam in self.det_cams:
            if cam["depth"] not in obs:
                continue
            masks = self._goal_masks(cam, obs)
            if not masks:
                continue
            depth_m = np.asarray(obs[cam["depth"]]).squeeze() * cam["max_depth"]
            cam_pos, rotvec = self._camera_pose(cam)      # this camera's world pose + intrinsics
            for im in masks:
                if im.shape != depth_m.shape:             # rgb/depth res mismatch
                    import cv2
                    im = cv2.resize(im.astype(np.uint8), (depth_m.shape[1], depth_m.shape[0]),
                                    interpolation=cv2.INTER_NEAREST).astype(bool)
                valid = im & (depth_m > 1e-4) & (depth_m < self.last_mile_range)
                k = int(valid.sum())
                if k == 0:
                    continue
                ys, xs = np.nonzero(valid)
                pts = backproject_pixels(xs, ys, depth_m[ys, xs],
                                         cam["fx"], cam["fy"], cam["cx"], cam["cy"],
                                         cam_pos, rotvec)
                cands.append(np.median(pts, axis=0)); n_px += k
        if n_px < self.min_goal_pixels or not cands:
            return None
        pos, _ = self._pose()                          # agent world pose -> nearest instance wins
        ox, oz = float(pos[0]), float(pos[2])
        return min(cands, key=lambda m: float(np.hypot(m[0] - ox, m[2] - oz)))

    def detection_viz(self, obs):
        """VIZ-ONLY (read-only): per-camera per-instance goal masks + which instance is the
        SELECTED goal, using the SAME nearest rule + min_goal_pixels gate as _detect_target.
        Returns {cam_name: {"masks": [HxW bool, ...], "selected": local_idx or None}} -- the
        renderer paints the selected instance green and the other candidates red. `selected`
        is set on the ONE camera holding the globally-nearest instance (others -> None); if
        the total goal pixels are below min_goal_pixels (no last-mile yet) nothing is
        selected. Runs its own detection pass, independent of the decision path."""
        pos, _ = self._pose()
        ox, oz = float(pos[0]), float(pos[2])
        out, best, n_px = {}, None, 0        # best = (dist, cam_name, local_idx)
        for cam in self.det_cams:
            masks = self._goal_masks(cam, obs) if cam["depth"] in obs else None
            if not masks:
                out[cam["name"]] = {"masks": [], "selected": None}
                continue
            depth_m = np.asarray(obs[cam["depth"]]).squeeze() * cam["max_depth"]
            cam_pos, rotvec = self._camera_pose(cam)
            disp = []
            for li, im in enumerate(masks):
                disp.append(im)                          # original (rgb/sam3 res) for display
                m = im
                if m.shape != depth_m.shape:
                    import cv2
                    m = cv2.resize(m.astype(np.uint8), (depth_m.shape[1], depth_m.shape[0]),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
                valid = m & (depth_m > 1e-4) & (depth_m < self.last_mile_range)
                k = int(valid.sum())
                if k == 0:
                    continue
                n_px += k
                ys, xs = np.nonzero(valid)
                med = np.median(backproject_pixels(xs, ys, depth_m[ys, xs], cam["fx"], cam["fy"],
                                                   cam["cx"], cam["cy"], cam_pos, rotvec), axis=0)
                d = float(np.hypot(med[0] - ox, med[2] - oz))
                if best is None or d < best[0]:
                    best = (d, cam["name"], li)
            out[cam["name"]] = {"masks": disp, "selected": None}
        if best is not None and n_px >= self.min_goal_pixels:
            out[best[1]]["selected"] = best[2]
        return out

    def _per_camera_counts(self, obs):
        """Goal-object pixel count for EACH detection camera separately (diagnostic:
        shows whether the sub-camera sees the object when the main does not). Uses the
        active detector's mask, gated by valid depth within last_mile_range."""
        out = {}
        for cam in self.det_cams:
            if cam["depth"] not in obs:
                out[cam["name"]] = -1; continue
            mask = self._goal_mask(cam, obs)
            if mask is None:
                out[cam["name"]] = -1; continue
            depth_m = np.asarray(obs[cam["depth"]]).squeeze() * cam["max_depth"]
            if mask.shape != depth_m.shape:
                import cv2
                mask = cv2.resize(mask.astype(np.uint8), (depth_m.shape[1], depth_m.shape[0]),
                                  interpolation=cv2.INTER_NEAREST).astype(bool)
            out[cam["name"]] = int((mask & (depth_m > 1e-4)
                                    & (depth_m < self.last_mile_range)).sum())
        return out

    # ------------------------------------------------------------------ #
    def _explore_action(self, obs=None):
        """Search phase, navigated with FMM. Returns the controller action toward
        the search target. In random mode, resample a new navigable point whenever
        the controller arrives / can't progress (never STOP). In gt_goal mode the
        STOP (arrival at the GT viewpoint) is returned as-is and handled by
        get_action (-> fall back to last-mile toward the GT object)."""
        def controller_action():
            if self.search_mode == "topo":
                return self.controller.get_next_action(
                    self.explore_goal,
                    goal_radius=self.topo_arrival_radius,
                )
            return self.controller.get_next_action(self.explore_goal)

        face_dir = getattr(self, "_face_dir", None)
        arrival_key = (getattr(self, "_cur_node", None), face_dir)
        arrival_latched = (
            self.search_mode == "topo"
            and face_dir is not None
            and getattr(self, "_topo_arrival_latch", None) == arrival_key
        )
        if getattr(self, "_topo_arrival_latch", None) is not None and not arrival_latched:
            self._topo_arrival_latch = None
        action = self.controller.A["stop"] if arrival_latched else controller_action()
        # On controller STOP (arrived at / can't progress to the subgoal) resolve the topo
        # arrival + next-subgoal handoff in one loop. A loop (not a one-shot) is needed because
        # the NEXT subgoal can be another VIEW OF THE SAME NODE: the controller then STOPs again
        # immediately (already at that position), and we must turn to FACE the new view rather
        # than emit a spurious move -- previously a turn_left fallback fired here, so an adjacent
        # (e.g. one-step turn_right) view was reached via turn_left + turn_right*2.
        while action == self.controller.A["stop"] and self.search_mode != "gt_goal":
            # topo: on arrival, turn in place to FACE this node's best-scoring view (the detector
            # runs every step, so it inspects that direction) before advancing.
            if self.search_mode == "topo" and getattr(self, "_face_dir", None) is not None:
                self._topo_arrival_latch = (
                    getattr(self, "_cur_node", None), self._face_dir
                )
                fa = self._face_view()
                if fa is not None:
                    return fa                        # still turning to face the view
                # FACED (yaw aligned): mark visited NOW (not at selection), then re-score it from
                # the live image (dynamic update) before choosing the next subgoal.
                self._complete_topology_view(obs)
            nxt = self._next_search_goal()           # next topo-plan subgoal (sets _face_dir) or random
            if nxt is None:                          # high-level plan exhausted -> fall back
                nxt = self._sample_navigable()
            self.explore_goal = nxt
            self.subgoal = self.explore_goal
            action = controller_action()
            # If STOP again with a topo view to face (same-node next subgoal), loop back and turn
            # toward it. Otherwise (random fallback, no view) nudge with a turn to keep moving.
            if action == self.controller.A["stop"] and not (
                    self.search_mode == "topo" and getattr(self, "_face_dir", None) is not None):
                action = self.controller.A["turn_left"]
                break
        return action

    def _face_view(self):
        """Turn in place toward the current subgoal's best view yaw (dir index * VIEW_STEP_DEG,
        world; dir_00 = yaw 0, turn_right = +yaw). Returns a turn action, or None when facing is
        within one turn step or the turn cap is hit. Every intermediate yaw actually observed at
        the node is marked visited before the next turn, since detection already ran on that view."""
        _, heading_rad = self._pose()
        self._mark_observed_topology_yaw(heading_rad)
        desired = self._face_dir * self.VIEW_STEP_DEG
        delta = (desired - np.degrees(heading_rad) + 180.0) % 360.0 - 180.0
        cap = getattr(self, "max_face_turns", int(round(360.0 / self.turn_angle)) + 1)
        # Use the shared half-turn + 5 deg deadband: close enough to the requested view to
        # inspect it, with enough slack to avoid a boundary flip under pose/yaw noise.
        tolerance = self.turn_angle / 2.0 + 5.0
        if abs(delta) <= tolerance or self._face_view_turns >= cap:
            return None
        if self._face_view_turns == 0:
            print(f"    [topo] arrived; facing best view dir{self._face_dir} "
                  f"(yaw {desired:.0f} deg, delta {delta:+.0f})", flush=True)
        self._face_view_turns += 1
        return self.controller.A["turn_right"] if delta > 0 else self.controller.A["turn_left"]

    # ------------------------------------------------------------------ #
    def _free_labels(self):
        """Connected components of the occ-map free space (cached per episode).

        Reachability == connectivity on the grid, so a candidate cell is reachable
        iff it shares the agent's component. Avoids the isolated-pocket trap that
        a plain Euclidean nearest-free lookup falls into.
        """
        if self._free_lab is None:
            from scipy.ndimage import label
            planner_free = getattr(
                self.controller.map, "free_seen", self.controller.map.free)
            self._free_lab, _ = label(planner_free)
        return self._free_lab

    def _build_last_mile_source_mask(self, agent_pos, object_target):
        """Nearest safe reachable planner cells around the detected object.

        The raw detected object remains a continuous world-space target and may
        lie on an obstacle.  FMM sources, in contrast, must be robot-radius-padded
        ``free_seen`` cells in the agent's connected component.  A candidate is
        also rejected when the straight robot-to-source segment crosses unseen
        space, which prevents a source from appearing beyond the observed map.
        Seen obstacle cells on that ray are deliberately allowed: this filter
        identifies unknown wall-occluded space, while FMM remains responsible
        for finding a collision-free route. Among every valid candidate no farther
        than ``source_max_distance``, only cells from the nearest distance through
        ``nearest + source_band_width`` become sources. An empty search is an invalid
        planning state and raises instead of silently reviving a point fallback.
        """
        g = self.controller.map
        planner_free = getattr(g, "free_seen", g.free)
        seen = g.seen
        lab = self._free_labels()

        arc = g.point_to_grid(agent_pos)
        ar = int(np.clip(round(arc[0]), 0, g.H - 1))
        ac = int(np.clip(round(arc[1]), 0, g.W - 1))
        if lab[ar, ac] == 0:
            ar, ac = g.nearest_free(ar, ac)
        agent_label = int(lab[ar, ac])
        sources = np.zeros_like(planner_free, dtype=bool)
        if agent_label == 0:
            return sources

        target = np.asarray(object_target, dtype=np.float64)
        trc = g.point_to_grid(target)
        tr, tc = float(trc[0]), float(trc[1])

        radius_m = float(self.source_max_distance)
        radius_cells = radius_m / float(g.res)
        pad = int(np.ceil(radius_cells)) + 1
        r0 = max(0, int(np.floor(tr)) - pad)
        r1 = min(g.H, int(np.ceil(tr)) + pad + 1)
        c0 = max(0, int(np.floor(tc)) - pad)
        c1 = min(g.W, int(np.ceil(tc)) + pad + 1)
        rr, cc = np.mgrid[r0:r1, c0:c1]
        xs = cc * float(g.res) + float(g.min_x)
        zs = rr * float(g.res) + float(g.min_z)
        d2 = (xs - float(target[0])) ** 2 + (zs - float(target[2])) ** 2
        valid = ((lab[r0:r1, c0:c1] == agent_label)
                 & planner_free[r0:r1, c0:c1]
                 & (d2 <= radius_m ** 2 + 1e-12))
        for lr, lc in np.argwhere(valid):
            sr, sc = int(r0 + lr), int(c0 + lc)
            ray = FMMController._supercover_cells(ar, ac, sr, sc)
            ray_r, ray_c = zip(*ray)
            if not np.all(seen[np.asarray(ray_r), np.asarray(ray_c)]):
                valid[lr, lc] = False

        if valid.any():
            nearest = float(np.sqrt(np.min(d2[valid])))
            cutoff = min(radius_m, nearest + float(self.source_band_width))
            selected = valid & (d2 <= cutoff ** 2 + 1e-12)
            sources[r0:r1, c0:c1] = selected
            return sources

        raise NoReachableLastMileSource(
            "no reachable last-mile source within "
            f"maximum radius {self.source_max_distance:.2f} m of object target "
            f"({target[0]:.3f}, {target[2]:.3f})"
        )

    # ------------------------------------------------------------------ #
    def _face_then_stop(self, obs=None):
        """Called when last-mile decides to STOP: first turn in place so the goal
        object sits roughly at the image center, and only then emit STOP.

        Tolerance is ``turn_angle/2 + 5 deg`` (see face_tolerance_deg in __init__): the
        extra 5 deg absorbs per-frame target jitter around the discrete half-turn boundary.
        ``max_face_turns`` caps the number of turns.
        """
        pos, heading_rad = self._pose()
        heading = np.degrees(heading_rad)
        tx, tz = float(self.last_mile_goal[0]), float(self.last_mile_goal[2])
        desired = np.degrees(np.arctan2(tx - pos[0], -(tz - pos[2])))
        delta = (desired - heading + 180.0) % 360.0 - 180.0

        if abs(delta) <= self.face_tolerance_deg or self._face_turns >= self.max_face_turns:
            if obs is not None:
                print(f"    >>> STOP: per-cam goal-px={self._per_camera_counts(obs)}")
            return 0  # centered (as much as the turn granularity allows) -> STOP
        self._face_turns += 1
        return (self.controller.A["turn_right"] if delta > 0
                else self.controller.A["turn_left"])

    # ------------------------------------------------------------------ #
    def get_action(self, obs, state_info=None):
        # Render/record diagnostics are per decision. Clear the previous step before any
        # early return; an active shield override below records both the requested and emitted
        # actions so the status bar can explain an STG/action mismatch.
        self._shield_raw_action = None
        self._shield_action = None
        # Parallel detection: start SAM 3 (GPU / server socket) on this frame's RGB NOW so it
        # overlaps the CPU particle-filter predict+update below. SAM 3 needs no pose, and the
        # localizer needs no detection, so they run concurrently; the detection helpers read
        # the pre-computed masks (joining the worker first). No-op unless the detector is SAM 3.
        self._start_segmentation(obs)
        # Observation-based localization (no-op when --localization oracle):
        # first roll the motion model forward for the action we issued last step
        # (the bump flag says whether the forward actually happened), then correct
        # with the current depth scan.
        if self.localizer is not None:
            # delta_pose = the MEASURED odometry delta (real robot; set each step via
            # set_odom_delta) or None -> the discrete action's nominal motion model (sim).
            odom_delta = getattr(self, "_odom_delta", None)
            if self._last_action is not None or odom_delta is not None:
                self.localizer.predict(self._last_action,
                                       collided=self._collided(), delta_pose=odom_delta)
            self.localizer.update(obs, depth_key="depth")
            if self._shield_map is not None:      # keep the short-window collision map fresh
                self._shield_map.update(obs, self._collided(), self._last_action,
                                        delta_pose=odom_delta)
            gt = self._gt_position()  # eval-only ATE; None on the robot
            est = self.localizer.position
            self._ate.append(float(np.hypot(gt[0] - est[0], gt[2] - est[2])) if gt is not None else float('nan'))
            if getattr(self.localizer, "_grid_done", False) and not self._grid_reported:
                self._grid_reported = True   # eval only: was the global sweep right?
                print(f"    [localization] global grid init at step "
                      f"{len(self._ate)} -> error {self._ate[-1]:.2f} m")

            # Start pose unknown: the estimate is meaningless (and often inside a
            # wall) until the cloud collapses, so spin in place to gather views
            # from a single spot instead of navigating on a bogus pose.
            if not getattr(self.localizer, "converged", True):
                self._loc_steps += 1
                if self._loc_steps < self.max_localize_steps:
                    # OPT-IN early_nav: safe pre-convergence motion from an odometry local
                    # map. Returns an action, or None to defer to the baseline spin schedule
                    # below (None when early_nav is off -> early_nav is None here).
                    if getattr(self, "early_nav", None) is not None:
                        # head toward the goal if detected (anchored in the local map)
                        gxz = self._goal_robot_xz(obs)
                        ea = self.early_nav.step(obs, self._collided(),
                                                 self._last_action, goal_robot_xz=gxz,
                                                 delta_pose=getattr(self, "_odom_delta", None))
                        if ea is not None:
                            # pre-convergence / re-localization safety: gate the forward
                            # through the pose-INDEPENDENT short-window local map too (early_nav's
                            # own check is scan-cone + full map; the short window catches fresh
                            # FOV-blind walls). Turn instead of ramming; no re-globalize here
                            # (already re-localizing). Opt-in with safe_veto.
                            if (self.safe_veto and self._shield_map is not None
                                    and ea == self.controller.A["move_forward"]
                                    and self._forward_blocked(obs)):
                                ea = self._record_shield_action(
                                    ea, self._open_turn_action(obs, ea))
                            self._last_action = ea
                            return ea
                    # First a full turn in place (cheap 360 deg of geometry), then
                    # ACTIVELY move: rotating only ever samples one viewpoint, and
                    # many spots in a repetitive floorplan look identical from a
                    # single spot. Translating breaks that ambiguity.
                    # Spin in place until the localizer has enough views for its
                    # global sweep: the sweep is far more accurate on scans taken
                    # from one spot (measured 83% vs 56% when the later frames are
                    # collected while driving on a not-yet-known pose).
                    n_spin = max(int(round(360.0 / self.turn_angle)),
                                 int(getattr(self.localizer, "grid_frames", 0)))
                    # Direction-1: if the localizer flags the current frame as too
                    # ambiguous (facing a wall) it suggests turning toward the more
                    # OPEN side to gather informative geometry; otherwise fall back to
                    # the blind spin-then-drive. Returns None once the frame is
                    # informative -> defer to the spin schedule.
                    sa = getattr(self.localizer, "suggest_action", None)
                    a = sa(obs) if sa is not None else None
                    if a is None:
                        if self._loc_steps <= n_spin:
                            a = self.controller.A["turn_left"]
                        elif self._collided() or (self._loc_steps % 9 == 0):
                            a = self.controller.A["turn_right"]
                        else:
                            a = self.controller.A["move_forward"]
                    self._last_action = a
                    return a
                print(f"    [localization] not converged after "
                      f"{self._loc_steps} steps (spread {getattr(self.localizer,'spread',-1):.2f} m)"
                      f" -- proceeding anyway")
            else:
                en = getattr(self, "early_nav", None)
                if (self.mode == "explore" and en is not None
                        and getattr(en, "_committed", False)
                        and getattr(en, "goal_odo", None) is not None
                        and (self._last_action is not None
                             or getattr(self, "_odom_delta", None) is not None)):
                    en.integrate_pending_motion(
                        self._last_action, self._collided(),
                        getattr(self, "_odom_delta", None))
                self._handoff_early_last_mile()
                if self._loc_steps and not self._loc_reported:
                    self._loc_reported = True
                    print(f"    [localization] converged after {self._loc_steps} spin steps"
                          f" (error {self._ate[-1]:.2f} m)")

        # Online (dynamic) occupancy update: once the pose is trustworthy, fuse the
        # current scan into the PLANNING map so the controller avoids NEW obstacles
        # not in the base map. Skipped while the localizer is still converging.
        if self.dynamic_map:
            loc_ok = self.localizer is None or bool(getattr(
                self.localizer, self.dynmap_gate, getattr(self.localizer, "converged", False)))
            if loc_ok:
                pos, yaw = self._pose()
                gmap = self.controller.map
                h_min = (self.dynmap_hmin if self.dynmap_hmin is not None
                         else getattr(gmap, "obstacle_height_min", 0.2))
                scan = scan_from_depth(
                    obs["depth"], self.fx, self.fy, self.cx, self.cy,
                    self.max_depth, self.sensor_height,
                    h_min, getattr(gmap, "obstacle_height_max", 1.0),
                    subsample=4, tilt_deg=np.degrees(self.sensor_tilt), max_range=5.0,
                    sensor_position=self.sensor_position)
                if gmap.update_dynamic(scan, float(pos[0]), float(pos[2]), float(yaw)):
                    self.controller.fmm_dist = None   # map changed -> replan next
                    self._free_lab = None
                    self._last_mile_source_mask = None

        action = self._decide_action(obs, state_info)
        converged = (self.localizer is not None
                     and getattr(self.localizer, "converged", False))
        # collision shield (opt-in): veto any forward that would ram + wall-follow around it,
        # re-globalizing only when it can never clear.
        if self.safe_veto and converged:
            action = self._safety_shield(obs, action)
        # divergence guard runs ALONGSIDE the shield (complementary, not replaced): the shield
        # prevents PREDICTED rams, while this still counts UNPREDICTED collisions (side obstacle
        # outside the scan cone) as mis-convergence evidence. The localizer's own likelihood-based
        # re-globalize is separate again and always on (see localizer.update). Re-reads the LIVE
        # converged flag, so it auto-skips the step the shield just re-globalized on.
        if (self.diverge_guard and self.localizer is not None
                and getattr(self.localizer, "converged", False)):
            if self.mode == "last_mile" and self._near_goal():
                self._badstep = 0     # approaching the object; grazes here aren't mis-convergence
                bad = False
            else:
                bad = self._collided() or (action == self.controller.A["move_forward"]
                                           and self._forward_blocked_scan(obs))
                self._badstep = self._badstep + 1 if bad else 0
            if bad and self._badstep >= self.diverge_bad_hold:
                print(f"    [divergence] converged policy hit obstacles {self._badstep}x"
                      f" -> re-globalize + re-localize")
                self.localizer._reglobalize()
                self._badstep = 0; self._loc_steps = 0; self._loc_reported = False
                self.mode = "explore"; self._det_count = 0
                if getattr(self, "early_nav", None) is not None:
                    self.early_nav.reset_map()
                if getattr(self, "_shield_map", None) is not None:
                    self._shield_map.reset()
                action = self.controller.A["turn_left"]      # turn instead of ramming
        self._last_action = action     # motion model input for the next predict()
        return action

    def _forward_blocked_scan(self, obs):
        """current depth scan says a wall is within one step directly ahead (robot frame)."""
        if self.localizer is None:
            return False
        scan = self.localizer.extract_scan(obs, "depth")
        if len(scan) == 0:
            return False
        ahead = scan[(np.abs(scan[:, 0]) < self.agent_radius + 0.12) & (scan[:, 1] < -0.05)]
        return (len(ahead) > 0
                and float(np.abs(ahead[:, 1]).min()) < (self.forward_step + self.agent_radius + 0.15))

    def _forward_blocked(self, obs):
        """Would a forward step hit something? The live scan cone OR (when enabled) the
        short-window odometry local map's swept body footprint -- the latter remembers
        nearby walls now outside the camera FOV and is pose-independent, so it catches the
        grazing / mis-localized collisions the single-frame cone misses. Used by the shield."""
        if self._forward_blocked_scan(obs):
            return True
        sm = getattr(self, "_shield_map", None)   # short-window odo local map (FOV-blind walls)
        if sm is not None:
            return sm.forward_blocked()
        return False

    def _near_goal(self):
        """In last-mile, is the robot within last_mile_veto_off of the (believed) detected
        goal? If so the object may be against a wall and grazing is the expected cost of
        reaching it -> the shield veto and dguard ram-count are suspended there."""
        g = getattr(self, "last_mile_goal", None)
        if g is None:
            return False
        p = self.localizer.position if self.localizer is not None else self._pose()[0]
        return float(np.hypot(p[0] - g[0], p[2] - g[2])) <= self.last_mile_veto_off

    # ------------------------------------------------------------------ #
    def _open_turn_action(self, obs, fmm_action):
        """Turn toward the more OPEN side (scan mean range L vs R); on a tie, follow the
        goal direction (the FMM's own desired turn). Chosen once per maneuver."""
        A = self.controller.A
        scan = self.localizer.extract_scan(obs, "depth") if self.localizer is not None else None
        goal_turn = fmm_action if fmm_action in (A["turn_left"], A["turn_right"]) else None
        if scan is None or len(scan) < 3:
            return goal_turn if goal_turn is not None else A["turn_left"]
        x = scan[:, 0]; rng = np.hypot(scan[:, 0], scan[:, 1])   # robot frame: x<0 left, x>0 right
        left = rng[x < 0].mean() if (x < 0).any() else 0.0
        right = rng[x > 0].mean() if (x > 0).any() else 0.0
        if abs(left - right) < 0.3 and goal_turn is not None:    # tie -> head toward the goal
            return goal_turn
        return A["turn_left"] if left >= right else A["turn_right"]

    def _shield_reglobalize(self):
        """safe_veto escalation: the shield can't get around -> the pose belief is likely
        wrong -> throw away convergence and re-localize from scratch (same as diverge_guard,
        but reached WITHOUT ramming -- via persistent map/scan disagreement)."""
        print("    [safe_veto] cannot clear obstacles -> re-globalize + re-localize")
        self.localizer._reglobalize()
        self._shield_persist = 0; self._avoid_mode = False; self._avoid_dir = None
        self._badstep = 0; self._loc_steps = 0; self._loc_reported = False
        self.mode = "explore"; self._det_count = 0
        if getattr(self, "early_nav", None) is not None:
            self.early_nav.reset_map()
        if getattr(self, "_shield_map", None) is not None:
            self._shield_map.reset()

    def _record_shield_action(self, requested_action, emitted_action):
        """Record an action replacement for this step's renderer/recorder diagnostics."""
        if emitted_action != requested_action:
            self._shield_raw_action = requested_action
            self._shield_action = emitted_action
        else:
            self._shield_raw_action = None
            self._shield_action = None
        return emitted_action

    def _safety_shield(self, obs, fmm_action):
        """Post-convergence collision shield (opt-in, --safe_veto). Never emit a forward the
        live scan says will hit; instead wall-follow -- turn to a clear heading, then ADVANCE
        along it (position MUST change to escape, not just heading) -- until the goal
        direction is open, then hand control back to FMM. Turns don't translate in habitat,
        so every forward stays veto-checked and episode collisions go toward zero.

        Re-globalize trigger: `_shield_persist` counts the WHOLE avoidance maneuver (turns
        AND advances while wall-following) and resets only when control returns to FMM (the
        goal direction is clear again). So `safe_veto_hold` (~20) means "the maneuver ran ~20
        steps without ever re-opening the goal direction => the pose is likely wrong"."""
        A = self.controller.A
        fwd = A["move_forward"]
        if self.mode == "last_mile" and self._near_goal():   # close approach: allow grazing
            self._avoid_mode = False; self._shield_persist = 0
            return self._record_shield_action(fmm_action, fmm_action)
        # explore, OR last-mile still FAR from the goal (that is navigation) -> apply the veto
        blocked = self._forward_blocked(obs)   # scan cone OR short-window odo local map

        if self._avoid_mode:
            # leave the maneuver only when FMM wants forward AND it is actually clear:
            # the robot is now pointed along the goal path and the path is open.
            if fmm_action == fwd and not blocked:
                self._avoid_mode = False; self._shield_persist = 0
                return self._record_shield_action(fmm_action, fwd)
            self._shield_persist += 1                     # count the whole maneuver (turns + advances)
            if self._shield_persist >= self.safe_veto_hold:
                self._shield_reglobalize()
                return self._record_shield_action(fmm_action, A["turn_left"])
            shield_action = self._avoid_dir if blocked else fwd
            return self._record_shield_action(fmm_action, shield_action)

        # not maneuvering: intervene ONLY if FMM's action would ram
        if fmm_action == fwd and blocked:
            self._avoid_mode = True
            self._avoid_dir = self._open_turn_action(obs, fmm_action)
            self._shield_persist = 1
            return self._record_shield_action(fmm_action, self._avoid_dir)
        self._shield_persist = 0
        return self._record_shield_action(fmm_action, fmm_action)

    # ------------------------------------------------------------------ #
    def _clear_goal_jump_candidate(self):
        self._goal_jump_candidate = None
        self._goal_jump_count = 0

    def _refine_last_mile_goal(self, target):
        """Update the last-mile target, requiring repeated evidence for a large correction.

        Returns True only when ``last_mile_goal`` was updated. A single jump beyond
        ``goal_jump_reject`` remains ignored, while two consecutive large detections within
        ``goal_jump_consistency`` rebase the stale goal to the latest observation.
        """
        target = np.asarray(target, dtype=np.float64)
        if self.last_mile_goal is None:
            self.last_mile_goal = target.copy()
            self._clear_goal_jump_candidate()
            return True

        jump = float(np.linalg.norm((target - self.last_mile_goal)[[0, 2]]))
        if jump <= self.goal_jump_reject:
            a = self.goal_ema
            self.last_mile_goal = (1.0 - a) * self.last_mile_goal + a * target
            self._clear_goal_jump_candidate()
            return True

        pending = self._goal_jump_candidate
        consistent = (pending is not None and
                      float(np.linalg.norm((target - pending)[[0, 2]]))
                      <= self.goal_jump_consistency)
        self._goal_jump_count = self._goal_jump_count + 1 if consistent else 1
        self._goal_jump_candidate = target.copy()
        if self._goal_jump_count < self.goal_jump_confirmations:
            return False

        self.last_mile_goal = target.copy()
        self._clear_goal_jump_candidate()
        print(f"    [last_mile] accepted repeated target correction ({jump:.2f}m jump)")
        return True

    def _cancel_last_mile(self):
        """Drop a stale detected target and resume the interrupted search policy."""
        self.mode = "explore"
        self.last_mile_goal = None
        self._last_mile_detection_backed = False
        self.subgoal = getattr(self, "explore_goal", None)
        self._last_mile_source_mask = None
        self._lost_detection_count = 0
        self._clear_goal_jump_candidate()
        self._prev_xz = None
        self._stuck = 0
        self._face_turns = 0
        print("    [last-mile] target lost; resuming prior/score search")

    # ------------------------------------------------------------------ #
    def _decide_action(self, obs, state_info=None):
        target = self._detect_target(obs)

        if self.mode == "explore":
            if target is not None:
                self._det_count += 1
                if self._det_count >= self.goal_count_threshold:
                    self.mode = "last_mile"
                    self._last_mile_detection_backed = True
                    self._topo_arrival_latch = None
                    self.last_mile_goal = target
                    self._lost_detection_count = 0
                    self._clear_goal_jump_candidate()
                    self.subgoal = target
                    print(f"    >>> LAST-MILE START: '{self.goal_category}' detected @ "
                          f"{np.round(target, 2)}  per-cam goal-px={self._per_camera_counts(obs)}")
            else:
                self._det_count = 0
            if self.mode == "explore":
                action = self._explore_action(obs)
                if self.search_mode == "gt_goal" and action == self.controller.A["stop"]:
                    # Reached the GT viewpoint but never detected the object.
                    if self.gt_goal_on_miss == "gt_fallback":
                        # Last-mile toward the KNOWN GT object position.
                        self.mode = "last_mile"
                        self._last_mile_detection_backed = False
                        self._topo_arrival_latch = None
                        self.last_mile_goal = np.array(self.goal_pos, dtype=np.float64)
                        print(f"    >>> LAST-MILE: '{self.goal_category}' not detected; "
                              f"using GT object pos {np.round(self.goal_pos, 2)}")
                    else:
                        # Default: keep searching by random exploration until the
                        # detector fires (so last-mile/STOP only follow a detection).
                        self.search_mode = "random"
                        self.explore_goal = self._sample_navigable()
                        self.subgoal = self.explore_goal
                        print(f"    >>> GT goal reached without detecting "
                              f"'{self.goal_category}' -> switching to random explore")
                        return self._explore_action(obs)   # random branch never STOPs
                else:
                    return action

        # last_mile mode
        if target is not None:  # EMA for normal updates; repeated evidence for a large correction
            self._last_mile_detection_backed = True
            self._lost_detection_count = 0
            self._refine_last_mile_goal(target)
        else:
            self._clear_goal_jump_candidate()  # confirmations must be consecutive detections
            if getattr(self, "_last_mile_detection_backed", True):
                self._lost_detection_count = int(
                    getattr(self, "_lost_detection_count", 0)) + 1
                if self._lost_detection_count >= int(
                        getattr(self, "lost_detection_steps", 500)):
                    self._cancel_last_mile()
                    return self._explore_action(obs)

        pos, _ = self._pose()
        # The object estimate stays at its detected world position even when that
        # point is on furniture. Reaching the direct object-distance threshold is
        # an arrival even when pose/grid quantisation leaves the current cell outside
        # the FMM source mask. Source-cell arrival below uses the same face-then-stop
        # path, so either terminal condition has identical final alignment behaviour.
        self.subgoal = self.last_mile_goal
        object_dist = float(np.hypot(
            pos[0] - self.last_mile_goal[0],
            pos[2] - self.last_mile_goal[2],
        ))
        if object_dist <= getattr(self, "object_stop_distance", self.stop_distance):
            return self._face_then_stop(obs)

        # Otherwise plan to the nearest radius-padded, reachable source band around
        # the object, avoiding a discontinuous choice between fixed-radius tiers.
        # Visibility is defined from the robot's current cell, so refresh even
        # when the object estimate itself did not move this step.
        try:
            self._last_mile_source_mask = self._build_last_mile_source_mask(
                pos, self.last_mile_goal)
        except NoReachableLastMileSource:
            self._episode_failure_reason = (
                "no_reachable_last_mile_source "
                f"target=({self.last_mile_goal[0]:.3f},{self.last_mile_goal[2]:.3f}) "
                f"pose=({pos[0]:.3f},{pos[2]:.3f}) "
                f"max_radius={self.source_max_distance:.2f}m"
            )
            print(f"    [episode failure] {self._episode_failure_reason}", flush=True)
            return self.controller.A["stop"]

        sources = self._last_mile_source_mask
        action = self.controller.get_next_action(
            self.last_mile_goal, goal_mask=sources)
        if action == self.controller.A["stop"]:   # controller arrived/stuck
            return self._face_then_stop(obs)

        # Arrival failsafe: objects sit on furniture, so their nearest navigable
        # cell may be > stop_distance away; if we can no longer get closer (no XZ
        # movement for a while), we've reached the closest standpoint -> STOP.
        if self._prev_xz is not None and np.linalg.norm(pos[[0, 2]] - self._prev_xz) < 0.02:
            self._stuck += 1
        else:
            self._stuck = 0
        self._prev_xz = pos[[0, 2]]
        if self._stuck >= 12:
            return self._face_then_stop(obs)
        return action
