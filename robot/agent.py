"""RobotAgent = habitat-free REUSE of the sim's LastMileFMMAgent.

The whole decision pipeline -- get_action / _decide_action / detection & back-
projection / exploration / last-mile / the FMM controller / the particle-filter
localizer -- is INHERITED unchanged from `LastMileFMMAgent`, so shared decision-code
changes are picked up automatically. Bit-identical decisions require the same config,
observations, motion deltas, collision signals, and detector outputs; the deployed
robot intentionally differs from the normal sim run in several of those inputs.

Only the habitat I/O boundary is overridden here:
  * __init__ / reset : build the same state from the robot config (no habitat config
    / sim), loading the base map from a path.
  * _sample_navigable: a random explorable occupancy-map cell (no sim pathfinder).
  * _collided        : the last forward's bump flag, supplied from odometry.
  * _gt_position     : None (no oracle pose on a robot).
  * detector 'sam3'  : in-process Sam3Detector (same .segment interface).

CONTRACT: obs handed to get_action must be in HABITAT format -- rgb uint8 and depth
NORMALIZED to [0,1] (the localizer/detector multiply by max_depth internally). The
robot sensor layer converts metric depth -> depth / max_depth before calling.
"""
import gzip
import pickle
import numpy as np

from navigation.global_occupancy_map import GlobalOccupancyMap
from navigation.fmm_controller import FMMController
from navigation.localizer import make_localizer
from agents.last_mile_fmm_agent import LastMileFMMAgent
from utils.sam3_prompt import prompt as SAM3_PROMPT


class RobotAgent(LastMileFMMAgent):
    def __init__(self, cfg):
        self.cfg = cfg
        m = cfg.main_cam
        # --- specs the inherited logic reads (mirror LastMileFMMAgent.__init__) ---
        self.agent_radius = cfg.agent_radius_m
        self.turn_angle = cfg.motion.turn_angle_deg
        self.forward_step = cfg.motion.forward_step_m
        self.success_distance = cfg.detection.stop_distance_m
        self.localization = "particle_filter"
        self.dynamic_map = cfg.dynamic_map
        self.dynmap_kwargs = {}
        self.dynmap_hmin = None
        self.dynmap_gate = cfg.dynmap_gate
        self.max_localize_steps = cfg.max_localize_steps
        self.loc_overrides = dict(cfg.loc_overrides)
        self.ctrl_overrides = {}
        # early_nav / dguard / safe_veto flags use the SAME fallbacks as the sim. These are
        # nav-extension defaults only; PF overrides and real sensor/motion inputs can differ.
        self._init_nav_extensions(cfg)

        def cam_dict(spec, suffix):
            return {"name": "top" if suffix else "main",
                    "rgb": "rgb" + suffix, "depth": "depth" + suffix, "sem": "semantic" + suffix,
                    "W": spec.width, "H": spec.height, "max_depth": spec.max_depth_m,
                    "height": spec.height_m,
                    "position": np.array([spec.right_offset_m, spec.height_m,
                                          -spec.forward_offset_m], dtype=np.float64),
                    # Habitat camera convention: negative X rotation pitches downward.
                    "tilt": -np.deg2rad(spec.tilt_deg),
                    "fx": spec.fx, "fy": spec.fy, "cx": spec.cx, "cy": spec.cy}
        self.det_cams = [cam_dict(m, "")] + (
            [cam_dict(cfg.sub_cam, "_top")] if cfg.sub_cam is not None else [])
        self.detect_camera = "auto"
        self.W, self.H = m.width, m.height
        self.max_depth = m.max_depth_m
        self.sensor_height, self.sensor_tilt = m.height_m, np.deg2rad(m.tilt_deg)
        self.sensor_position = np.array(
            [m.right_offset_m, m.height_m, -m.forward_offset_m], dtype=np.float64)
        self.fx, self.fy, self.cx, self.cy = m.fx, m.fy, m.cx, m.cy
        self.rgb_key, self.depth_key, self.sem_key = "rgb", "depth", "semantic"

        det = cfg.detection
        self.detector = det.detector
        self.sam3 = None
        if self.detector == "sam3":
            from robot.sam3_detector import Sam3Detector
            self.sam3 = Sam3Detector(cfg.sam3.bpe_path, cfg.sam3.device, cfg.sam3.confidence)
        self.min_goal_pixels = det.min_goal_pixels
        self.last_mile_range = det.last_mile_range_m
        self.goal_count_threshold = det.goal_count_threshold
        self.lost_detection_steps = det.lost_detection_steps
        self.goal_ema = det.goal_ema
        self.goal_jump_reject = det.goal_jump_reject_m
        self.goal_jump_confirmations = det.goal_jump_confirmations
        self.goal_jump_consistency = det.goal_jump_consistency_m
        self.stop_distance = det.stop_distance_m
        self.object_stop_distance = det.object_stop_distance_m
        # Default = half a turn step + 5 deg: the slack prevents boundary oscillation while
        # preserving heading corrections larger than 20 deg (matches sim + early_nav).
        self.face_tolerance_deg = (det.face_tolerance_deg if det.face_tolerance_deg is not None
                                   else self.turn_angle / 2.0 + 5.0)
        self.max_face_turns = int(round(360.0 / self.turn_angle)) + 1
        self.source_max_distance = det.source_max_distance_m
        self.source_band_width = det.source_band_width_m
        self.search_mode_cfg = cfg.search_mode          # 'random' (default) or 'topo'
        self.gt_goal_on_miss = "explore"
        # topo search (per-view object priors). The goal's prior is LOADED from prior_scores
        # (object-first) and, when topo_update=='view', each arrived view is re-scored live from its
        # image by a resident Qwen VLM (see LastMileFMMAgent.update_view_prior) -- fits 16 GB with SAM 3.
        self.topo_update = cfg.topo_update
        self.topo_path = cfg.topo_path
        self.scenario_topologies = {}       # the robot selects one explicit --prior topology
        self.prior_scores_path = cfg.prior_scores
        self.qwen_vlm, self.qwen_vlm_load = "Qwen/Qwen3-VL-8B-Instruct", "4bit"
        self.VIEW_STEP_DEG = 30.0
        self._upd = None
        self._room_aff = {}
        self._prior = self._prior_obj = self._topo_cached = None
        self._topo_cache = {}
        self._active_topo_path = None
        self._topo_phase = None
        self._prior_hint = None

        self.sim = None
        self._last_collided = False
        self._gmap = None
        print(f"    [robot_agent] cameras {[c['name'] for c in self.det_cams]} | detector {self.detector}")

    # ------------------------------------------------------------------ #
    def reset(self, category=None):
        """Habitat-free reset (no env/sim). Mirrors the sim agent's reset state."""
        self.goal_category = category if category is not None else self.cfg.target_category
        self.sam3_prompt = SAM3_PROMPT.get(self.goal_category, self.goal_category)
        self.goal_ids = set()                 # SAM 3 only; no semantic mapping on a robot
        self.search_mode = self.search_mode_cfg
        with gzip.open(self.cfg.map_path, "rb") as f:
            d = pickle.load(f)
        res = float(d["map_params"]["resolution_m"])
        gmap = GlobalOccupancyMap.from_occupancy_dict(d, agent_radius_m=self.agent_radius)
        self._gmap = gmap

        self.controller = FMMController(
            None, goal_radius=self.stop_distance, resolution_m=res,
            agent_radius=self.agent_radius, turn_angle=self.turn_angle,
            forward_step_size=self.forward_step, **self.ctrl_overrides)
        self.controller.reset(np.zeros(3), occ_map=gmap)
        if self.dynamic_map:
            gmap.enable_dynamic(**self.dynmap_kwargs)

        self.localizer = make_localizer(
            "particle_filter", fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy, img_height=self.H,
            max_depth=self.max_depth, sensor_height=self.sensor_height,
            sensor_position=self.sensor_position,
            tilt_deg=np.degrees(self.sensor_tilt),
            forward_step=self.forward_step, turn_angle=self.turn_angle,
            actions=self.controller.A,
            **self.loc_overrides)   # start is ALWAYS unknown -> global MCL (the only mode)
        self.localizer.reset(gmap)
        self.controller.set_pose_source(self.localizer)

        # early_nav + collision-shield local map + their state (shared builder with the sim)
        self._build_early_nav()
        self._last_action = None
        self._ate = []
        self._loc_steps = 0
        self._loc_reported = False
        self._grid_reported = False
        self._prior = self._prior_obj = None               # reloaded for this goal (topo mode)
        self._visited = None                               # visited-graph rebuilt on first select
        self._cur_node = None
        self._face_dir = None
        self._topo_arrival_latch = None
        self._topo_phase = None
        self._prior_hint = None
        if self.search_mode == "topo":
            self._prepare_topology_search("")
        self._face_view_turns = 0
        self.explore_goal = self._next_search_goal()       # 'topo' -> hint then score, else random
        if self.explore_goal is None:
            self.explore_goal = self._sample_navigable()
        self.goal_pos = None
        self.mode = "explore"
        self.last_mile_goal = None
        self._last_mile_detection_backed = False
        self._goal_jump_candidate = None
        self._goal_jump_count = 0
        self.subgoal = self.explore_goal
        self._det_count = 0
        self._lost_detection_count = 0
        self._prev_xz = None
        self._stuck = 0
        self._face_turns = 0
        self._free_lab = None
        self._last_mile_source_mask = None
        self._episode_failure_reason = None

    # ---- sim I/O boundary overrides ---------------------------------- #
    def _sample_navigable(self):
        g = self._gmap
        base = getattr(g, "explorable", g.free)
        ko = getattr(g, "keepout", None)     # never wander into a planning keep-out
        if ko is not None and ko.any():
            base = base & ~ko
        fr, fc = np.nonzero(base)
        i = np.random.randint(len(fr))
        x, z = g.grid_to_world_xz(fr[i], fc[i])
        return np.array([x, 0.0, z], np.float32)

    def _collided(self):
        return bool(self._last_collided)

    def _gt_position(self):
        return None

    def set_collided(self, collided):
        """Called by the driver each step (from odometry: forward commanded but the
        robot did not move -> collided)."""
        self._last_collided = bool(collided)
