"""Typed loader for the real-robot config (configs/jackal.yaml).

Habitat-free: depends only on PyYAML + numpy. The yaml is the single source of truth
for the map, the CAMERAS (topics + intrinsics + resolution + depth scale + mounting),
and the task/nav knobs (localizer, detection/SAM 3, target, dynamic map, goal).

Only the SSH connection, the fixed 0.25 m / 30 deg motion steps, and the low-level
control constants live in `robot/jackal.py` (`JackalConnectionConfig`). `robot/jackal.py`
READS the camera topics/resolution from the RobotConfig this loader produces, so
nothing is duplicated between the two.
"""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import yaml


@dataclass
class CameraSpec:
    """Per-camera spec the agent/env consume, loaded from configs/jackal.yaml (also
    constructed directly by the habitat bit-identity check with sim-derived values)."""
    name: str
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    max_depth_m: float = 10.0
    depth_scale: float = 1.0        # raw_depth * depth_scale -> metres
    height_m: float = 1.0           # optical-centre height above the floor (mounting)
    tilt_deg: float = 0.0           # downward pitch (mounting)
    forward_offset_m: float = 0.0   # base rotation centre -> optical centre, forward
    right_offset_m: float = 0.0     # base rotation centre -> optical centre, right
    topic_rgb: str = ""
    topic_depth: str = ""

    @classmethod
    def from_dict(cls, name, d):
        # Every camera field is REQUIRED in the yaml -- this per-robot hardware must be
        # read from the config, never from a silent fallback (e.g. a missing depth_scale
        # defaulting to 1.0 would put D455 16-bit-mm depth off by 1000x). A missing key
        # raises a clear error instead. (Task/tuning params keep their defaults.)
        def req(key, cast):
            if key not in d:
                raise ValueError(
                    f"configs/*.yaml cameras.{name} is missing required '{key}'")
            return cast(d[key])
        return cls(name=name, fx=req("fx", float), fy=req("fy", float),
                   cx=req("cx", float), cy=req("cy", float),
                   width=req("width", int), height=req("height", int),
                   max_depth_m=req("max_depth_m", float),
                   depth_scale=req("depth_scale", float),
                   height_m=req("height_m", float),
                   tilt_deg=req("tilt_deg", float),
                   forward_offset_m=req("forward_offset_m", float),
                   right_offset_m=req("right_offset_m", float),
                   topic_rgb=req("topic_rgb", str),
                   topic_depth=req("topic_depth", str))


@dataclass
class MotionSpec:
    """Nominal discrete steps used by the planner/controller. The Jackal targets these
    0.25 m / 30 deg goals, while localization uses measured odometry deltas when they
    are available rather than assuming the target motion was achieved exactly."""
    forward_step_m: float = 0.25
    turn_angle_deg: float = 30.0


@dataclass
class DetectionSpec:
    detector: str = "sam3"          # 'sam3' (only real option on a robot) or 'none'
    min_goal_pixels: int = 50
    last_mile_range_m: float = 5.0
    goal_count_threshold: int = 1
    lost_detection_steps: int = 500
    goal_ema: float = 0.9
    goal_jump_reject_m: float = 1.5
    goal_jump_confirmations: int = 2
    goal_jump_consistency_m: float = 0.5
    stop_distance_m: float = 0.5
    object_stop_distance_m: float = 0.7  # direct global last-mile object-arrival threshold
    face_tolerance_deg: Optional[float] = None   # None -> turn_angle/2 + 5 deg (matches agent)
    source_max_distance_m: float = 1.0     # maximum distance searched for a safe source
    source_band_width_m: float = 0.20      # nearest valid source distance + this band
    # (max_face_turns is NOT a config knob: the agent derives it from turn_angle as
    #  int(round(360/turn_angle))+1, exactly like the habitat agent.)


@dataclass
class Sam3Spec:
    bpe_path: str = "third_party/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    device: str = "cuda"
    # SAM 3 detection confidence. 0.5 = the robot_utils reference (Sam3Processor
    # default). Earlier objnav tests used 0.3 (more permissive); see robot/README.md.
    confidence: float = 0.5


@dataclass
class GoalSpec:
    x: float = 0.0
    z: float = 0.0
    yaw_deg: float = 0.0
    radius_m: float = 0.5

    @property
    def world(self):
        # map-frame (x, y=0, z); y is unused by the 2-D planner/localizer
        return np.array([self.x, 0.0, self.z], dtype=np.float64)


@dataclass
class RobotConfig:
    map_path: str
    agent_radius_m: float
    motion: MotionSpec
    goal: GoalSpec
    main_cam: CameraSpec
    sub_cam: Optional[CameraSpec] = None
    max_localize_steps: int = 80
    loc_overrides: dict = field(default_factory=dict)
    # object-goal navigation (optional). target_category set -> full pipeline
    # (topology search -> SAM 3 detection -> last-mile -> stop); else drive to `goal`.
    target_category: Optional[str] = None
    detection: DetectionSpec = field(default_factory=DetectionSpec)
    sam3: Sam3Spec = field(default_factory=Sam3Spec)
    dynamic_map: bool = False
    dynmap_gate: str = "settled"
    # Pre-convergence early_nav + post-convergence collision shield / divergence guard.
    # These nav-extension fallbacks mirror run_eval; other robot inputs/config can differ.
    early_nav: bool = True
    diverge_guard: bool = True
    diverge_bad_hold: int = 8
    safe_veto: bool = True
    safe_veto_hold: int = 20
    safe_veto_use_map: bool = True
    last_mile_veto_off: float = 1.2
    fmm_lastmile: bool = False
    combo_w_info: float = 1.0
    combo_w_sub: float = 1.0
    # live render window size as a fraction of the composed frame (display only; the saved
    # video keeps full resolution). Set via a `render: {window_scale: ...}` block.
    window_scale: float = 0.6
    # low-level drive tuning -> JackalConnectionConfig (speed, acceleration, and
    # odometry-based distance braking). Only the keys present in a `drive:`
    # block override the JackalConnectionConfig defaults.
    drive: dict = field(default_factory=dict)
    # search phase before detection (mirrors the sim's --lastmile_search). Set via a `search:`
    # block. Default 'topo' visits nodes highest-object-prior first and faces the best-scoring
    # view. The goal's prior is loaded from `prior_scores` (object-first) and, if
    # `topo_update: view`, each arrived view is re-scored live in-process (dynamic scenes).
    search_mode: str = "topo"
    topo_update: str = "off"          # 'view' = live per-view update on arrival, else stored only
    topo_path: str = "topo_data/topo_graph/base_topology.pkl"
    prior_scores: str = "topo_data/score/scores.json"

    @property
    def cameras(self):
        return [self.main_cam] + ([self.sub_cam] if self.sub_cam is not None else [])

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = yaml.safe_load(f)
        # Per-robot HARDWARE (map + cameras) is REQUIRED from the yaml -- no fallback,
        # so a missing value errors instead of silently using a wrong default. The
        # task/tuning blocks below (localizer/detection/sam3/dynamic_map/goal) keep
        # their robot/config.py defaults if omitted.
        mp = d.get("map") or {}
        for key in ("path", "agent_radius_m"):
            if key not in mp:
                raise ValueError(f"configs/*.yaml must define map.{key}")
        cams = d.get("cameras") or {}
        if "main" not in cams:
            raise ValueError("configs/*.yaml must define cameras.main")
        loc = d.get("localizer", {}) or {}
        nav = d.get("nav", {}) or {}   # early_nav / dguard / safe_veto (defaults mirror the sim)
        search = d.get("search", {}) or {}   # lastmile_search / topo_update (default topo search)
        return cls(
            map_path=str(mp["path"]),
            agent_radius_m=float(mp["agent_radius_m"]),
            motion=MotionSpec(),                  # fixed 0.25 m / 30 deg (Jackal goals)
            goal=GoalSpec(**(d.get("goal", {}) or {})),
            main_cam=CameraSpec.from_dict("main", cams["main"]),
            sub_cam=(CameraSpec.from_dict("sub", cams["sub"]) if "sub" in cams else None),
            max_localize_steps=int(loc.get("max_localize_steps", 80)),
            loc_overrides=dict(loc.get("overrides", {}) or {}),
            target_category=((d.get("target") or {}).get("category")),
            detection=DetectionSpec(**(d.get("detection", {}) or {})),
            sam3=Sam3Spec(**(d.get("sam3", {}) or {})),
            dynamic_map=bool((d.get("dynamic_map") or {}).get("enabled", False)),
            dynmap_gate=str((d.get("dynamic_map") or {}).get("gate", "settled")),
            early_nav=bool(nav.get("early_nav", True)),
            diverge_guard=bool(nav.get("diverge_guard", True)),
            diverge_bad_hold=int(nav.get("diverge_bad_hold", 8)),
            safe_veto=bool(nav.get("safe_veto", True)),
            safe_veto_hold=int(nav.get("safe_veto_hold", 20)),
            safe_veto_use_map=bool(nav.get("safe_veto_use_map", True)),
            last_mile_veto_off=float(nav.get("last_mile_veto_off", 1.2)),
            fmm_lastmile=bool(nav.get("fmm_lastmile", False)),
            combo_w_info=float(nav.get("combo_w_info", 1.0)),
            combo_w_sub=float(nav.get("combo_w_sub", 1.0)),
            window_scale=float((d.get("render") or {}).get("window_scale", 0.6)),
            drive={k: float(v) for k, v in (d.get("drive", {}) or {}).items()
                   if k in ("linear_speed", "angular_speed",
                            "linear_acceleration", "angular_acceleration",
                            "distance_brake_delay_s", "angle_brake_delay_s")},
            search_mode=str(search.get("lastmile_search", "topo")),
            # YAML may coerce a bare off/on to bool -> normalize back to 'off'/'view'
            topo_update={False: "off", True: "view"}.get(
                search.get("topo_update", "off"), str(search.get("topo_update", "off"))),
            topo_path=str(search.get(
                "topo_path", "topo_data/topo_graph/base_topology.pkl"
            )),
            prior_scores=str(search.get("prior_scores", "topo_data/score/scores.json")),
        )
