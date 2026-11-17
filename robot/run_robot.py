"""Entry point: run the sim objnav agent on the real Jackal, habitat-free.

  python robot/run_robot.py --config configs/jackal.yaml

The decision code is the sim's LastMileFMMAgent, REUSED via RobotAgent (localize ->
optional prior hint -> score-based topology search -> SAM 3 last-mile -> STOP). A controlled
matched-input check verifies the shared code can decide bit-identically to run_eval;
the deployed robot's sensors, motion inputs, detector, and PF overrides intentionally
differ from the default sim run. Shared-agent changes are still picked up automatically.
Sensors (odom / RGB / depth) and the three discrete motion goals go
over one SSH channel to the Jackal (robot/jackal.py): the Jackal HARDWARE + ROS
settings live there, the task/nav knobs in configs/jackal.yaml.
"""
import argparse
from dataclasses import asdict
import os
from pathlib import Path
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _project_path(value):
    """Resolve a path setting; relative values are taken from the repository root."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


# SAM 3 source checkout (default: <repo>/third_party/sam3). Override with $SAM3_ROOT.
SAM3_ROOT = _project_path(os.environ.get("SAM3_ROOT", "third_party/sam3"))
# Optional interpreter to re-exec into (e.g. a separate SAM 3 conda env). Unset = current Python.
SAM3_PYTHON = (_project_path(os.environ["SAM3_PYTHON"])
               if os.environ.get("SAM3_PYTHON") else None)
SAM3_BPE = SAM3_ROOT / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"


def odom_to_robot_delta(prev, cur):
    """Two (x, y, yaw) odom poses (fixed ROS frame) -> the robot-frame motion delta
    (d_fwd, d_lat, dyaw) the localizer's motion model expects. Standard get_rel_pose_change
    (references/pose_utils), then map ROS (x forward, y LEFT, yaw CCW+) to the localizer
    convention where turn_left = NEGATIVE yaw -- which mirrors the lateral axis too.
    NOTE: verify the signs on the robot (drive forward -> the estimate advances; turn left ->
    the estimate turns left); flip d_lat / dyaw here if a run shows them reversed."""
    import math
    px, py, po = prev
    cx, cy, co = cur
    dx, dy = cx - px, cy - py
    dist = math.hypot(dx, dy)
    theta = math.atan2(dy, dx) - po                       # motion heading in the prev robot frame
    dyaw_ros = (co - po + math.pi) % (2 * math.pi) - math.pi
    return (dist * math.cos(theta),                       # d_fwd (same sign as ROS forward)
            -dist * math.sin(theta),                      # d_lat (ROS +left -> localizer mirror)
            -dyaw_ros)                                    # dyaw  (ROS CCW+ -> localizer turn_left=-)


def ensure_sam3_runtime():
    """Re-exec with $SAM3_PYTHON when set, then make the SAM3 source importable."""
    expected = SAM3_PYTHON.resolve() if SAM3_PYTHON is not None else None
    current = Path(sys.executable).resolve()
    if expected is not None and current != expected:
        if not expected.is_file():
            raise RuntimeError(f"SAM3 conda Python not found: {expected}")
        os.execv(
            str(expected),
            [str(expected), str(Path(__file__).resolve()), *sys.argv[1:]],
        )

    for path in (PROJECT_ROOT, SAM3_ROOT):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    if not SAM3_BPE.is_file():
        raise RuntimeError(f"SAM3 BPE vocabulary not found: {SAM3_BPE}")


ACTION = {0: "stop", 1: "forward", 2: "turn left", 3: "turn right"}


def snapshot_observation(obs, cfg, backend, np):
    """Copy camera data before the next refresh overwrites the backend cache."""
    frames = []
    for cam in cfg.cameras:
        suffix = "" if cam.name == "main" else "_top"
        frames.append((
            cam,
            np.asarray(obs["rgb" + suffix])[..., :3].astype(np.uint8).copy(),
            np.asarray(backend.get_depth(cam.name), dtype=np.uint16).copy(),
        ))
    return frames


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/jackal.yaml")
    ap.add_argument("--goal", default=None,
                    help="goal object category (SAM 3 prompt), e.g. --goal chair. This "
                         "is the way to set the goal -- no yaml edit needed. Omit for "
                         "go-to-pose (drive to the yaml `goal` x/z).")
    ap.add_argument("--prior", type=int, choices=(1, 2, 3), default=None,
                    help="stored scene prior topology to visit before score search; "
                         "omit to use base_topology.pkl and start with score search")
    ap.add_argument("--jackal-host", default="jackalw",
                    help="SSH host/alias for the Jackal")
    ap.add_argument("--max_steps", type=int, default=500)
    ap.add_argument("--render", action="store_true",
                    help="show a live window: main/sub rgb+depth, ego scan, top-down "
                         "localization + FMM/dynamic map (needs a display)")
    ap.add_argument("--record", nargs="?", const="__auto__", default=None,
                    help="record metadata, step logs, rendered video, and raw camera frames "
                         "in one run directory; bare --record creates an automatic path")
    args = ap.parse_args(argv)
    return args


def topology_path_for_prior(prior):
    """Absolute topology selected by ``--prior``; no selection means the base graph."""
    if prior is not None and prior not in (1, 2, 3):
        raise ValueError("prior must be one of 1, 2, 3, or None")
    name = "base_topology.pkl" if prior is None else f"prior{prior}_topology.pkl"
    return str(PROJECT_ROOT / "topo_data" / "topo_graph" / name)


def configure_topology_search(cfg, prior):
    """Apply the run_robot topology policy after loading the YAML config."""
    cfg.search_mode = "topo"
    cfg.topo_path = topology_path_for_prior(prior)


def current_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def close_render_windows(cv2_module, render_enabled):
    """Avoid GUI calls in record-only runs that use headless OpenCV."""
    if cv2_module is not None and render_enabled:
        cv2_module.destroyAllWindows()


def format_step_status(agent, step, action_name):
    """One-line live status with both coarse mode and the derived navigation stage."""
    from utils.panels import navigation_stage

    pos, yaw = agent._pose()
    return (f"[{step}] mode={agent.mode} stage={navigation_stage(agent)} act={action_name} "
            f"converged={agent.localizer.converged} "
            f"pos=({pos[0]:.2f},{pos[2]:.2f}) yaw={yaw:.2f}")


def classify_agent_stop(agent):
    """Distinguish a successful navigation STOP from an agent-declared failure STOP."""
    failure_reason = getattr(agent, "_episode_failure_reason", None)
    if failure_reason:
        return False, "planning_failure", failure_reason
    return True, "stop", None


def main(args=None):
    args = get_args() if args is None else args
    # Importing this module is side-effect free so its CLI can be regression-tested.
    # Only an actual robot run switches to the deployment SAM3 environment.
    ensure_sam3_runtime()
    from robot.config import RobotConfig
    from robot.robot_env import RobotEnv
    from robot.agent import RobotAgent
    from robot.jackal import JackalConnection, JackalConnectionConfig

    # configs/jackal.yaml is the single source for the map + cameras (topics /
    # intrinsics / resolution / depth / mounting) + task; JackalConnectionConfig adds
    # only the SSH + low-level control constants. The backend reads its camera
    # topics/resolution from the same cfg, so nothing can drift.
    cfg = RobotConfig.load(args.config)
    configure_topology_search(cfg, args.prior)
    hw = JackalConnectionConfig(host=args.jackal_host, **cfg.drive)
    # goal category from the CLI (--goal); yaml `target.category` is only a fallback,
    # so changing the goal never needs a yaml edit.
    category = args.goal if args.goal is not None else cfg.target_category
    if cfg.detection.detector == "sam3":
        cfg.sam3.bpe_path = str(SAM3_BPE)
        print(f">>> SAM3 runtime | python {sys.executable} | source {SAM3_ROOT}")

    renderer = cv2 = np = None
    render_hfov = None
    record_dir = None
    if args.render or args.record:
        import numpy as _np
        import cv2 as _cv2
        np = _np
        cv2 = _cv2
        from robot.render import RobotRenderer, show_frame
        m = cfg.main_cam
        render_hfov = float(np.degrees(2.0 * np.arctan(m.width / (2.0 * m.fx))))
    if args.record:
        from robot.recording import resolve_record_dir
        record_dir = resolve_record_dir(
            args.record, project_root=PROJECT_ROOT, category=category, prior=args.prior,
        )

    backend = None
    recorder = None
    arrived = False
    stop_reason = "max_steps"
    total_steps = 0
    run_error = None
    failure_reason = None
    try:
        backend = JackalConnection(cfg, settings=hw)  # SSH sensors + motion goals
        env = RobotEnv(cfg, backend)
        agent = RobotAgent(cfg)                        # REUSE the sim decision code
        agent.reset(category=category)
        if render_hfov is not None:
            # RobotRenderer reads agent.localizer during construction, so create it
            # only after the agent exists and has been reset.
            renderer = RobotRenderer(agent, hfov_deg=render_hfov)
        if record_dir is not None:
            from robot.recording import RunRecorder
            recorder = RunRecorder(
                record_dir,
                metadata={
                    "target_category": category,
                    "sam3_prompt": getattr(agent, "sam3_prompt", category),
                    "prior": args.prior,
                    "topology_path": cfg.topo_path,
                    "prior_scores_path": cfg.prior_scores,
                    "config_path": str(Path(args.config).expanduser().resolve()),
                    "resolved_config": asdict(cfg),
                    "git_commit": current_git_commit(),
                    "command": [sys.executable, *sys.argv],
                    "jackal_host": args.jackal_host,
                    "max_steps": args.max_steps,
                },
                cv2_module=cv2,
                np_module=np,
                video_fps=4,
            )
            print(f">>> recording run to {record_dir}")

        print(f">>> jackal objnav | target '{category}' | host {args.jackal_host} "
              f"| map {cfg.map_path} | cameras {[c.name for c in cfg.cameras]}")
        obs = env.reset()
        prev_odom = env.odom_pose()                    # ROS odom pose in the fixed odom frame
        for t in range(args.max_steps):
            agent.set_collided(env.last_collided)      # odometry collision flag
            cur_odom = env.odom_pose()                 # pose after the PREVIOUS env.step
            # Feed the REQUIRED measured odom motion (not the nominal action) to
            # predict/early_nav/shield. RobotEnv raises before this point when ROS odom
            # is missing, stale, or malformed, so real runs never silently fall back.
            odom_delta = odom_to_robot_delta(prev_odom, cur_odom)
            agent.set_odom_delta(odom_delta)
            prev_odom = cur_odom
            action = agent.get_action(obs, None)       # robot decides the discrete action
            action_name = ACTION.get(action, str(action))
            print(format_step_status(agent, t, action_name))

            renderer_masks = saved_masks = detection_counts = None
            if recorder is not None:
                from robot.recording import snapshot_detection
                if category is not None:
                    renderer_masks, saved_masks, detection_counts = snapshot_detection(
                        agent, obs, np,
                    )
                else:
                    renderer_masks, saved_masks, detection_counts = {}, {}, {}

            frame = None
            if renderer is not None:                   # read-only viz (never affects decisions)
                frame = renderer.frame(
                    obs, action_name, t, sam3_masks=renderer_masks,
                )
            if recorder is not None:
                from robot.recording import build_step_record
                recorder.record_step(
                    build_step_record(
                        agent, step=t, action_id=action, action_name=action_name,
                        odom_pose=cur_odom, odom_delta=odom_delta,
                        collided=env.last_collided, detection_counts=detection_counts,
                        wall_time_s=time.time(), monotonic_s=time.monotonic(),
                    ),
                    frames=snapshot_observation(obs, cfg, backend, np),
                    masks=saved_masks,
                    render_frame=frame,
                )
            total_steps = t + 1
            if frame is not None and args.render:
                if show_frame("run_robot", frame, scale=cfg.window_scale) == 27:   # Esc quits
                    stop_reason = "operator_escape"
                    break
            if action == agent.controller.A["stop"]:
                arrived, stop_reason, failure_reason = classify_agent_stop(agent)
                if failure_reason:
                    print(f">>> failure_reason {failure_reason}")
                break
            obs = env.step(action)                     # jackal executes it closed-loop
    except BaseException as exc:
        run_error = exc
        stop_reason = "error"
        raise
    finally:
        if backend is not None:
            try:
                backend.stop()
            except Exception as exc:
                print(f">>> warning: final stop failed: {exc}")
            try:
                backend.close()
            except Exception as exc:
                print(f">>> warning: backend close failed: {exc}")
        if recorder is not None:
            try:
                recorder.finish(
                    status=("failed" if run_error is not None else "completed"),
                    stop_reason=stop_reason,
                    total_steps=total_steps,
                    error=run_error,
                    failure_reason=failure_reason,
                )
                print(f">>> saved run record: {record_dir}")
            except Exception as exc:
                print(f">>> warning: final run recording failed: {exc}")
        close_render_windows(cv2, args.render)
    if arrived:
        print(">>> STOP (goal reached)")
    elif failure_reason:
        print(f">>> stopped (planning failure: {failure_reason})")
    else:
        print(">>> stopped (max_steps)")


if __name__ == "__main__":
    main(get_args())
