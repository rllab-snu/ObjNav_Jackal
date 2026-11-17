"""In-place 360-deg scan + capture on the real Jackal (habitat-free).

  python robot/run_scan_capture.py --config configs/jackal.yaml [--render]

Rotates in place 30 deg at a time, 12 times (= 360 deg). At each of the 12 directions it
saves the main + sub camera RGB and depth, while the SAME likelihood-based localizer as
run_robot (global-MCL particle filter) runs -- predicting each turn and updating on each
view -- so the pose is estimated from the in-place spin alone. It ONLY spins in place, so
early_nav / navigation / detection are not used (RobotAgent is built only for its localizer
+ occupancy map + camera intrinsics; get_action is never called).

Outputs (default outputs/scan-<timestamp>/):
  dir_<ii>_deg<ddd>_<main|sub>_<rgb|depth>.png   # rgb = color, depth = raw uint16 (mm)
  scan.json                                      # per-direction localizer estimate + meta
"""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import cv2

from robot.config import RobotConfig
from robot.robot_env import RobotEnv
from robot.agent import RobotAgent
from robot.jackal import JackalConnection, JackalConnectionConfig

ACTION = {0: "stop", 1: "forward", 2: "turn left", 3: "turn right"}


def odom_to_robot_delta(prev, cur):
    """(x, y, yaw) ROS-odom poses -> robot-frame (d_fwd, d_lat, dyaw). Same as run_robot."""
    px, py, po = prev
    cx, cy, co = cur
    dx, dy = cx - px, cy - py
    dist = math.hypot(dx, dy)
    theta = math.atan2(dy, dx) - po
    dyaw_ros = (co - po + math.pi) % (2 * math.pi) - math.pi
    return (dist * math.cos(theta), -dist * math.sin(theta), -dyaw_ros)


def save_views(out_dir, i, deg, obs, cfg, backend):
    """Save main + sub RGB (color) and depth (raw uint16 mm) for one direction."""
    for cam in cfg.cameras:
        suffix = "" if cam.name == "main" else "_top"
        stem = f"dir_{i:02d}_deg{deg:03d}_{cam.name}"
        rgb = np.asarray(obs["rgb" + suffix])[..., :3].astype(np.uint8)
        cv2.imwrite(str(out_dir / f"{stem}_rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        try:
            depth = np.asarray(backend.get_depth(cam.name), dtype=np.uint16)
            cv2.imwrite(str(out_dir / f"{stem}_depth.png"), depth)
        except Exception as exc:
            print(f">>> warning: depth save failed for {cam.name}: {exc}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/jackal.yaml")
    ap.add_argument("--jackal-host", default="jackalw", help="SSH host/alias for the Jackal")
    ap.add_argument("--num_dirs", type=int, default=12, help="directions (12 -> 30 deg each)")
    ap.add_argument("--turn", default="right", choices=["left", "right"],
                    help="spin direction; 'right' (+yaw, CW) so the deg label matches the "
                         "heading rotated from the start (turn_left would flip the sign)")
    ap.add_argument("--out", default=None, help="output dir (default outputs/scan-<timestamp>)")
    ap.add_argument("--render", action="store_true",
                    help="live window: rgb/depth, ego scan, top-down localization")
    ap.add_argument("--save_video", nargs="?", const="__auto__", default=None,
                    help="record the rendered frames to mp4: bare --save_video -> auto path under "
                         "outputs/, or an explicit path (independent of --render)")
    args = ap.parse_args()
    if args.save_video == "__auto__":
        args.save_video = str(PROJECT_ROOT / "outputs" /
                              ("scan-" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".mp4"))

    out_dir = Path(args.out) if args.out else (
        PROJECT_ROOT / "outputs" / ("scan-" + datetime.now().strftime("%Y%m%d-%H%M%S")))
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = RobotConfig.load(args.config)
    hw = JackalConnectionConfig(host=args.jackal_host, **cfg.drive)
    cfg.detection.detector = "none"          # localization only; no SAM 3

    renderer = viz_frames = render_hfov = None
    if args.render or args.save_video:
        from robot.render import RobotRenderer, show_frame
        m = cfg.main_cam
        render_hfov = float(np.degrees(2.0 * np.arctan(m.width / (2.0 * m.fx))))
        viz_frames = [] if args.save_video else None

    turn_deg = int(round(cfg.motion.turn_angle_deg))
    if args.num_dirs * turn_deg != 360:
        print(f">>> note: {args.num_dirs} dirs x {turn_deg} deg = {args.num_dirs * turn_deg} deg "
              f"(not a full 360)")

    backend = None
    manifest = {"config": args.config, "map": cfg.map_path, "num_dirs": args.num_dirs,
                "turn_deg": turn_deg, "turn": args.turn, "directions": []}
    try:
        backend = JackalConnection(cfg, settings=hw)
        env = RobotEnv(cfg, backend)
        agent = RobotAgent(cfg)              # built only for the localizer + occ map + intrinsics
        agent.reset(category="scan")
        loc = agent.localizer
        turn_act = agent.controller.A["turn_left" if args.turn == "left" else "turn_right"]
        if render_hfov is not None:
            # in-place scan: no navigation -> render only obs + localization groups
            renderer = RobotRenderer(agent, hfov_deg=render_hfov, groups=("obs", "localization"))

        print(f">>> jackal in-place scan | host {args.jackal_host} | map {cfg.map_path} "
              f"| {args.num_dirs} dirs x {turn_deg} deg | out {out_dir}")
        obs = env.reset()
        prev_odom = env.odom_pose()
        last_action = None
        for i in range(args.num_dirs):
            deg = (i * turn_deg) % 360
            # localizer: predict the PREVIOUS turn (skip i==0), then update on this view --
            # exactly as run_robot's get_action, but driven by a pure in-place spin.
            cur_odom = env.odom_pose()
            odom_delta = odom_to_robot_delta(prev_odom, cur_odom)
            loc.predict(last_action, collided=env.last_collided, delta_pose=odom_delta)
            prev_odom = cur_odom
            loc.update(obs, depth_key="depth")

            save_views(out_dir, i, deg, obs, cfg, backend)
            est = loc.position
            rec = {"index": i, "nominal_deg": deg,
                   "est_x": float(est[0]), "est_z": float(est[2]), "est_yaw_deg": float(np.degrees(loc.yaw)),
                   "converged": bool(loc.converged), "spread_m": float(getattr(loc, "spread", float("nan")))}
            manifest["directions"].append(rec)
            print(f"dir {i:02d} ({deg:3d} deg) saved | est=({est[0]:.2f},{est[2]:.2f}) "
                  f"yaw={np.degrees(loc.yaw):.0f} converged={loc.converged} "
                  f"spread={getattr(loc, 'spread', float('nan')):.2f}m")

            if renderer is not None:
                frame = renderer.frame(obs, ACTION.get(turn_act), i)
                if frame is not None:
                    if viz_frames is not None:
                        viz_frames.append(frame)
                    if args.render:
                        if show_frame("run_scan_capture", frame, scale=cfg.window_scale) == 27:
                            break
            last_action = turn_act
            obs = env.step(turn_act)          # rotate 30 deg in place
    except KeyboardInterrupt:
        print("\n>>> stop requested (Ctrl-C)")
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
        cv2.destroyAllWindows()
        if viz_frames:
            from utils.nav_viz import write_video
            write_video(args.save_video, viz_frames, fps=4)
            print(f">>> saved video: {args.save_video} ({len(viz_frames)} frames)")
        manifest["final"] = manifest["directions"][-1] if manifest["directions"] else None
        with open(out_dir / "scan.json", "w") as f:
            json.dump(manifest, f, indent=2)
        print(f">>> scan complete: {len(manifest['directions'])} directions -> {out_dir}")


if __name__ == "__main__":
    main()
