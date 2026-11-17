"""Durable, single-directory recording for one real-robot navigation run."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from queue import Queue
import re
from threading import Thread

from utils.nav_viz import H264VideoWriter


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def resolve_record_dir(record_arg, *, project_root, category, prior, now=None):
    """Resolve an explicit run directory or create the deterministic automatic name."""
    if record_arg != "__auto__":
        return Path(record_arg).expanduser().resolve()
    now = now or datetime.now()
    target = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(category or "no_goal")).strip("_")
    topology = "base" if prior is None else f"prior{int(prior)}"
    name = f"{now.strftime('%Y%m%d-%H%M%S')}_{target}_{topology}"
    return Path(project_root) / "outputs" / "robot_runs" / name


def _vec(value):
    if value is None:
        return None
    return [float(v) for v in value]


def build_step_record(agent, *, step, action_id, action_name, odom_pose, odom_delta,
                      collided, detection_counts, wall_time_s, monotonic_s):
    """Snapshot decision state into JSON-safe values after one get_action call."""
    from utils.panels import navigation_stage

    pos, yaw = agent._pose()
    loc = getattr(agent, "localizer", None)
    odom = _vec(odom_pose)
    delta = _vec(odom_delta)
    record = {
        "step": int(step),
        "wall_time_s": float(wall_time_s),
        "monotonic_s": float(monotonic_s),
        "stage": navigation_stage(agent),
        "mode": str(getattr(agent, "mode", "?")),
        "action": {"id": int(action_id), "name": str(action_name)},
        "estimated_pose": {
            "x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2]),
            "yaw_rad": float(yaw),
        },
        "odom_pose": (None if odom is None else {
            "x": odom[0], "y": odom[1], "yaw_rad": odom[2],
        }),
        "odom_delta": (None if delta is None else {
            "forward_m": delta[0], "lateral_m": delta[1], "yaw_rad": delta[2],
        }),
        "collided": bool(collided),
        "localization": (None if loc is None else {
            "converged": bool(getattr(loc, "converged", False)),
            "settled": bool(getattr(loc, "settled", False)),
            "spread_m": float(getattr(loc, "spread", float("nan"))),
            "steps": int(getattr(agent, "_loc_steps", 0)),
        }),
        "topology": {
            "phase": getattr(agent, "_topo_phase", None),
            "node": getattr(agent, "_cur_node", None),
            "direction": getattr(agent, "_face_dir", None),
        },
        "detection_pixels": {
            str(name): int(count) for name, count in (detection_counts or {}).items()
        },
        "subgoal": _vec(getattr(agent, "subgoal", None)),
        "explore_goal": _vec(getattr(agent, "explore_goal", None)),
        "last_mile_goal": _vec(getattr(agent, "last_mile_goal", None)),
    }
    return record


def snapshot_detection(agent, obs, np_module):
    """Copy the already-cached decision masks for rendering and durable recording.

    ``detection_viz`` reads ``_seg_cache`` populated by the decision path. It never needs a
    second model inference for the same observation because ``_sam3_result`` caches both
    parallel and inline results.
    """
    viz = agent.detection_viz(obs)
    renderer_masks = {}
    saved_masks = {}
    for detector_name, frame_name in (("main", "main"), ("top", "sub")):
        info = viz.get(detector_name)
        if info is None:
            continue
        renderer_masks[frame_name] = info
        masks = [np_module.asarray(mask, dtype=bool) for mask in info.get("masks", [])]
        if masks:
            union = masks[0].copy()
            for mask in masks[1:]:
                union |= mask
            saved_masks[frame_name] = union
    return renderer_masks, saved_masks, agent._per_camera_counts(obs)


class RunRecorder:
    """Write structured logs synchronously and heavy image/video data asynchronously."""

    def __init__(self, output_dir, metadata, cv2_module, np_module, video_fps=4):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.frames_dir = self.output_dir / "frames"
        self.frames_dir.mkdir()
        self.cv2 = cv2_module
        self.np = np_module
        self.video_fps = float(video_fps)
        self.metadata = dict(metadata)
        self.metadata.update({"status": "running", "started_at_utc": _utc_now()})
        self._write_metadata()
        self._steps = (self.output_dir / "steps.jsonl").open("a", encoding="utf-8")
        self._log = (self.output_dir / "run.log").open("a", encoding="utf-8")
        self._queue = Queue()
        self._errors = []
        self._video = None
        self._thread = Thread(target=self._run, name="robot-run-recorder", daemon=True)
        self._thread.start()

    def _write_metadata(self):
        path = self.output_dir / "metadata.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.metadata, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
        os.replace(tmp, path)

    def record_step(self, record, *, frames, masks, render_frame):
        """Persist one step log and enqueue copies of its binary artifacts."""
        payload = dict(record)
        self._steps.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._steps.flush()
        action = payload.get("action") or {}
        pose = payload.get("estimated_pose") or {}
        delta = payload.get("odom_delta")
        delta_text = "none" if delta is None else "({:+.3f},{:+.3f},{:+.3f})".format(
            delta["forward_m"], delta["lateral_m"], delta["yaw_rad"]
        )
        loc = payload.get("localization") or {}
        topo = payload.get("topology") or {}
        self._log.write(
            "step={step} stage={stage} action={action} "
            "est=({x},{z},{yaw}) odom_delta={delta} collided={collided} "
            "converged={converged} settled={settled} spread={spread} "
            "topo=({phase},{node},{direction})\n".format(
                step=payload.get("step"), stage=payload.get("stage"),
                action=action.get("name"), x=pose.get("x"), z=pose.get("z"),
                yaw=pose.get("yaw_rad"), delta=delta_text,
                collided=payload.get("collided"), converged=loc.get("converged"),
                settled=loc.get("settled"), spread=loc.get("spread_m"),
                phase=topo.get("phase"), node=topo.get("node"),
                direction=topo.get("direction"),
            )
        )
        self._log.flush()

        copied_frames = [
            (camera, self.np.asarray(rgb).copy(), self.np.asarray(depth).copy())
            for camera, rgb, depth in frames
        ]
        copied_masks = {
            str(name): self.np.asarray(mask, dtype=bool).copy()
            for name, mask in (masks or {}).items()
        }
        rendered = None if render_frame is None else self.np.asarray(render_frame).copy()
        self._queue.put((int(payload["step"]), copied_frames, copied_masks, rendered))

    def _write_images(self, step, frames, masks):
        for camera, rgb, depth in frames:
            camera_dir = self.frames_dir / camera.name
            camera_dir.mkdir(exist_ok=True)
            stem = camera_dir / f"step_{step:06d}"
            rgb_path = Path(str(stem) + "_rgb.png")
            depth_path = Path(str(stem) + "_depth.png")
            mask_path = Path(str(stem) + "_sam3.png")
            if not self.cv2.imwrite(str(rgb_path), self.cv2.cvtColor(rgb, self.cv2.COLOR_RGB2BGR)):
                raise RuntimeError(f"could not save {rgb_path}")
            if not self.cv2.imwrite(str(depth_path), depth):
                raise RuntimeError(f"could not save {depth_path}")
            mask = masks.get(camera.name)
            if mask is None:
                mask = self.np.zeros(rgb.shape[:2], dtype=self.np.uint8)
            else:
                mask = self.np.asarray(mask, dtype=bool).astype(self.np.uint8) * 255
            if not self.cv2.imwrite(str(mask_path), mask):
                raise RuntimeError(f"could not save {mask_path}")

    def _write_video(self, frame):
        if frame is None:
            return
        if self._video is None:
            self._video = H264VideoWriter(
                self.output_dir / "render.mp4", fps=self.video_fps)
        self._video.append_data(frame)

    def _run(self):
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                step, frames, masks, render_frame = item
                self._write_images(step, frames, masks)
                self._write_video(render_frame)
            except Exception as exc:  # surfaced by finish after draining every queued item
                self._errors.append(exc)
            finally:
                self._queue.task_done()

    def finish(self, *, status, stop_reason, total_steps, error=None, failure_reason=None):
        """Drain all artifacts and atomically finalize metadata."""
        self._queue.put(None)
        self._queue.join()
        self._thread.join()
        if self._video is not None:
            self._video.close()
        self._steps.close()
        self._log.close()
        self.metadata.update({
            "status": str(status),
            "stop_reason": str(stop_reason),
            "success": str(stop_reason) == "stop" and str(status) == "completed",
            "total_steps": int(total_steps),
            "finished_at_utc": _utc_now(),
        })
        if error is not None:
            self.metadata["error"] = str(error)
        if failure_reason is not None:
            self.metadata["failure_reason"] = str(failure_reason)
        if self._errors:
            self.metadata["recording_errors"] = [str(exc) for exc in self._errors]
        self._write_metadata()
        if self._errors:
            raise RuntimeError(f"{len(self._errors)} recording job(s) failed")
