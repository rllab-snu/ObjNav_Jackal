"""Navigation visualization helpers (adapted from complex/objnav/visualization).

Compose a debug frame: [RGB | semantic | top-down]. The top-down is built from a
(global) occupancy map and shows:
  - occupancy: obstacle (dark) / free (light) / unknown (grey)
  - the FMM geodesic distance field (heatmap over free space), and/or the
    medial-axis skeleton of the free space
  - the current goal (green), selected STG (cyan), and the agent (white dot) +
    its camera FOV fan
"""
import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

import skimage.morphology

from navigation.geometry import heading_from_rotvec  # noqa: F401 (re-exported)


class H264VideoWriter:
    """Streaming RGB writer for broadly compatible H.264/AVC MP4 files.

    The explicit FFMPEG backend prevents ImageIO from dispatching ``.mp4`` to an
    unrelated plugin.  There is deliberately no ``mp4v`` fallback: silently
    producing a file that some players cannot decode is worse than reporting the
    missing ``imageio-ffmpeg`` dependency immediately.
    """

    def __init__(self, path, fps=4):
        try:
            import imageio.v2 as imageio
            import imageio_ffmpeg  # noqa: F401  (validates bundled FFmpeg availability)
        except Exception as exc:
            raise RuntimeError(
                "H.264 video recording requires imageio and imageio-ffmpeg"
            ) from exc
        self.path = str(path)
        self._frame_size = None
        self._size = None
        self._writer = imageio.get_writer(
            self.path,
            format="FFMPEG",
            fps=float(fps),
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=None,
            ffmpeg_log_level="error",
            output_params=[
                "-preset", "medium",
                "-crf", "20",
                "-profile:v", "high",
                "-tag:v", "avc1",
                "-movflags", "+faststart",
            ],
        )

    @staticmethod
    def _rgb8(frame):
        frame = np.ascontiguousarray(np.asarray(frame)[..., :3]).astype(np.uint8)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"video frame must be HxWx3 RGB, got {frame.shape}")
        return frame

    def append_data(self, frame):
        frame = self._rgb8(frame)
        h, w = frame.shape[:2]
        if self._size is None:
            # yuv420p requires even dimensions. Preserve every input pixel by
            # padding at most one row/column instead of macroblock-resizing it.
            self._frame_size = (w, h)
            self._size = (w + (w % 2), h + (h % 2))
        out_w, out_h = self._size
        src_w, src_h = self._frame_size
        if frame.shape[:2] != (src_h, src_w):
            if cv2 is None:
                raise RuntimeError("OpenCV is required to resize variable-sized video frames")
            frame = cv2.resize(frame, (src_w, src_h), interpolation=cv2.INTER_AREA)
        if (src_h, src_w) != (out_h, out_w):
            frame = np.pad(
                frame,
                ((0, out_h - src_h), (0, out_w - src_w), (0, 0)),
                mode="edge",
            )
        self._writer.append_data(np.ascontiguousarray(frame))

    def close(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None


def write_video(path, frames, fps=4):
    """Write RGB frames as H.264/AVC High, yuv420p, fast-start MP4."""
    if not frames:
        return False
    writer = H264VideoWriter(path, fps=fps)
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()
    return True


def colorize_semantic(sem):
    """(H,W[,1]) semantic-id image -> stable RGB (background id 0 -> black)."""
    sem = np.asarray(sem).squeeze().astype(np.int64)
    ids = sem & 0xFFFFFF
    out = np.stack([(ids * 131 % 256), (ids * 241 % 256), (ids * 67 % 256)], -1).astype(np.uint8)
    out[sem <= 0] = 0
    return out


def _label(img, text, band=22):
    """Prepend a label STRIP above the image so the label never covers panel content
    (the returned image is `band` px taller). A blank strip is added when text is empty,
    so every panel in a row grows by the same amount and heights stay uniform."""
    img = np.ascontiguousarray(img)
    if cv2 is None:
        return img
    bar = np.zeros((band, img.shape[1], 3), np.uint8)
    if text:
        cv2.putText(bar, str(text), (5, band - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return np.ascontiguousarray(np.concatenate([bar, img], axis=0))


def compose(panels, labels=None, text_lines=None, gap=8, gap_color=(20, 20, 20)):
    """Stack labeled panels side by side (resized to a common height) + text bar.

    `gap`: width (px) of the separator strip inserted between panels so adjacent
    ones are easy to tell apart (0 disables it)."""
    labels = labels or [None] * len(panels)
    h = min(p.shape[0] for p in panels)
    out = []
    for p, lb in zip(panels, labels):
        p = np.ascontiguousarray(np.asarray(p)[..., :3]).astype(np.uint8)
        if p.shape[0] != h:
            p = cv2.resize(p, (int(p.shape[1] * h / p.shape[0]), h))
        out.append(_label(p, lb))
    if gap > 0 and len(out) > 1:
        sep = np.full((out[0].shape[0], gap, 3), gap_color, np.uint8)   # match label-inclusive height
        pieces = []
        for i, pan in enumerate(out):
            if i:
                pieces.append(sep)
            pieces.append(pan)
        out = pieces
    row = np.concatenate(out, axis=1)
    if text_lines and cv2 is not None:
        bar = np.zeros((24 * len(text_lines) + 8, row.shape[1], 3), np.uint8)
        for i, line in enumerate(text_lines):
            cv2.putText(bar, line, (8, 18 + 24 * i), cv2.FONT_HERSHEY_DUPLEX,
                        0.55, (240, 240, 240), 1, cv2.LINE_AA)
        row = np.concatenate([bar, row], axis=0)
    return row


def render_topdown(gmap, agent_pos, heading, goal=None, fmm_dist=None,
                   show_fmm=True, show_skeleton=True, hfov_deg=None,
                   fan_range_m=4.0, out_height=480, goal_color=(0, 220, 0),
                   gt_position=None, stg_position=None,
                   stg_color=(0, 255, 255), goal_sources=None,
                   goal_source_color=(255, 210, 0)):
    """Whole-map top-down of a global occupancy map with goal / agent fan / FMM.

    gmap: object with .H .W .res, .point_to_grid([x,y,z]); `navigable` and/or
          `free` boolean grids (row=+z, col=+x).
    heading: heading_from_rotvec (atan2(fwd_x, -fwd_z)).
    fmm_dist: optional (H,W) geodesic field (cells) to overlay as a heatmap.
    hfov_deg: sensor horizontal FOV (deg) for the agent's view fan; pass the value
              from config so the fan follows the camera. None -> no fan is drawn.
    goal_color: RGB for the current goal marker (e.g. green=explore, red=last-mile).
    gt_position: optional world pos of the true object, drawn as a blue X.
    stg_position: optional world position of the FMM short-term goal. A cyan
                  marker and agent-to-STG line show the immediate greedy target.
    goal_sources: optional boolean grid of multi-source FMM terminal cells.
    """
    if cv2 is None:
        return np.zeros((out_height, out_height, 3), np.uint8)

    H, W, res = gmap.H, gmap.W, gmap.res
    free = getattr(gmap, "free", None)
    nav = getattr(gmap, "navigable", free)

    img = np.full((H, W, 3), 90, np.uint8)            # unknown / non-navigable
    if nav is not None:
        img[nav] = (205, 205, 205)                    # navigable / free (light)
        img[~nav] = (45, 50, 65)                      # obstacle (dark)

    # Dynamic-map (online) NEW obstacles: cells the base map called free but the
    # live scans have marked occupied -> orange, so map updates are visible.
    dyn = getattr(gmap, "dyn_occ", None)
    if dyn is not None and np.any(dyn):
        img[dyn] = (255, 140, 0)

    # FMM geodesic field as a heatmap over reachable free space.
    if show_fmm and fmm_dist is not None and free is not None:
        fd = np.asarray(fmm_dist, dtype=float)
        valid = free & np.isfinite(fd) & (fd < fd.max())
        if valid.any():
            vmax = max(1e-6, float(np.percentile(fd[valid], 99)))
            norm = np.clip(fd / vmax, 0, 1)
            heat = cv2.cvtColor(
                cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET),
                cv2.COLOR_BGR2RGB)
            img[valid] = (0.45 * img[valid] + 0.55 * heat[valid]).astype(np.uint8)

    # Medial-axis skeleton of the free space (the "corridor centerlines").
    # Use free & seen so the skeleton doesn't sprout into never-mapped exterior.
    if show_skeleton and free is not None:
        sk = skimage.morphology.skeletonize(getattr(gmap, "free_seen", free))
        img[sk] = (255, 90, 90)

    # GT object position: blue X.
    if gt_position is not None:
        rc = gmap.point_to_grid(gt_position)
        ty, tx = int(round(rc[0])), int(round(rc[1]))
        if 0 <= ty < H and 0 <= tx < W:
            s = max(4, H // 55)
            cv2.line(img, (tx - s, ty - s), (tx + s, ty + s), (40, 110, 255), 2)
            cv2.line(img, (tx - s, ty + s), (tx + s, ty - s), (40, 110, 255), 2)

    # Agent: FOV fan (translucent) + white dot + heading tick. The fan is only
    # drawn when the caller supplies the sensor HFOV (from config); with no value
    # we skip it rather than bake in a wrong angle if the camera setup changes.
    rc = gmap.point_to_grid(agent_pos)
    ay, ax = int(round(rc[0])), int(round(rc[1]))
    if hfov_deg is not None:
        ang = np.degrees(np.arctan2(-np.cos(heading), np.sin(heading)))  # image-frame
        rad = max(6, int(round(fan_range_m / res)))
        fan = img.copy()
        cv2.ellipse(fan, (ax, ay), (rad, rad), 0,
                    ang - hfov_deg / 2, ang + hfov_deg / 2, (90, 170, 255), -1)
        img = cv2.addWeighted(fan, 0.30, img, 0.70, 0)

    # Multi-source terminal region. Draw after the translucent FOV fan so source
    # cells keep an unambiguous colour in both live and recorded renderings.
    if goal_sources is not None:
        sources = np.asarray(goal_sources, dtype=bool)
        if sources.shape == (H, W):
            img[sources] = goal_source_color

    # Redraw the current goal after the source region/FOV. In last-mile this is
    # the raw detected object position, not a navigable proxy or an FMM source.
    if goal is not None:
        rc = gmap.point_to_grid(goal)
        gy, gx = int(round(rc[0])), int(round(rc[1]))
        if 0 <= gy < H and 0 <= gx < W:
            cv2.circle(img, (gx, gy), max(3, H // 70), goal_color, -1)

    # The controller's immediate greedy target. Draw this after the translucent
    # FOV fan so its configured RGB remains unambiguous in recorded videos.
    if stg_position is not None:
        rc = gmap.point_to_grid(stg_position)
        sy, sx = int(round(rc[0])), int(round(rc[1]))
        if 0 <= sy < H and 0 <= sx < W:
            cv2.line(img, (ax, ay), (sx, sy), stg_color,
                     max(1, H // 150), cv2.LINE_AA)
            cv2.circle(img, (sx, sy), max(3, H // 70), stg_color, -1)
    cv2.circle(img, (ax, ay), max(3, H // 70), (255, 255, 255), -1)

    scale = out_height / H
    return cv2.resize(img, (max(1, int(W * scale)), out_height),
                      interpolation=cv2.INTER_NEAREST)
