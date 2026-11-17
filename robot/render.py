"""Live per-frame visualization for run_robot.

Composes the SAME shared panels as the habitat-sim nav-viz (utils/panels.py), grouped by
theme -- OBS+DETECTION | LOCALIZATION | NAVIGATION -- so the sim and the robot render
IDENTICALLY (same functions, same layout grouping). The robot has no semantic sensor and no
ground-truth pose, so those panels/overlays are simply omitted (want_semantic=False, gt=None).

Purely observational: it only READS agent state and never affects decisions. run_robot shows
the returned frame in a live cv2 window (`--render`) and/or writes it to an mp4 (--save_video).
"""
import numpy as np

try:
    import cv2
except Exception:                        # pragma: no cover
    cv2 = None

import utils.panels as P

_shown_windows = set()


def show_frame(name, frame_rgb, scale=0.6):
    """Show an RGB frame in a RESIZABLE cv2 window opened at `scale`x the frame size, so the
    live window is smaller than the full composed frame (still freely draggable to any size).
    Returns the waitKey code (& 0xFF) so the caller can handle Esc. Display only -- the frame
    and any saved video keep the full resolution."""
    if cv2 is None or frame_rgb is None:
        return -1
    if name not in _shown_windows:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        h, w = frame_rgb.shape[:2]
        cv2.resizeWindow(name, max(1, int(w * scale)), max(1, int(h * scale)))
        _shown_windows.add(name)
    cv2.imshow(name, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    return cv2.waitKey(1) & 0xFF


class RobotRenderer:
    """Compose one live frame from (obs, agent). Read-only."""

    def __init__(self, agent, hfov_deg=None, row_height=300,
                 groups=("obs", "localization", "navigation")):
        self.agent = agent
        self.hfov = hfov_deg
        self.row_height = int(row_height)
        self.groups = tuple(groups)          # which theme groups to render (task-specific)

    def frame(self, obs, action=None, step=None, sam3_masks=None):
        a = self.agent
        if cv2 is None:
            return None
        h = self.row_height
        pos, yaw = a._pose()                         # localizer estimate (robot has no GT)
        pos = np.asarray(pos, float)
        est = (float(pos[0]), float(pos[2]), yaw)

        rows = []
        # --- obs + detection: rgb | depth | sam3 (main, then sub). SAM 3 per-instance masks,
        #     coloured selected-goal green / other candidates red (robot: no semantic). ---
        if "obs" in self.groups:
            masks = dict(sam3_masks or {})
            if sam3_masks is None and getattr(a, "sam3", None) is not None:
                viz = a.detection_viz(obs)           # {cam_name: {"masks", "selected"}}
                for cam_name, row_key in (("main", "main"), ("top", "sub")):
                    if cam_name in viz and row_key not in masks:
                        masks[row_key] = viz[cam_name]
            rows += P.obs_detection_rows(obs, h, sam3_masks=masks, want_semantic=False,
                                         prompt=getattr(a, "goal_category", None))
        # --- localization: ego-scan | local occ map | likelihood | particles (gt=None) ---
        if "localization" in self.groups:
            rows.append(P.localization_row(a, obs, h, est=est, gt=None, hfov_deg=self.hfov))
        # --- navigation: top-down FMM | dynamic map (dropped for in-place tasks like scan) ---
        if "navigation" in self.groups:
            rows.append(P.navigation_row(a, h, pos, yaw, gt_position=None, hfov_deg=self.hfov))

        # shared status text (robot has no GT pose -> gt=None; no sim metrics)
        text = P.status_lines(a, step=step, action=action, est=est, gt=None)
        return P.vstack(rows, text)
