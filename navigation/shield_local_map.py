"""Short-window odometry local map for the POST-convergence collision shield (--safe_veto).

Deliberately SEPARATE from EarlyNav's (full-accumulation, pre-convergence) exploration map
and from the global planning map:

  * odometry-framed  -> stays correct even when the GLOBAL localizer is mis-converged, so it
                        prevents the mis-localization collisions the FMM/scan can't foresee;
  * short window     -> only the last `window` frames, so it barely drifts and never carries
                        stale obstacles, yet still remembers nearby walls that are now OUTSIDE
                        the camera's forward FOV (seen a few steps ago) -- exactly the walls
                        the single-frame forward cone grazes.

Integrates delta-pose from the discrete actions (nominal motion model, matching the localizer;
on the real robot the ROS-topic delta would be fed the same way). Answers whether the swept
forward BODY footprint is blocked.
"""
from collections import deque

import numpy as np


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class ShieldLocalMap:
    def __init__(self, actions, extract_scan, forward_step, turn_angle_deg,
                 agent_radius=0.18, window=12):
        self.A = actions
        self.extract_scan = extract_scan          # obs -> (N,2) robot-frame [x=right, z=back]
        self.fwd = float(forward_step)
        self.turn = np.deg2rad(float(turn_angle_deg))
        self.rad = float(agent_radius)
        self.window = int(window)
        self.reset()

    def reset(self):
        """Wipe odometry + the recent-frame buffer (call on episode start / re-globalize)."""
        self.ox = self.oz = self.oyaw = 0.0
        self._recent = deque(maxlen=self.window)

    # ------------------------------------------------------------------ #
    def update(self, obs, collided, last_action, delta_pose=None):
        """Integrate one step of odometry, then buffer this frame's obstacle points (odo
        frame). ``delta_pose=(d_fwd, d_lat, dyaw)`` uses a measured odometry delta (real
        robot); otherwise the discrete action maps to the nominal motion model (sim)."""
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
        # robot-frame (forward, lateral) -> odo world, matching EarlyNav/localizer convention
        self.ox += d_fwd * np.sin(self.oyaw) + d_lat * np.cos(self.oyaw)
        self.oz += -d_fwd * np.cos(self.oyaw) + d_lat * np.sin(self.oyaw)
        scan = self.extract_scan(obs)
        if scan is not None and len(scan):
            ca, sa = np.cos(self.oyaw), np.sin(self.oyaw)
            wx = scan[:, 0] * ca - scan[:, 1] * sa + self.ox
            wz = scan[:, 0] * sa + scan[:, 1] * ca + self.oz
            self._recent.append(np.stack([wx, wz], axis=1))

    def forward_blocked(self, min_pts=2):
        """Is the swept forward BODY footprint (one step long, body-width) blocked by an
        obstacle point seen in the last `window` frames? min_pts rejects stray-pixel noise."""
        if not self._recent:
            return False
        P = np.concatenate(self._recent, axis=0)          # (M,2) odo-frame obstacle points
        ca, sa = np.cos(self.oyaw), np.sin(self.oyaw)
        dx = P[:, 0] - self.ox; dz = P[:, 1] - self.oz
        rx = dx * ca + dz * sa                             # robot frame: x = right
        rz = -dx * sa + dz * ca                            # forward = -z
        hit = ((rz < self.rad) & (rz > -(self.fwd + self.rad + 0.15))
               & (np.abs(rx) < self.rad + 0.10))          # swept body footprint over one step
        return bool(np.count_nonzero(hit) >= int(min_pts))
