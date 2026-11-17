"""Robot environment wrapper: Jackal sensors in, discrete actions out (habitat-free).

Produces observations in **habitat format** so the inherited agent code (RobotAgent)
consumes them unchanged: rgb uint8 and depth NORMALIZED to [0,1] (metric depth /
max_depth), keyed rgb/depth (main) and rgb_top/depth_top (sub). The three discrete
actions map onto the Jackal's odometry goals (25 cm forward, 30 deg left/right); a
forward that odometry shows did not move sets the collision flag the localizer's
motion model reads.

Action space (matches the FMM controller / localizer):
    0 stop | 1 move_forward | 2 turn_left | 3 turn_right
"""
import numpy as np


class RobotEnv:
    def __init__(self, cfg, backend):
        self.cfg = cfg
        self.backend = backend          # JackalConnection: SensorSource + CommandSink
        self.last_collided = False

    # ------------------------------------------------------------------ #
    def odom_pose(self):
        """The robot's pose in the FIXED odom frame from ROS /odometry/filtered:
        (x, y, yaw) in metres/radians. This is NOT a per-step delta and NOT
        robot-relative -- it is an absolute (drifting) pose in a frame fixed at odom
        start. Consumers difference consecutive poses into the robot-frame motion the
        localizer wants (see run_robot.odom_to_robot_delta).

        Odometry is mandatory for real-robot motion estimation. JackalConnection.get_odom
        obtains a fresh bridge status and already raises for missing or stale ROS odometry;
        propagate that error instead of silently falling back to nominal action motion.
        """
        raw = self.backend.get_odom()
        if raw is None:
            raise RuntimeError("robot backend returned no odometry")
        try:
            pose = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("robot backend returned invalid odometry") from exc
        if pose.shape != (3,) or not np.all(np.isfinite(pose)):
            raise RuntimeError(
                "robot backend returned invalid odometry: expected three finite values"
            )
        return tuple(float(value) for value in pose)

    def reset(self):
        self.backend.wait_ready()
        self.last_collided = False
        return self._frames()

    def get_obs(self):
        return self._frames()

    def _frames(self):
        """rgb + NORMALIZED depth for every camera (habitat obs format). One
        refresh_frames() call retrieves all RGB-D topics atomically enough for one
        observation and caches them for the get_rgb/get_depth calls below."""
        obs = {}
        self.backend.refresh_frames()
        for cam in self.cfg.cameras:
            suffix = "" if cam.name == "main" else "_top"
            rgb = np.asarray(self.backend.get_rgb(cam.name))[..., :3].astype(np.uint8)
            depth_m = np.asarray(self.backend.get_depth(cam.name), np.float32) * cam.depth_scale
            # habitat format: metric depth / max_depth -> [0,1]. After clipping, anything
            # ABOVE 0.99 (metric > 0.99*max_depth ~ the near-ceiling band, INCLUDING the
            # clip's 1.0 from real far returns and the D455's 0xFFFF saturation, seen up to
            # ~65 m in raw captures) is INVALIDATED to 0 -- otherwise a finite ~max_depth
            # reading could be mistaken for a real max-range obstacle. 0 is the invalid
            # sentinel the localizer (z>0.15) and detector (depth>1e-4) already drop, so
            # far/saturated pixels never enter a scan. (0.99*max_depth = 9.9 m here also
            # matches the localizer's own max_depth-0.1 far cutoff.)
            depth_norm = np.clip(depth_m / cam.max_depth_m, 0.0, 1.0)
            depth_norm[depth_norm > 0.99] = 0.0
            obs["rgb" + suffix] = rgb
            obs["depth" + suffix] = depth_norm
        return obs

    # ------------------------------------------------------------------ #
    def step(self, action):
        """Execute one discrete action as a Jackal odometry goal, then observe. Sets
        last_collided when a commanded forward produced (almost) no displacement."""
        step_m = self.cfg.motion.forward_step_m
        self.last_collided = False
        if action == 1:
            result = self.backend.forward_25cm()
            # forward_25cm returns the goal result dict; its odometry `progress`
            # (metres) flags a collision. Non-dict -> assume it ran to completion.
            moved = float(result.get("progress", step_m)) if isinstance(result, dict) else step_m
            self.last_collided = moved < 0.5 * step_m
        elif action == 2:
            self.backend.turn_left_30deg()
        elif action == 3:
            self.backend.turn_right_30deg()
        # action 0 (stop) -> no motion
        self.backend.stop()
        return self.get_obs()
