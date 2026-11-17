#!/usr/bin/env python
"""On-robot ROS1 bridge for the Jackal SSH backend.

This file is NOT imported by the client. robot/jackal.py reads it as text and
sends it over the SSH channel to run ON the Jackal (ROS Melodic / Python 2),
where it publishes geometry_msgs/Twist to /cmd_vel and subscribes to odometry +
the dual-D455 image topics. Kept Python-2-compatible and dependency-light
(rospy / cv2 / numpy are imported lazily inside and only exist on the robot).

The client (robot/jackal.py JackalConnection) speaks to it over stdin/stdout
JSON. Commands: move / drive_distance / rotate_angle / stop / status / observe /
shutdown. The smooth odometry-goal control (_start_goal / _goal_desired /
_control_loop) lives here."""
from __future__ import print_function

import base64
from collections import deque
import json
import math
import sys
import threading
import time
import traceback


def monotonic_time():
    fn = getattr(time, "monotonic", None)
    return fn() if fn is not None else time.time()


def message_bytes(data):
    if sys.version_info[0] < 3:
        return str(data)
    return bytes(bytearray(data))


EMIT_LOCK = threading.Lock()


def emit(payload):
    with EMIT_LOCK:
        sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
        sys.stdout.flush()


class Bridge(object):
    def __init__(self, config):
        import cv2
        import numpy as np
        import rospy
        from geometry_msgs.msg import Twist
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import Image

        self.cv2 = cv2
        self.np = np
        self.rospy = rospy
        self.Twist = Twist
        self.lock = threading.RLock()
        self.shutdown_requested = False

        self.cmd_topic = config["cmd_vel_topic"]
        self.odom_topic = config["odom_topic"]
        self.camera_topics = list(config["camera_topics"])
        self.rgb_topics = set(config["rgb_topics"])
        self.depth_topics = set(config["depth_topics"])
        self.camera_pairs = list(config.get("camera_pairs", []))
        self.pair_max_skew = float(config.get("pair_max_skew_seconds", 0.10))
        self.rate_hz = float(config.get("publish_rate", 20.0))
        self.jpeg_quality = int(config.get("jpeg_quality", 90))
        self.linear_speed = float(config.get("linear_speed", 0.18))
        self.angular_speed = float(config.get("angular_speed", 0.45))
        self.forward_distance = float(config.get("forward_distance_m", 0.25))
        self.turn_angle = math.radians(float(config.get("turn_angle_deg", 30.0)))
        self.linear_acceleration = float(config.get("linear_acceleration", 0.25))
        self.angular_acceleration = float(config.get("angular_acceleration", 0.8))
        self.distance_tolerance = float(config.get("distance_tolerance", 0.003))
        self.distance_brake_delay = float(
            config.get("distance_brake_delay_s", 0.10)
        )
        self.angle_brake_delay = float(config.get("angle_brake_delay_s", 0.10))
        self.angle_tolerance = math.radians(
            float(config.get("angle_tolerance_deg", 0.5))
        )
        self.rotation_kp = float(config.get("rotation_kp", 1.5))
        self.min_angular_correction_speed = float(
            config.get("min_angular_correction_speed", 0.08)
        )
        self.rotation_settle_seconds = float(
            config.get("rotation_settle_seconds", 0.25)
        )
        self.goal_timeout = float(config.get("goal_timeout", 20.0))
        # Safety bounds for the general velocity command (move).
        self.max_linear = abs(float(config.get("max_linear", 0.8)))
        self.max_angular = abs(float(config.get("max_angular", 1.5)))
        self.max_duration = float(config.get("max_command_duration", 10.0))
        if self.distance_brake_delay < 0.0 or self.distance_brake_delay > 1.0:
            raise ValueError("distance_brake_delay_s must be between 0 and 1 second")
        if self.angle_brake_delay < 0.0 or self.angle_brake_delay > 1.0:
            raise ValueError("angle_brake_delay_s must be between 0 and 1 second")

        # Keep a short uncompressed history.  RGB and depth callbacks are
        # asynchronous, so a single "latest" slot can combine different sensor
        # instants.  observe selects the closest timestamped pair from this history.
        self.frame_history = dict(
            (topic, deque(maxlen=12)) for topic in self.camera_topics
        )
        self.odom_x = None
        self.odom_y = None
        self.odom_yaw = None
        self.odom_yaw_accumulated = 0.0
        self.odom_linear = 0.0
        self.odom_angular = 0.0
        self.odom_stamp = 0.0
        self.odom_received_at = 0.0
        self.goal = None
        self.last_goal_result = None
        self.current_linear = 0.0
        self.current_angular = 0.0
        self.target = (0.0, 0.0)          # timed velocity command (move)
        self.command_deadline = 0.0
        self.stop_burst = 0
        self.was_moving = False

        rospy.init_node("complex_tamp_jackal_bridge", anonymous=True,
                        disable_signals=True)
        self.publisher = rospy.Publisher(self.cmd_topic, Twist, queue_size=1)
        self.subscribers = [
            rospy.Subscriber(self.odom_topic, Odometry, self._odom_callback,
                             queue_size=1)
        ]
        for topic in self.camera_topics:
            self.subscribers.append(
                rospy.Subscriber(topic, Image, self._image_callback,
                                 callback_args=topic, queue_size=1,
                                 buff_size=2 ** 24)
            )

        self.control_thread = threading.Thread(target=self._control_loop)
        self.control_thread.daemon = True
        self.control_thread.start()
        emit({
            "event": "ready",
            "odom_topic": self.odom_topic,
            "cmd_vel_topic": self.cmd_topic,
            "camera_topics": self.camera_topics,
        })

    @staticmethod
    def _normalize_angle(angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    @staticmethod
    def _approach(current, target, maximum_change):
        if current < target:
            return min(current + maximum_change, target)
        if current > target:
            return max(current - maximum_change, target)
        return target

    def _odom_callback(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        try:
            stamp = msg.header.stamp.to_sec()
        except Exception:
            stamp = 0.0
        try:
            linear = float(msg.twist.twist.linear.x)
        except Exception:
            linear = 0.0
        try:
            angular = float(msg.twist.twist.angular.z)
        except Exception:
            angular = 0.0
        with self.lock:
            if self.odom_yaw is not None:
                self.odom_yaw_accumulated += self._normalize_angle(
                    yaw - self.odom_yaw
                )
            self.odom_x = float(p.x)
            self.odom_y = float(p.y)
            self.odom_yaw = yaw
            self.odom_linear = linear
            self.odom_angular = angular
            self.odom_stamp = stamp
            self.odom_received_at = time.time()

    def _image_callback(self, msg, topic):
        try:
            if topic in self.depth_topics:
                image = self._depth_image(msg)
                transport = "png16"
            else:
                image = self._bgr_image(msg)
                transport = "jpeg"
            try:
                stamp = msg.header.stamp.to_sec()
            except Exception:
                stamp = 0.0
            with self.lock:
                self.frame_history[topic].append({
                    "topic": topic,
                    "stamp": stamp,
                    "received_at": time.time(),
                    "width": int(msg.width),
                    "height": int(msg.height),
                    "source_encoding": str(msg.encoding),
                    "transport": transport,
                    "image": image,
                })
        except Exception as exc:
            emit({"event": "camera_error", "topic": topic,
                  "error": str(exc)})

    def _depth_image(self, msg):
        encoding = str(msg.encoding).lower()
        if encoding not in ("16uc1", "mono16"):
            raise RuntimeError("depth topic {} has unsupported encoding {}".format(
                msg._connection_header.get("topic", "?"), msg.encoding
            ))
        dtype = self.np.dtype(">u2" if bool(msg.is_bigendian) else "<u2")
        raw = self.np.frombuffer(message_bytes(msg.data), dtype=dtype)
        row_values = int(msg.step) // 2
        image = raw.reshape(int(msg.height), row_values)[:, :int(msg.width)]
        return image.astype(self.np.uint16, copy=False)

    def _bgr_image(self, msg):
        encoding = str(msg.encoding).lower()
        channels = {"rgb8": 3, "bgr8": 3,
                    "rgba8": 4, "bgra8": 4, "mono8": 1}.get(encoding)
        if channels is None:
            raise RuntimeError("unsupported RGB encoding {}".format(msg.encoding))
        raw = self.np.frombuffer(message_bytes(msg.data), dtype=self.np.uint8)
        row_bytes = int(msg.width) * channels
        rows = raw.reshape(int(msg.height), int(msg.step))[:, :row_bytes]
        if channels == 1:
            gray = rows.reshape(int(msg.height), int(msg.width))
            return self.cv2.cvtColor(gray, self.cv2.COLOR_GRAY2BGR)
        image = rows.reshape(int(msg.height), int(msg.width), channels)
        if encoding == "rgb8":
            return self.cv2.cvtColor(image, self.cv2.COLOR_RGB2BGR)
        if encoding == "rgba8":
            return self.cv2.cvtColor(image, self.cv2.COLOR_RGBA2BGR)
        if encoding == "bgra8":
            return self.cv2.cvtColor(image, self.cv2.COLOR_BGRA2BGR)
        return image

    @staticmethod
    def _pair_timing(rgb, depth):
        """Return (timestamp skew, newest local arrival time) for a pair."""
        rgb_stamp = float(rgb.get("stamp", 0.0))
        depth_stamp = float(depth.get("stamp", 0.0))
        if rgb_stamp > 0.0 and depth_stamp > 0.0:
            skew = abs(rgb_stamp - depth_stamp)
        else:
            skew = abs(float(rgb["received_at"]) - float(depth["received_at"]))
        newest = max(float(rgb["received_at"]), float(depth["received_at"]))
        return skew, newest

    def _closest_pair(self, rgb_topic, depth_topic):
        rgb_frames = self.frame_history.get(rgb_topic, ())
        depth_frames = self.frame_history.get(depth_topic, ())
        if not rgb_frames or not depth_frames:
            return None
        candidates = [(rgb, depth) for rgb in rgb_frames for depth in depth_frames]
        synchronized = [
            pair for pair in candidates
            if self._pair_timing(pair[0], pair[1])[0] <= self.pair_max_skew
        ]
        if synchronized:
            # Rendering must show the newest synchronized pair, not an older pair
            # that happens to have a fractionally smaller timestamp difference.
            return max(synchronized, key=lambda pair: (
                self._pair_timing(pair[0], pair[1])[1],
                -self._pair_timing(pair[0], pair[1])[0],
            ))
        # Let the client report a meaningful skew error if no valid pair exists.
        return min(candidates, key=lambda pair: self._pair_timing(pair[0], pair[1])[0])

    def _latest_frame(self, topic):
        history = self.frame_history.get(topic)
        return history[-1] if history else None

    def _encode_frame(self, frame):
        image = frame["image"]
        if frame["transport"] == "png16":
            ok, encoded = self.cv2.imencode(".png", image)
        else:
            ok, encoded = self.cv2.imencode(
                ".jpg", image,
                [int(self.cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
            )
        if not ok:
            raise RuntimeError("could not encode {}".format(frame["topic"]))
        try:
            data = encoded.tobytes()
        except AttributeError:
            data = encoded.tostring()
        result = dict(frame)
        result.pop("image", None)
        result["data"] = base64.b64encode(data).decode("ascii")
        result["age"] = max(0.0, time.time() - result["received_at"])
        return result

    def _twist(self, linear, angular):
        msg = self.Twist()
        msg.linear.x = float(linear)
        msg.angular.z = float(angular)
        return msg

    def _arm_stop_burst(self):
        self.stop_burst = max(self.stop_burst, int(0.4 * self.rate_hz))

    def _finish_goal(self, result, reason=None):
        if self.goal is None:
            return
        completed = {
            "kind": self.goal["kind"],
            "target": self.goal["target"] * self.goal["direction"],
            "progress": self.goal.get("progress", 0.0) * self.goal["direction"],
            "result": result,
        }
        if reason is not None:
            completed["reason"] = reason
        self.last_goal_result = completed
        self.goal = None

    def _goal_desired(self, now):
        goal = self.goal
        if goal is None:
            return None
        if self.odom_x is None or time.time() - self.odom_received_at > 1.0:
            self._finish_goal("aborted", "odometry became stale")
            return (0.0, 0.0)
        if now > goal["deadline"]:
            self._finish_goal("aborted", "goal timeout")
            return (0.0, 0.0)

        if goal["kind"] == "distance":
            dx = self.odom_x - goal["start_x"]
            dy = self.odom_y - goal["start_y"]
            projection = (dx * math.cos(goal["start_yaw"]) +
                          dy * math.sin(goal["start_yaw"]))
            progress = goal["direction"] * projection
            remaining = goal["target"] - progress
            goal["progress"] = progress
            goal["remaining"] = remaining
            # End cruise based on the measured velocity, not pose alone.  The
            # base will keep moving for `delay` seconds after a new cmd_vel is
            # requested, so reserve v*delay of the remaining path before the
            # normal v^2/(2a) braking curve.  Both terms are refreshed from
            # odometry every control cycle; speed and acceleration are unchanged.
            progress_velocity = max(0.0, goal["direction"] * self.odom_linear)
            delayed_remaining = max(
                0.0, remaining - progress_velocity * self.distance_brake_delay
            )
            braking = math.sqrt(max(
                0.0, 2.0 * goal["acceleration"] *
                max(0.0, delayed_remaining - self.distance_tolerance)
            ))
            linear = goal["direction"] * min(goal["speed"], braking)
            yaw_error = self._normalize_angle(self.odom_yaw - goal["start_yaw"])
            angular = max(-0.25, min(0.25, -1.5 * yaw_error))
            if remaining <= self.distance_tolerance:
                return (0.0, 0.0)
            return (linear, angular)

        progress = goal["direction"] * (
            self.odom_yaw_accumulated - goal["start_yaw_accumulated"]
        )
        remaining = goal["target"] - progress
        goal["progress"] = progress
        goal["remaining"] = remaining
        magnitude = abs(remaining)
        if magnitude <= self.angle_tolerance:
            return (0.0, 0.0)
        # As in distance control, reserve the angle that the base will still
        # traverse while the next cmd_vel update reaches the drivetrain. This
        # uses the live odometry angular velocity to shorten cruise only.
        progress_velocity = max(0.0, goal["direction"] * self.odom_angular)
        delayed_magnitude = max(
            0.0, magnitude - progress_velocity * self.angle_brake_delay
        )
        # Same shape as distance control: min(cruise, sqrt(2 a (remaining - tol))). No
        # proportional (kp) term -- that made the command = kp*remaining, which BINDS below
        # the sqrt curve over a whole 30 deg turn, so the turn never cruised and instead
        # decayed linearly from the start (fast start, long slow crawl). Dropping it lets the
        # turn ramp -> cruise (angular_speed) -> sqrt-brake, a symmetric trapezoid like the
        # forward step. (rotation_kp is now unused by the controller.)
        braking = math.sqrt(2.0 * goal["acceleration"]
                            * max(0.0, delayed_magnitude - self.angle_tolerance))
        # Below roughly 0.05 rad/s the real Jackal can remain stationary even though cmd_vel
        # is non-zero, so floor the command at min_angular_correction_speed (a physical
        # rotational deadband that the linear drive does not need) until it reaches tolerance.
        angular = math.copysign(
            min(goal["speed"], max(self.min_angular_correction_speed, braking)),
            goal["direction"] * remaining,
        )
        return (0.0, angular)

    def _control_loop(self):
        rate = self.rospy.Rate(self.rate_hz)
        previous = monotonic_time()
        zero = self._twist(0.0, 0.0)
        while not self.rospy.is_shutdown() and not self.shutdown_requested:
            message = None
            with self.lock:
                now = monotonic_time()
                dt = max(0.0, min(now - previous, 2.0 / self.rate_hz))
                previous = now
                goal_desired = self._goal_desired(now)
                if goal_desired is not None:
                    desired = goal_desired
                elif now < self.command_deadline:
                    desired = self.target        # timed velocity command (move)
                else:
                    desired = (0.0, 0.0)
                self.current_linear = self._approach(
                    self.current_linear, desired[0], self.linear_acceleration * dt
                )
                self.current_angular = self._approach(
                    self.current_angular, desired[1], self.angular_acceleration * dt
                )
                moving = (abs(self.current_linear) > 1e-4 or
                          abs(self.current_angular) > 1e-4)

                if self.goal is not None:
                    if self.goal["kind"] == "rotation":
                        at_goal = abs(self.goal.get("remaining", 1e9)) <= self.angle_tolerance
                        settle_seconds = self.rotation_settle_seconds
                    else:
                        at_goal = self.goal.get("remaining", 1e9) <= self.distance_tolerance
                        settle_seconds = 0.0
                    if at_goal and not moving:
                        since = self.goal.get("settled_since")
                        if since is None:
                            self.goal["settled_since"] = now
                        elif now - since >= settle_seconds:
                            self._finish_goal("succeeded")
                    else:
                        self.goal["settled_since"] = None

                active = moving or self.goal is not None or now < self.command_deadline
                if active:
                    message = self._twist(self.current_linear, self.current_angular)
                    self.was_moving = True
                else:
                    if self.was_moving:
                        self._arm_stop_burst()
                        self.was_moving = False
                    if self.stop_burst > 0:
                        message = zero
                        self.stop_burst -= 1
            if message is not None:
                self.publisher.publish(message)
            try:
                rate.sleep()
            except Exception:
                break

    def _start_goal(self, kind, value, speed=None, acceleration=None, timeout=None):
        """Smooth odometry goal for an ARBITRARY distance (m) / angle (rad). `value`
        sign gives the direction (+ forward / left, - back / right)."""
        if self.odom_x is None or time.time() - self.odom_received_at > 1.0:
            raise RuntimeError("odometry is not ready or is stale")
        if self.goal is not None:
            raise RuntimeError("another odometry goal is already active")
        if abs(float(value)) < 1e-9:
            raise RuntimeError("goal value must be non-zero")
        default_speed = self.linear_speed if kind == "distance" else self.angular_speed
        default_accel = (self.linear_acceleration if kind == "distance"
                         else self.angular_acceleration)
        speed = default_speed if speed is None else abs(float(speed))
        acceleration = default_accel if acceleration is None else abs(float(acceleration))
        if speed <= 0.0 or acceleration <= 0.0:
            raise RuntimeError("goal speed and acceleration must be positive")
        target = abs(float(value))
        direction = 1.0 if float(value) >= 0.0 else -1.0
        timeout = self.goal_timeout if timeout is None else float(timeout)
        # A pending timed velocity command must not fight this goal.
        self.command_deadline = 0.0
        self.target = (0.0, 0.0)
        self.last_goal_result = None
        self.goal = {
            "kind": kind,
            "target": target,
            "direction": direction,
            "speed": speed,
            "acceleration": acceleration,
            "progress": 0.0,
            "remaining": target,
            "deadline": monotonic_time() + timeout,
            "start_x": self.odom_x,
            "start_y": self.odom_y,
            "start_yaw": self.odom_yaw,
            "start_yaw_accumulated": self.odom_yaw_accumulated,
            "settled_since": None,
        }
        return target * direction

    def _goal_status(self):
        if self.goal is None:
            return None
        return {
            "kind": self.goal["kind"],
            "target": self.goal["target"] * self.goal["direction"],
            "progress": self.goal.get("progress", 0.0) * self.goal["direction"],
            "remaining": self.goal.get("remaining", self.goal["target"]),
        }

    def handle(self, request):
        response = {"id": request.get("id"), "ok": True}
        command = request.get("command")
        try:
            if command == "move":
                # Arbitrary bounded velocity for `duration` s, then auto-zero.
                linear = float(request.get("linear", 0.0))
                angular = float(request.get("angular", 0.0))
                duration = float(request.get("duration", 0.25))
                if not 0.0 < duration <= self.max_duration:
                    raise ValueError("duration must be > 0 and <= {:.3f}s".format(
                        self.max_duration))
                if abs(linear) > self.max_linear or abs(angular) > self.max_angular:
                    raise ValueError(
                        "velocity exceeds limits (|linear|<={}, |angular|<={})".format(
                            self.max_linear, self.max_angular))
                with self.lock:
                    if self.goal is not None:
                        self._finish_goal("cancelled", "overridden by velocity command")
                    self.target = (linear, angular)
                    self.command_deadline = monotonic_time() + duration
                response.update({"linear": linear, "angular": angular,
                                 "duration": duration})
            elif command == "drive_distance":
                # Arbitrary forward/back distance (m); sign = direction.
                with self.lock:
                    response["target"] = self._start_goal(
                        "distance", request.get("distance", 0.0),
                        request.get("speed"), request.get("acceleration"),
                        request.get("timeout"))
            elif command == "rotate_angle":
                # Arbitrary rotation (rad); + = left, - = right.
                with self.lock:
                    response["target"] = self._start_goal(
                        "rotation", request.get("angle", 0.0),
                        request.get("speed"), request.get("acceleration"),
                        request.get("timeout"))
            elif command == "stop":
                with self.lock:
                    self.command_deadline = 0.0
                    self.target = (0.0, 0.0)
                    if self.goal is not None:
                        self._finish_goal("cancelled", "stop requested")
                    self.current_linear = 0.0
                    self.current_angular = 0.0
                    self._arm_stop_burst()
                response["stopped"] = True
            elif command == "status":
                with self.lock:
                    odom = None
                    if self.odom_x is not None:
                        odom = {
                            "x": self.odom_x,
                            "y": self.odom_y,
                            "yaw": self.odom_yaw,
                            "stamp": self.odom_stamp,
                            "received_at": self.odom_received_at,
                            "topic": self.odom_topic,
                        }
                    response.update({
                        "odom": odom,
                        "goal": self._goal_status(),
                        "last_goal_result": self.last_goal_result,
                        "current_velocity": {
                            "linear": self.current_linear,
                            "angular": self.current_angular,
                        },
                        "cmd_vel_connections": self.publisher.get_num_connections(),
                    })
            elif command == "observe":
                topics = request.get("topics") or self.camera_topics
                with self.lock:
                    snapshots = dict(
                        (topic, self._latest_frame(topic))
                        for topic in topics
                    )
                    # Replace the independent latest frames with the closest
                    # RGB-depth pair available for every configured camera.
                    for pair in self.camera_pairs:
                        rgb_topic = pair["rgb"]
                        depth_topic = pair["depth"]
                        if rgb_topic not in snapshots or depth_topic not in snapshots:
                            continue
                        matched = self._closest_pair(rgb_topic, depth_topic)
                        if matched is not None:
                            snapshots[rgb_topic], snapshots[depth_topic] = matched
                frames = []
                missing = []
                for topic in topics:
                    frame = snapshots.get(topic)
                    if frame is None:
                        missing.append(topic)
                        continue
                    frames.append(self._encode_frame(frame))
                response.update({"frames": frames, "missing": missing})
            elif command == "shutdown":
                with self.lock:
                    self.goal = None
                    self.command_deadline = 0.0
                    self.target = (0.0, 0.0)
                    self.current_linear = 0.0
                    self.current_angular = 0.0
                zero = self._twist(0.0, 0.0)
                for unused in range(4):
                    self.publisher.publish(zero)
                    time.sleep(0.03)
                self.shutdown_requested = True
                response["shutdown"] = True
            else:
                raise ValueError("unsupported command {!r}".format(command))
        except Exception as exc:
            response.update({"ok": False, "error": str(exc)})
        emit(response)


def main():
    try:
        config = json.loads(base64.b64decode(sys.argv[1]).decode("utf-8"))
        bridge = Bridge(config)
        while True:
            line = sys.stdin.readline()
            if not line:
                break
            bridge.handle(json.loads(line))
            if bridge.shutdown_requested:
                break
    except Exception as exc:
        emit({"event": "fatal", "error": str(exc),
              "traceback": traceback.format_exc()})
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
