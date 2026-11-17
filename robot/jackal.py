"""The Jackal SSH backend: connection + low-level motion control.

``JackalConnectionConfig`` holds the connection + low-level control constants: SSH,
odom / cmd_vel topics, the fixed 0.25 m / 30 deg steps, and the velocity /
acceleration / tolerance constants. It does NOT hold the
cameras -- those (topics, intrinsics, resolution, depth scale, mounting) live in
``configs/jackal.yaml``; ``JackalConnection`` reads the camera topics/resolution off
the ``RobotConfig`` it is given, so nothing is duplicated.

``JackalConnection`` is the combined SensorSource/CommandSink over one SSH channel.
The on-robot bridge (``robot/jackal_bridge.py``) is read as text and sent over the
channel at runtime; it subscribes to odometry + the dual-D455 topics and exposes the
general motion commands: ``drive_distance``
(arbitrary metres), ``rotate_angle`` (arbitrary radians) -- both smooth odometry
goals -- and ``move`` (bounded velocity for a duration). ``forward_25cm`` /
``turn_left_30deg`` / ``turn_right_30deg`` are thin fixed-value wrappers over those
(the discrete action space RobotEnv drives).
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shlex
import threading
import time
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class JackalConnectionConfig:
    """Connection + low-level control constants for the Daejeon Jackal.

    Camera topics / intrinsics / resolution / depth-scale / mounting are NOT here --
    they live in configs/jackal.yaml (the single source), and JackalConnection reads
    the topics + resolution off the RobotConfig it is given. (For reference, the
    dual-D455 serials by topic: /camera_1 = 338122303922, /camera_2 = 333422300812.)
    """

    host: str = "jackalw"
    ssh_user: str = "jackal"
    # Fill in the Jackal SSH password before use, or leave empty to use key-based auth
    # (~/.ssh/config IdentityFile / ssh-agent). Never commit a real password.
    ssh_password: Optional[str] = ""
    ssh_port: int = 22
    connect_timeout: float = 10.0
    request_timeout: float = 10.0

    odom_topic: str = "/odometry/filtered"
    cmd_vel_topic: str = "/cmd_vel"
    # Upstream D455 FPS.  Both cameras are presently on the Jackal's USB2 hub;
    # 5 Hz keeps two 640x480 RGB8+Z16 streams within its usable bandwidth.
    camera_fps: float = 5.0
    jpeg_quality: int = 90
    # The bridge selects the closest RGB/depth pair from a short timestamp
    # history.  Keep this strict enough to reject a genuinely stale pairing.
    max_rgb_depth_skew_seconds: float = 0.10

    forward_distance_m: float = 0.25
    turn_angle_deg: float = 30.0
    linear_speed: float = 0.18
    angular_speed: float = 0.45
    linear_acceleration: float = 0.25
    angular_acceleration: float = 0.8
    distance_tolerance_m: float = 0.003
    # cmd_vel / odometry feedback delay used to end the cruise phase early.
    # This changes neither cruise speed nor acceleration.
    distance_brake_delay_s: float = 0.10
    # Angular equivalent of distance_brake_delay_s for rotation goals.
    angle_brake_delay_s: float = 0.10
    # 0.1 degree made the controller issue sub-deadband Twist commands near
    # the target and wait forever for motion that the physical base could not
    # produce.  Half a degree is still much smaller than one 30 degree action.
    angle_tolerance_deg: float = 0.5
    rotation_kp: float = 1.5
    min_angular_correction_speed: float = 0.08
    rotation_settle_seconds: float = 0.25
    goal_timeout: float = 20.0
    # Safety bounds for the general velocity command (move); the fixed 25 cm / 30 deg
    # goals ignore these and use the smooth odometry controller.
    max_linear: float = 0.8
    max_angular: float = 1.5
    max_command_duration: float = 10.0

    ros_setup: str = "/opt/ros/melodic/setup.bash"
    workspace_setup: str = "/home/jackal/catkin_ws/devel/setup.bash"
    camera_launch_package: str = "jackal_dual_d455"
    camera_launch_file: str = "dual_d455.launch"
    camera_node: str = "/dual_d455"
    camera_log: str = "/tmp/complex_tamp_dual_d455.log"
    auto_start_cameras: bool = True


class JackalConnectionError(RuntimeError):
    pass


class JackalCommandError(JackalConnectionError):
    pass


# The on-robot bridge is a separate real file (robot/jackal_bridge.py), read as text
# and sent over SSH -- NOT imported here (it targets ROS Melodic / Python 2).
_BRIDGE_PATH = Path(__file__).with_name("jackal_bridge.py")


def _load_bridge_source():
    """Read the on-robot bridge source (sent verbatim over SSH, run on the
    Jackal under ROS Melodic -- NOT imported here). See robot/jackal_bridge.py."""
    try:
        return _BRIDGE_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        raise JackalConnectionError(
            "remote bridge source is unavailable: {}".format(_BRIDGE_PATH)
        ) from exc


class JackalConnection:
    """Combined SensorSource/CommandSink implemented over one SSH channel."""

    is_jackal_connection = True

    def __init__(
        self,
        cfg,
        settings: Optional[JackalConnectionConfig] = None,
        *,
        host: Optional[str] = None,
    ) -> None:
        self.settings = settings or JackalConnectionConfig(host=host or "jackalw")
        if host is not None and settings is not None and host != settings.host:
            raise ValueError("host and settings.host disagree")
        if not math.isclose(self.settings.forward_distance_m, 0.25, abs_tol=1e-12):
            raise ValueError("Jackal forward command must remain fixed at 0.25 m")
        if not math.isclose(self.settings.turn_angle_deg, 30.0, abs_tol=1e-12):
            raise ValueError("Jackal turn commands must remain fixed at 30 degrees")
        # Camera topics + resolution come from configs/jackal.yaml (via RobotConfig).
        self._cameras = list(cfg.cameras)
        for cam in self._cameras:
            if not cam.topic_rgb or not cam.topic_depth:
                raise ValueError(
                    "camera '{}' needs topic_rgb/topic_depth in configs/jackal.yaml".format(
                        cam.name
                    )
                )
        self._client = None
        self._channel = None
        self._stdin = None
        self._stdout = None
        self._stderr = None
        self._request_lock = threading.Lock()
        self._next_id = 1
        self._closed = False
        self._camera_launch_pid: Optional[int] = None
        self._frame_cache: Dict[str, np.ndarray] = {}
        self._connect()

    @property
    def camera_topics(self) -> Tuple[str, ...]:
        topics = []
        for camera in self._cameras:
            topics.extend((camera.topic_rgb, camera.topic_depth))
        return tuple(topics)

    def _connect(self) -> None:
        try:
            import paramiko
        except ImportError as exc:
            raise JackalConnectionError(
                "Jackal backend requires paramiko (pip install paramiko)"
            ) from exc

        hostname = self.settings.host
        username = self.settings.ssh_user
        port = self.settings.ssh_port
        key_files = []
        ssh_config_path = Path.home() / ".ssh" / "config"
        if ssh_config_path.exists():
            ssh_config = paramiko.SSHConfig()
            with ssh_config_path.open(encoding="utf-8") as stream:
                ssh_config.parse(stream)
            entry = ssh_config.lookup(self.settings.host)
            hostname = entry.get("hostname", hostname)
            username = entry.get("user", username)
            port = int(entry.get("port", port))
            for key_file in entry.get("identityfile", []):
                expanded = os.path.expanduser(key_file)
                if os.path.isfile(expanded):
                    key_files.append(expanded)

        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.RejectPolicy())
        try:
            client.connect(
                hostname=hostname,
                username=username,
                port=port,
                password=self.settings.ssh_password or None,   # "" -> key-based auth only
                key_filename=key_files or None,
                timeout=self.settings.connect_timeout,
                banner_timeout=self.settings.connect_timeout,
                auth_timeout=self.settings.connect_timeout,
                allow_agent=True,
                look_for_keys=True,
            )
            self._client = client
            self._ensure_camera_launch()
            self._start_bridge()
        except Exception as exc:
            try:
                self._stop_owned_camera_launch()
            finally:
                client.close()
                self._client = None
            if isinstance(exc, JackalConnectionError):
                raise
            raise JackalConnectionError("Jackal SSH setup failed: {}".format(exc)) from exc

    def _ros_shell(self, body: str) -> str:
        return "source {} && source {} && {}".format(
            shlex.quote(self.settings.ros_setup),
            shlex.quote(self.settings.workspace_setup),
            body,
        )

    def _remote_capture(self, body: str, timeout: float = 15.0) -> Tuple[int, str, str]:
        if self._client is None:
            raise JackalConnectionError("SSH client is not connected")
        command = "bash -lc " + shlex.quote(body)
        unused_stdin, stdout, stderr = self._client.exec_command(
            command, timeout=timeout
        )
        output = stdout.read().decode("utf-8", errors="replace")
        error = stderr.read().decode("utf-8", errors="replace")
        status = stdout.channel.recv_exit_status()
        return status, output, error

    def _camera_node_running(self) -> bool:
        status, output, unused_error = self._remote_capture(
            self._ros_shell("rosnode list 2>/dev/null"), timeout=10.0
        )
        return status == 0 and self.settings.camera_node in output.splitlines()

    def _ensure_camera_launch(self) -> None:
        if self._camera_node_running():
            return
        if not self.settings.auto_start_cameras:
            raise JackalConnectionError(
                "{} is not running on the Jackal".format(self.settings.camera_node)
            )
        camera_fps = max(1, int(round(self.settings.camera_fps)))
        launch = "roslaunch {} {} fps:={}".format(
            shlex.quote(self.settings.camera_launch_package),
            shlex.quote(self.settings.camera_launch_file),
            camera_fps,
        )
        body = "source {} && source {} && {{ {} & echo $!; }}".format(
            shlex.quote(self.settings.ros_setup),
            shlex.quote(self.settings.workspace_setup),
            "setsid nohup {} >{} 2>&1 </dev/null".format(
                launch, shlex.quote(self.settings.camera_log)
            ),
        )
        status, output, error = self._remote_capture(body, timeout=10.0)
        if status != 0 or not output.strip().splitlines():
            raise JackalConnectionError(
                "failed to launch dual D455 cameras: {}".format(error.strip())
            )
        self._camera_launch_pid = int(output.strip().splitlines()[-1])
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            if self._camera_node_running():
                return
            time.sleep(0.25)
        raise JackalConnectionError(
            "dual D455 launch did not become ready; see {}".format(
                self.settings.camera_log
            )
        )

    def _start_bridge(self) -> None:
        if self._client is None:
            raise JackalConnectionError("SSH client is not connected")
        camera_topics = list(self.camera_topics)
        payload = {
            "cmd_vel_topic": self.settings.cmd_vel_topic,
            "odom_topic": self.settings.odom_topic,
            "camera_topics": camera_topics,
            "rgb_topics": [camera.topic_rgb for camera in self._cameras],
            "depth_topics": [camera.topic_depth for camera in self._cameras],
            "camera_pairs": [
                {"rgb": camera.topic_rgb, "depth": camera.topic_depth}
                for camera in self._cameras
            ],
            "pair_max_skew_seconds": self.settings.max_rgb_depth_skew_seconds,
            "jpeg_quality": self.settings.jpeg_quality,
            "linear_speed": self.settings.linear_speed,
            "angular_speed": self.settings.angular_speed,
            "forward_distance_m": self.settings.forward_distance_m,
            "turn_angle_deg": self.settings.turn_angle_deg,
            "linear_acceleration": self.settings.linear_acceleration,
            "angular_acceleration": self.settings.angular_acceleration,
            "distance_tolerance": self.settings.distance_tolerance_m,
            "distance_brake_delay_s": self.settings.distance_brake_delay_s,
            "angle_brake_delay_s": self.settings.angle_brake_delay_s,
            "angle_tolerance_deg": self.settings.angle_tolerance_deg,
            "rotation_kp": self.settings.rotation_kp,
            "min_angular_correction_speed": (
                self.settings.min_angular_correction_speed
            ),
            "rotation_settle_seconds": self.settings.rotation_settle_seconds,
            "goal_timeout": self.settings.goal_timeout,
            "max_linear": self.settings.max_linear,
            "max_angular": self.settings.max_angular,
            "max_command_duration": self.settings.max_command_duration,
            "publish_rate": 20.0,
        }
        encoded_config = base64.b64encode(json.dumps(payload).encode()).decode()
        remote_python = (
            "source {} && exec python -u -c {} {}".format(
                shlex.quote(self.settings.ros_setup),
                shlex.quote(_load_bridge_source()),
                shlex.quote(encoded_config),
            )
        )
        channel = self._client.get_transport().open_session(
            timeout=self.settings.connect_timeout
        )
        channel.settimeout(self.settings.request_timeout)
        channel.exec_command("bash -lc " + shlex.quote(remote_python))
        self._channel = channel
        self._stdin = channel.makefile("wb", 0)
        self._stdout = channel.makefile("rb", 0)
        self._stderr = channel.makefile_stderr("rb", 0)
        deadline = time.monotonic() + self.settings.connect_timeout
        while time.monotonic() < deadline:
            message = self._read_message()
            if message.get("event") == "ready":
                return
            if message.get("event") == "fatal":
                raise JackalConnectionError(message.get("error", "bridge failed"))
        raise JackalConnectionError("remote ROS bridge did not become ready")

    def _read_message(self) -> dict:
        if self._stdout is None:
            raise JackalConnectionError("remote bridge stdout is unavailable")
        line = self._stdout.readline()
        if not line:
            details = ""
            if self._stderr is not None:
                try:
                    details = self._stderr.read().decode("utf-8", errors="replace")
                except Exception:
                    pass
            raise JackalConnectionError("remote ROS bridge closed: " + details)
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="replace")
        try:
            return json.loads(line)
        except json.JSONDecodeError as exc:
            raise JackalConnectionError(
                "invalid response from remote bridge: {!r}".format(line)
            ) from exc

    def _request(self, command: str, **payload) -> dict:
        if self._closed or self._stdin is None:
            raise JackalConnectionError("Jackal connection is closed")
        with self._request_lock:
            request_id = self._next_id
            self._next_id += 1
            message = {"id": request_id, "command": command}
            message.update(payload)
            data = (json.dumps(message, separators=(",", ":")) + "\n").encode()
            self._stdin.write(data)
            self._stdin.flush()
            while True:
                response = self._read_message()
                if response.get("id") != request_id:
                    # Asynchronous diagnostics do not satisfy a request.
                    continue
                if not response.get("ok", False):
                    raise JackalCommandError(
                        response.get("error", "remote command failed")
                    )
                return response

    def status(self) -> dict:
        return self._request("status")

    def get_odom(self) -> np.ndarray:
        status = self.status()
        odom = status.get("odom")
        if odom is None:
            raise JackalConnectionError("no odometry received yet")
        if time.time() - float(odom["received_at"]) > 1.0:
            raise JackalConnectionError("odometry is stale")
        return np.array([odom["x"], odom["y"], odom["yaw"]], dtype=np.float64)

    def refresh_frames(
        self,
        timeout: float = 5.0,
        *,
        _allow_owned_camera_restart: bool = True,
    ) -> Dict[str, np.ndarray]:
        try:
            import cv2
        except ImportError as exc:
            raise JackalConnectionError(
                "Jackal camera decoding requires opencv-python"
            ) from exc

        deadline = time.monotonic() + timeout
        wanted = list(self.camera_topics)
        missing = list(wanted)
        latest_ages: Dict[str, float] = {}
        while time.monotonic() < deadline:
            response = self._request("observe", topics=wanted)
            decoded: Dict[str, np.ndarray] = {}
            stamps: Dict[str, float] = {}
            for frame in response.get("frames", []):
                topic = frame["topic"]
                age = float(frame.get("age", float("inf")))
                latest_ages[topic] = age
                if age > 2.0:
                    continue
                encoded = np.frombuffer(base64.b64decode(frame["data"]), np.uint8)
                image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
                if image is None:
                    raise JackalConnectionError(
                        "could not decode {}".format(topic)
                    )
                if frame.get("transport") == "jpeg":
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    if image.dtype != np.uint8 or image.ndim != 3:
                        raise JackalConnectionError(
                            "RGB frame {} was not decoded as uint8 HxWx3".format(
                                topic
                            )
                        )
                elif image.dtype != np.uint16 or image.ndim != 2:
                    raise JackalConnectionError(
                        "depth frame {} was not decoded as uint16 HxW".format(
                            topic
                        )
                    )
                decoded[topic] = image
                stamps[topic] = float(frame.get("stamp", 0.0))
            missing = [topic for topic in wanted if topic not in decoded]
            if not missing:
                for camera in self._cameras:
                    rgb = decoded[camera.topic_rgb]
                    depth = decoded[camera.topic_depth]
                    expected_hw = (camera.height, camera.width)
                    if rgb.shape[:2] != expected_hw or depth.shape != expected_hw:
                        raise JackalConnectionError(
                            "{} frame size mismatch: RGB {}, depth {}, expected {}".format(
                                camera.name, rgb.shape, depth.shape, expected_hw
                            )
                        )
                    rgb_stamp = stamps[camera.topic_rgb]
                    depth_stamp = stamps[camera.topic_depth]
                    if (rgb_stamp > 0.0 and depth_stamp > 0.0 and
                            abs(rgb_stamp - depth_stamp) >
                            self.settings.max_rgb_depth_skew_seconds):
                        raise JackalConnectionError(
                            "{} RGB/depth timestamps differ by {:.3f}s".format(
                                camera.name, abs(rgb_stamp - depth_stamp)
                            )
                        )
                self._frame_cache = decoded
                return dict(decoded)
            time.sleep(0.05)

        # The minimal dual-D455 publisher can remain alive while one RealSense
        # pipeline silently stops yielding frames.  Restart only a launch that
        # this connection created; never terminate a user/system-owned node.
        if _allow_owned_camera_restart and self._camera_launch_pid is not None:
            print(
                ">>> warning: camera stream stalled; restarting owned "
                "dual-D455 launch once"
            )
            self._restart_owned_camera_launch()
            return self.refresh_frames(
                timeout=max(10.0, timeout),
                _allow_owned_camera_restart=False,
            )

        age_details = []
        for topic in missing:
            age = latest_ages.get(topic)
            age_details.append(
                "{} ({})".format(
                    topic,
                    "never received" if age is None else "stale {:.1f}s".format(age),
                )
            )
        raise JackalConnectionError(
            "camera timeout; missing fresh frame(s): {}".format(
                ", ".join(age_details)
            )
        )

    def _topic_for_camera(self, camera_name: str, kind: str) -> str:
        for camera in self._cameras:
            if camera.name == camera_name:
                return camera.topic_rgb if kind == "rgb" else camera.topic_depth
        raise KeyError("unknown Jackal camera {!r}".format(camera_name))

    def get_rgb(self, camera_name: str) -> np.ndarray:
        topic = self._topic_for_camera(camera_name, "rgb")
        if topic not in self._frame_cache:
            self.refresh_frames()
        return self._frame_cache[topic].copy()

    def get_depth(self, camera_name: str) -> np.ndarray:
        topic = self._topic_for_camera(camera_name, "depth")
        if topic not in self._frame_cache:
            self.refresh_frames()
        return self._frame_cache[topic].copy()

    def wait_ready(self, timeout: float = 25.0) -> bool:
        deadline = time.monotonic() + timeout
        last_error: Optional[Exception] = None
        while time.monotonic() < deadline:
            try:
                status = self.status()
                if status.get("odom") is None:
                    raise JackalConnectionError("waiting for odometry")
                if int(status.get("cmd_vel_connections", 0)) < 1:
                    raise JackalConnectionError("waiting for /cmd_vel subscriber")
                self.refresh_frames(timeout=min(5.0, max(0.1, deadline - time.monotonic())))
                return True
            except JackalConnectionError as exc:
                last_error = exc
                time.sleep(0.1)
        raise JackalConnectionError(
            "Jackal did not become ready: {}".format(last_error or "timeout")
        )

    def _wait_for_goal(self, accepted: dict, timeout: Optional[float] = None) -> dict:
        budget = self.settings.goal_timeout if timeout is None else float(timeout)
        deadline = time.monotonic() + budget + 2.0
        while time.monotonic() < deadline:
            status = self.status()
            if status.get("goal") is None:
                result = status.get("last_goal_result")
                if result is None:
                    raise JackalCommandError("goal ended without a result")
                if result.get("result") != "succeeded":
                    target = float(result.get("target", 0.0))
                    progress = float(result.get("progress", 0.0))
                    if result.get("kind") == "rotation":
                        detail = (
                            "target={:.2f}deg, progress={:.2f}deg, "
                            "remaining={:.2f}deg"
                        ).format(
                            math.degrees(target),
                            math.degrees(progress),
                            math.degrees(target - progress),
                        )
                    else:
                        detail = (
                            "target={:.3f}m, progress={:.3f}m, "
                            "remaining={:.3f}m"
                        ).format(target, progress, target - progress)
                    raise JackalCommandError(
                        "goal {}: {} ({})".format(
                            result.get("result", "failed"),
                            result.get("reason", "unknown reason"),
                            detail,
                        )
                    )
                return result
            time.sleep(0.05)
        self.stop()
        raise JackalCommandError("timeout waiting for odometry goal")

    # --- general motion: arbitrary distance / angle / velocity ------------------- #
    def drive_distance(self, distance, *, speed=None, acceleration=None,
                       timeout=None, wait=True) -> dict:
        """Drive an ARBITRARY odometry distance (m); sign = direction. Smoothly ramps
        and brakes on /odometry/filtered. wait=True blocks for the terminal result."""
        payload = {"distance": float(distance)}
        if speed is not None:
            payload["speed"] = float(speed)
        if acceleration is not None:
            payload["acceleration"] = float(acceleration)
        if timeout is not None:
            payload["timeout"] = float(timeout)
        accepted = self._request("drive_distance", **payload)
        return self._wait_for_goal(accepted, timeout) if wait else accepted

    def rotate_angle(self, angle, *, speed=None, acceleration=None,
                     timeout=None, wait=True) -> dict:
        """Rotate by an ARBITRARY angle (rad); + = left, - = right."""
        payload = {"angle": float(angle)}
        if speed is not None:
            payload["speed"] = float(speed)
        if acceleration is not None:
            payload["acceleration"] = float(acceleration)
        if timeout is not None:
            payload["timeout"] = float(timeout)
        accepted = self._request("rotate_angle", **payload)
        return self._wait_for_goal(accepted, timeout) if wait else accepted

    def move(self, linear, angular=0.0, *, duration=0.25) -> dict:
        """Publish a bounded velocity (m/s, rad/s) for `duration` s, then auto-zero."""
        return self._request("move", linear=float(linear), angular=float(angular),
                             duration=float(duration))

    # --- fixed convenience actions (RobotEnv uses these) = the general goals ----- #
    def forward_25cm(self) -> dict:
        return self.drive_distance(self.settings.forward_distance_m)

    def turn_left_30deg(self) -> dict:
        return self.rotate_angle(math.radians(self.settings.turn_angle_deg))

    def turn_right_30deg(self) -> dict:
        return self.rotate_angle(-math.radians(self.settings.turn_angle_deg))

    def stop(self) -> None:
        if not self._closed:
            self._request("stop")

    def close(self) -> None:
        if self._closed:
            return
        try:
            try:
                self._request("shutdown")
            except Exception:
                pass
            if self._channel is not None:
                self._channel.close()
            self._stop_owned_camera_launch()
        finally:
            self._closed = True
            if self._client is not None:
                self._client.close()
            self._client = None

    def _stop_owned_camera_launch(self) -> None:
        if self._camera_launch_pid is None or self._client is None:
            return
        # Stop only a launch process created by this connection.  A camera node
        # that was already running belongs to the user/system.
        body = (
            "kill -INT -- -{pid} 2>/dev/null || "
            "kill -INT {pid} 2>/dev/null || true"
        ).format(pid=self._camera_launch_pid)
        try:
            self._remote_capture(body, timeout=5.0)
        except Exception:
            pass
        finally:
            self._camera_launch_pid = None

    def _restart_owned_camera_launch(self) -> None:
        """Restart only the dual-D455 launch created by this connection."""
        if self._camera_launch_pid is None:
            raise JackalConnectionError(
                "cannot restart a camera launch not owned by this connection"
            )
        old_pid = self._camera_launch_pid
        self._stop_owned_camera_launch()

        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            if not self._camera_node_running():
                break
            time.sleep(0.2)
        else:
            # The process group belongs to this connection, so escalating from
            # SIGINT to SIGTERM cannot affect a pre-existing camera launch.
            body = "kill -TERM -- -{pid} 2>/dev/null || true".format(
                pid=old_pid
            )
            self._remote_capture(body, timeout=5.0)
            time.sleep(0.5)
            if self._camera_node_running():
                raise JackalConnectionError(
                    "owned dual-D455 launch did not stop for recovery"
                )

        self._frame_cache.clear()
        self._ensure_camera_launch()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
