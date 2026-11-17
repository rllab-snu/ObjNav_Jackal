"""Camera geometry helpers (Habitat convention).

Habitat's camera looks down its local -Z axis, +X right, +Y up; the image origin
is top-left (row v down, col u right). The camera/agent pose is a world position
plus a rotation vector (axis-angle), so ``R = Rotation.from_rotvec(rotvec)`` maps
camera-frame vectors to world frame. This matches the v0.2.2 base_mp3d
`pixel_to_abs_pos`, generalized to whole images / pixel batches.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R


def intrinsics_from_hfov(width, height, hfov_deg):
    """Return (fx, fy, cx, cy) for a pinhole camera with horizontal FOV."""
    fx = (width / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)
    fy = fx  # square pixels
    cx, cy = width / 2.0, height / 2.0
    return fx, fy, cx, cy


def vertical_half_fov_down(fy, cy, height):
    """Downward half of the vertical FOV (rad): angle from the optical axis to the
    bottom image row. Derived from the intrinsics (hence the sensor HFOV / aspect
    ratio) so it follows the camera config, and uses the actual principal point so
    it stays correct even if the sensor is not vertically centered."""
    return np.arctan((float(height) - float(cy)) / float(fy))


def backproject_pixels(u, v, z, fx, fy, cx, cy, cam_pos, cam_rotvec):
    """Back-project pixel coords (u=col, v=row) at depth z (m) to world points.

    Returns an (N, 3) array of world (x, y=height, z) coordinates.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    x = (u - cx) * z / fx
    y = -(v - cy) * z / fy
    cam = np.stack([x, y, -z], axis=-1)  # habitat camera frame

    rot = R.from_rotvec(np.asarray(cam_rotvec, dtype=np.float64))
    world = np.asarray(cam_pos, dtype=np.float64)[None, :] + rot.apply(cam)
    return world


def backproject_depth_image(
    depth_m, fx, fy, cx, cy, cam_pos, cam_rotvec, subsample=2, z_max=None
):
    """Back-project a (subsampled) depth image to a world point cloud.

    Keeps pixels with depth in (0, z_max). Pixels clamped at the sensor max range
    are unreliable (radial streaks in the occupancy map), so pass z_max a little
    below max_depth to drop them. Returns (points_world (M,3), us, vs).
    """
    H, W = depth_m.shape
    vs, us = np.mgrid[0:H:subsample, 0:W:subsample]
    z = depth_m[::subsample, ::subsample].reshape(-1)
    u = us.reshape(-1).astype(np.float64)
    v = vs.reshape(-1).astype(np.float64)

    valid = z > 1e-4
    if z_max is not None:
        valid &= z < z_max
    u, v, z = u[valid], v[valid], z[valid]
    pts = backproject_pixels(u, v, z, fx, fy, cx, cy, cam_pos, cam_rotvec)
    return pts, u, v


def scan_from_depth(depth, fx, fy, cx, cy, max_depth, sensor_height,
                    h_min, h_max, subsample=8, tilt_deg=0.0, max_range=5.0,
                    sensor_position=None):
    """Depth image -> (N, 2) obstacle points [x, z] in the ROBOT frame, sliced to
    the [h_min, h_max] height band the occupancy map was built with.

    Mirrors ``ParticleFilterLocalizer.extract_scan`` but standalone, so the
    dynamic-map update can extract a scan without a localizer (e.g. oracle pose)."""
    dm = np.asarray(depth).squeeze() * float(max_depth)
    s = int(subsample)
    dm = dm[::s, ::s]
    H, W = dm.shape
    vs, us = np.mgrid[0:H, 0:W]
    us = us.reshape(-1) * s
    vs = vs.reshape(-1) * s
    z = dm.reshape(-1)
    ok = (z > 0.15) & (z < min(float(max_range), float(max_depth) - 0.1))
    if ok.sum() < 10:
        return np.zeros((0, 2))
    cam_pos = np.asarray(
        [0.0, float(sensor_height), 0.0]
        if sensor_position is None else sensor_position,
        dtype=np.float64,
    )
    if cam_pos.shape != (3,):
        raise ValueError("sensor_position must be [right, height, backward] in metres")
    pts = backproject_pixels(us[ok], vs[ok], z[ok], fx, fy, cx, cy,
                             cam_pos,
                             np.array([-np.deg2rad(tilt_deg), 0.0, 0.0]))
    h = pts[:, 1]
    band = (h > float(h_min)) & (h < float(h_max))
    if band.sum() < 10:
        return np.zeros((0, 2))
    return np.stack([pts[band, 0], pts[band, 2]], axis=-1)


def heading_from_rotvec(cam_rotvec):
    """Agent heading (radians) in the world XZ plane.

    The camera looks down -Z in its own frame; rotate that into the world and
    take atan2(x, -z) so headings are comparable on the top-down map.
    """
    rot = R.from_rotvec(np.asarray(cam_rotvec, dtype=np.float64))
    fwd = rot.apply(np.array([0.0, 0.0, -1.0]))
    return np.arctan2(fwd[0], -fwd[2])
