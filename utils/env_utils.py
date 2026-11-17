"""Build a habitat.Env + dataset the way the v0.2.2 scripts did, on v0.3.3.

v0.2.2 used ``NavRLEnv`` and stamped ``episode.scene_dataset_config`` from
``config.SIMULATOR.SCENE_DATASET`` before constructing the env. v0.3.3 has no
``NavRLEnv``; we use ``habitat.Env`` directly. The Env reads the FIRST episode's
``scene_dataset_config`` at construction time to choose the scene-dataset config
file for the simulator, so the stamping must happen on the dataset BEFORE the
Env is built -- exactly as before. For MP3D this is what aligns the render mesh
(.glb) and the semantic mesh (_semantic.ply) so the GT semantic sensor is
correct.
"""
import copy

import numpy as np
import habitat
from omegaconf import OmegaConf

from utils.habitat_cfg import dataset_type, scene_dataset, MAIN_AGENT


# obs keys of the sub (top) camera -- distinct from the main "rgb"/"depth"/"semantic".
SUB_RGB_UUID = "rgb_top"
SUB_DEPTH_UUID = "depth_top"
SUB_SEMANTIC_UUID = "semantic_top"


def add_sub_camera(config, height=None, delta_h=0.5, use_semantic=None, tilt_deg=0.0):
    """Add a SECOND (higher) camera to main_agent, mirroring the main sensors.

    Clones ``rgb_sensor`` -> ``rgb_sensor_top`` and ``depth_sensor`` ->
    ``depth_sensor_top`` (and, if the main agent has a semantic sensor and
    ``use_semantic`` isn't False, ``semantic_sensor`` -> ``semantic_sensor_top``),
    raised in +y. Each clone gets a distinct ``uuid`` so its observation shows up
    under a separate key (``rgb_top`` / ``depth_top`` / ``semantic_top``).

    height: absolute sensor y (m). If None, use main height + ``delta_h``.
    use_semantic: None -> mirror main (add if main has semantic); True/False forces.
    tilt_deg: pitch the sub camera DOWN by this many degrees (0 = level).

    Call this BEFORE make_env, for navigation only (occ-map collection ignores it).
    Returns the (mutated) config.
    """
    ss = config.habitat.simulator.agents[MAIN_AGENT].sim_sensors
    OmegaConf.set_readonly(config, False)
    OmegaConf.set_struct(config, False)

    def _clone(src_key, new_key, uuid):
        node = copy.deepcopy(ss[src_key])
        OmegaConf.set_struct(node, False)   # allow adding the uuid field
        OmegaConf.set_readonly(node, False)
        pos = list(node.position)
        pos[1] = float(height) if height is not None else float(pos[1]) + float(delta_h)
        node.position = pos
        # Pitch the camera DOWN by tilt_deg: the camera looks along -Z, and a
        # negative rotation about +X swings -Z toward -Y (floor-ward).
        if tilt_deg:
            node.orientation = [float(-np.deg2rad(tilt_deg)), 0.0, 0.0]
        node.uuid = uuid                # -> obs key (Sensor honors config.uuid)
        ss[new_key] = node

    _clone("rgb_sensor", "rgb_sensor_top", SUB_RGB_UUID)
    _clone("depth_sensor", "depth_sensor_top", SUB_DEPTH_UUID)
    want_sem = ("semantic_sensor" in ss) if use_semantic is None else bool(use_semantic)
    if want_sem and "semantic_sensor" in ss:
        _clone("semantic_sensor", "semantic_sensor_top", SUB_SEMANTIC_UUID)

    OmegaConf.set_struct(config, True)
    OmegaConf.set_readonly(config, True)
    return config


def has_sub_camera(config):
    ss = config.habitat.simulator.agents[MAIN_AGENT].sim_sensors
    return "rgb_sensor_top" in ss


def occ_map_reachable(gmap, start_xyz, goal_xyz_list):
    """True if some goal is reachable from ``start`` on the OCCUPANCY MAP.

    The sim navmesh and the depth-built occupancy map do not agree: the map's free
    space is fragmented (thick/noisy walls seal doorways), so a start that is only
    navmesh-reachable can leave the FMM planner with no path at all. Connectivity on
    the map grid is exactly what the planner can use, so test that.
    """
    from scipy.ndimage import label

    lab, _ = label(gmap.free)

    def cell(p):
        rc = gmap.point_to_grid(p)
        r = int(np.clip(round(rc[0]), 0, gmap.H - 1))
        c = int(np.clip(round(rc[1]), 0, gmap.W - 1))
        return r, c

    sr, sc = cell(start_xyz)
    # The START cell itself must be free on the map: the sim may call a pose
    # navigable while the (radius-dilated, depth-noisy) map marks it blocked, and
    # then the planner has no valid cell to start from and reports 'stuck'.
    # No nearest_free fallback here -- that is exactly what hid the bad starts.
    if not gmap.free[sr, sc]:
        return False
    s_lab = lab[sr, sc]
    if s_lab == 0:
        return False
    for g in goal_xyz_list:
        gr, gc = cell(g)
        if not gmap.free[gr, gc]:
            gr, gc = gmap.nearest_free(gr, gc)
        if lab[gr, gc] == s_lab:
            return True
    return False


def _has_min_obstacle_clearance(sim, point, min_clearance):
    if min_clearance <= 0.0:
        return True
    clearance = float(
        sim.pathfinder.distance_to_closest_obstacle(point, min_clearance)
    )
    return bool(np.isfinite(clearance) and clearance >= min_clearance)


def randomize_episode_start(env, min_dist=3.0, rng=None, max_tries=300, gmap=None,
                            min_clearance=0.0):
    """Teleport the agent to a random navigable start at least ``min_dist`` (geodesic)
    from the episode's goal viewpoints, with a random heading.

    The daejeon episodes all start at the origin, which makes every episode the same
    difficulty. Call right AFTER ``env.reset()`` (so the correct scene is loaded) and
    BEFORE the agent's own reset. Returns (obs, geodesic_dist) -- or (None, None) if
    no valid start was found, in which case keep the episode's own start.

    ``gmap``: if given (the agent's occupancy map), the start is additionally required
    to be reachable to a goal ON THAT MAP -- otherwise the FMM planner starts 'stuck'
    and the episode is unwinnable for reasons unrelated to the policy.
    ``min_clearance`` optionally rejects starts closer than that many metres to a
    navmesh obstacle. It defaults to zero to preserve ordinary random-start behavior.
    """
    import quaternion as q

    sim = env.sim
    ep = env.current_episode
    rng = rng or np.random
    targets = [vp.agent_state.position for g in ep.goals for vp in g.view_points]
    if not targets:
        targets = [g.position for g in ep.goals]

    for _ in range(max_tries):
        p = np.array(sim.sample_navigable_point(), dtype=np.float32)
        if not _has_min_obstacle_clearance(sim, p, min_clearance):
            continue
        d = sim.geodesic_distance(p, targets)
        if not np.isfinite(d) or d < min_dist:
            continue
        if gmap is not None and not occ_map_reachable(gmap, p, targets):
            continue
        theta = float(rng.uniform(-np.pi, np.pi))
        rot = q.from_rotation_vector([0.0, theta, 0.0])
        obs = sim.get_observations_at(p, rot, keep_agent_at_new_pose=True)
        return obs, float(d)
    return None, None


def relocate_start_if_keepout(env, gmap, rng=None, min_dist=1.0,
                              min_clearance=0.3):
    """If the episode's start pose sits inside a planning KEEP-OUT cell (a region excluded from
    planning, e.g. the entrance we marked keep-out), teleport to the nearest navigable,
    goal-reachable start with at least ``min_clearance`` metres of navmesh-obstacle
    clearance instead -- a start in keep-out is a map-edit artifact, not a real scenario,
    and leaves the FMM planner 'stuck' from step 0. Call AFTER env.reset() + the agent's
    reset (needs the agent's gmap). Returns (obs, geodesic_dist) if relocated, else
    (None, None). NOTE: like --random_start, the post-reset teleport makes habitat SPL less
    meaningful for the relocated episode (SR is still valid)."""
    import quaternion as q

    if gmap is None or getattr(gmap, "keepout", None) is None:
        return None, None
    sim = env.sim
    ep = env.current_episode
    rng = rng or np.random
    st = np.asarray(sim.get_agent_state().position, dtype=np.float32)
    rc = gmap.point_to_grid(st)
    r0, c0 = int(round(rc[0])), int(round(rc[1]))
    if not (gmap.in_bounds(r0, c0) and bool(gmap.keepout[r0, c0])):
        return None, None                              # start not in keep-out -> nothing to do
    targets = [vp.agent_state.position for g in ep.goals for vp in g.view_points]
    if not targets:
        targets = [g.position for g in ep.goals]
    # nearest traversable (free_seen -> excludes keep-out) cell, snapped to the navmesh
    r, c = gmap.nearest_free(r0, c0)
    x, z = gmap.grid_to_world_xz(r, c)
    p = np.array(sim.pathfinder.snap_point([float(x), float(st[1]), float(z)]), dtype=np.float32)
    if (np.all(np.isfinite(p))
            and _has_min_obstacle_clearance(sim, p, min_clearance)
            and occ_map_reachable(gmap, p, targets)):
        theta = float(rng.uniform(-np.pi, np.pi))
        rot = q.from_rotation_vector([0.0, theta, 0.0])
        obs = sim.get_observations_at(p, rot, keep_agent_at_new_pose=True)
        return obs, float(sim.geodesic_distance(p, targets))
    # fallback: any reachable, sufficiently clear navigable start
    # (also excludes keep-out via occ_map_reachable)
    return randomize_episode_start(
        env, min_dist, rng, gmap=gmap, min_clearance=min_clearance
    )


def apply_start_override(env, overrides):
    """Pin a DETERMINISTIC start pose for specific episodes (identical across every run/config).

    Used to replace an unwinnable auto-relocated start (e.g. a keep-out entrance that traps the
    planner) with one hand-picked open, goal-reachable pose, so the episode is comparable across
    all settings. ``overrides`` maps (scene_substr, category) -> (x, z, yaw_deg_or_None); the world
    y is snapped to the navmesh. yaw_deg None -> face the goal centroid; else it is a rotation about
    +y (same convention as randomize_episode_start's theta). Call AFTER env.reset() + the agent's
    reset. Returns (obs, geodesic_dist) if this episode was overridden, else (None, None). Like
    --random_start, the post-reset teleport makes SPL less meaningful (SR stays valid)."""
    import quaternion as q

    if not overrides:
        return None, None
    ep = env.current_episode
    match = None
    for (sub, cat), pose in overrides.items():
        if sub in ep.scene_id and getattr(ep, "object_category", None) == cat:
            match = pose
            break
    if match is None:
        return None, None
    sim = env.sim
    x, z, yaw_deg = match
    st = np.asarray(sim.get_agent_state().position, dtype=np.float32)
    p = np.array(sim.pathfinder.snap_point([float(x), float(st[1]), float(z)]), dtype=np.float32)
    if not np.all(np.isfinite(p)):
        return None, None
    targets = [vp.agent_state.position for g in ep.goals for vp in g.view_points]
    if not targets:
        targets = [g.position for g in ep.goals]
    if yaw_deg is None:                                # face the goal centroid
        gc = np.mean([np.asarray(g.position, np.float32) for g in ep.goals], axis=0)
        dx, dz = float(gc[0] - p[0]), float(gc[2] - p[2])
        theta = float(np.arctan2(-dx, -dz))            # agent forward is local -Z, rotated about +y
    else:
        theta = float(np.deg2rad(yaw_deg))
    rot = q.from_rotation_vector([0.0, theta, 0.0])
    obs = sim.get_observations_at(p, rot, keep_agent_at_new_pose=True)
    return obs, float(sim.geodesic_distance(p, targets))


def parse_start_overrides(spec):
    """Parse a --start_override string into {(scene_substr, category): (x, z, yaw_deg|None)}.

    Format: "sceneSubstr:category=x,z[,yaw_deg];sceneSubstr:category=x,z" (';'-separated entries).
    yaw_deg omitted -> face the goal at start."""
    out = {}
    if not spec:
        return out
    for entry in spec.split(";"):
        entry = entry.strip()
        if not entry or "=" not in entry or ":" not in entry.split("=", 1)[0]:
            continue
        key, val = entry.split("=", 1)
        sub, cat = key.split(":", 1)
        nums = [float(v) for v in val.split(",") if v.strip() != ""]
        x, z = nums[0], nums[1]
        yaw = nums[2] if len(nums) >= 3 else None
        out[(sub.strip(), cat.strip())] = (x, z, yaw)
    return out


def make_env_and_dataset(config, seed=42, sub_camera=None):
    """Return (env, dataset). Stamps the scene-dataset config onto every episode.

    sub_camera: None (off) or a dict of add_sub_camera kwargs
    (e.g. {"height": 1.6} or {"delta_h": 0.5, "use_semantic": True}) to attach a
    second, higher camera for navigation.
    """
    if sub_camera is not None:
        add_sub_camera(config, **sub_camera)

    dataset = habitat.make_dataset(
        dataset_type(config), config=config.habitat.dataset
    )

    sd = scene_dataset(config)
    if sd and sd != "default":
        for ep in dataset.episodes:
            ep.scene_dataset_config = sd

    env = habitat.Env(config, dataset=dataset)
    env.seed(seed)
    return env, dataset
