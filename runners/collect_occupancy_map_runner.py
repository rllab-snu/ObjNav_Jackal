"""Collect occupancy maps by teleport + 360° scan.

Accumulates an ``OccupancyMap`` (obstacle + seen, both 0/1) from depth: the agent
is **teleported** to each sampled navigable point and does a full **360° in-place
rotation**, integrating depth at every yaw. With enough sample points this fills
most of the floor quickly and uniformly (no path-following, no getting stuck).

Entry arg is the NUMBER OF SAMPLE POINTS (not timesteps); make it large to cover
the floor.

Output per scene/floor:
  <save_dir>/<scene>_<floor>_occ_map.dat.gz   # {'obstacle_map','seen_map','map_params'}
  <save_dir>/<scene>_<floor>_occ.png          # visualization
  <save_dir>/<scene>_<floor>_explore.mp4      # optional (--vis_video)
"""
import os
import gzip
import pickle
import numpy as np
import cv2
from tqdm import tqdm
import quaternion as q
import matplotlib.pyplot as plt

from maps.occupancy_map import OccupancyMap
from navigation.geometry import intrinsics_from_hfov
from utils.nav_viz import write_video
from utils import habitat_cfg


class CollectOccupancyMapRunner:
    def __init__(self, env, dataset, args, config):
        self.env = env
        self.dataset = dataset

        scene_ids = dataset.scene_ids
        self.scene_episodes_list = []
        if args.scene_idx == -1:
            for sid in scene_ids:
                self.scene_episodes_list.append(dataset.get_scene_episodes(sid))
        else:
            assert args.scene_idx in range(len(scene_ids)), \
                "args.scene_idx should be in [0,{}]".format(len(scene_ids) - 1)
            self.scene_episodes_list.append(
                dataset.get_scene_episodes(scene_ids[args.scene_idx]))
            print(">>> Using specific scene:", scene_ids[args.scene_idx])
        print(">>> Total {} scenes".format(len(self.scene_episodes_list)))

        self.num_points = int(args.num_points)
        self.only_floor = int(getattr(args, "floor", -1))
        self.no_objects = bool(getattr(args, "no_objects", False))
        self.agent_radius = float(config.habitat.simulator.agents.main_agent.radius)
        self.agent_height = float(config.habitat.simulator.agents.main_agent.height)
        turn_angle = float(config.habitat.simulator.turn_angle)
        self.num_rotations = int(getattr(args, "num_rotations", 0)) or max(1, round(360.0 / turn_angle))

        w, h = habitat_cfg.rgb_hw(config)
        rgb = habitat_cfg.sim_sensor(config, "rgb_sensor")
        self.fx, self.fy, self.cx, self.cy = intrinsics_from_hfov(w, h, float(rgb.hfov))
        self.sensor_height = float(rgb.position[1])
        depth = habitat_cfg.sim_sensor(config, "depth_sensor")
        self.min_depth = float(depth.min_depth)
        self.max_depth = float(depth.max_depth)

        occ = config.occ_map if "occ_map" in config else {}

        def _p(k, d):
            return occ[k] if k in occ else d

        self.resolution_m = float(_p("resolution_m", 0.05))
        # CLI values (--obstacle_height_min/max) take precedence over a config `occ_map` block.
        h_min = getattr(args, "obstacle_height_min", None)
        h_max = getattr(args, "obstacle_height_max", None)
        self.obstacle_height_min = float(h_min if h_min is not None else _p("obstacle_height_min", 0.2))
        self.obstacle_height_max = float(h_max if h_max is not None else
            _p("obstacle_height_max", float(config.habitat.simulator.agents.main_agent.height)))
        self.depth_eps = float(_p("depth_eps", 0.05))
        self.depth_subsample = int(_p("depth_subsample", 2))
        self.max_sample_tries = int(_p("max_sample_tries", 1000))

        self.save_dir = args.save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.vis = args.vis
        self.vis_video = bool(getattr(args, "vis_video", False))
        self.video_fps = int(getattr(args, "video_fps", 10))

        # Floors are taken from the scene's own semantic levels (daejeon is
        # single-floor). The old mp3d floor_dict lookup lives in legacy/.
        self._scene_has_floors = False

    # ------------------------------------------------------------------ #
    def get_nearest_floor(self, height):
        if len(self.floor_info) == 1:
            return 0
        return int(np.argmin(np.abs(np.array(self.floor_info) - height)))

    def _sample_on_floor(self, floor):
        sim = self.env.sim
        p = sim.sample_navigable_point()
        if not self._scene_has_floors:
            return np.array(p, dtype=np.float32)
        for _ in range(self.max_sample_tries):
            if self.get_nearest_floor(p[1]) == floor:
                return np.array(p, dtype=np.float32)
            p = sim.sample_navigable_point()
        return np.array(p, dtype=np.float32)

    def _new_map(self, floor_height=None):
        m = OccupancyMap(
            resolution_m=self.resolution_m,
            obstacle_height_min=self.obstacle_height_min,
            obstacle_height_max=self.obstacle_height_max,
            min_depth=self.min_depth, max_depth=self.max_depth,
            depth_eps=self.depth_eps, depth_subsample=self.depth_subsample,
        )
        m.reset(self.env.sim, floor_height=floor_height)
        return m

    def _video_frame(self, obs, occ_map):
        rgb = np.asarray(obs["rgb"])[..., :3].astype(np.uint8)
        H = rgb.shape[0]
        d = np.clip(obs["depth"].squeeze(), 0.0, 1.0)
        dvis = cv2.cvtColor(cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)
        occ = occ_map.visualize()
        ow = max(1, int(round(occ.shape[1] * H / occ.shape[0])))
        occ = cv2.resize(occ, (ow, H), interpolation=cv2.INTER_NEAREST)
        return np.concatenate([rgb, dvis, occ], axis=1)

    # ------------------------------------------------------------------ #
    def collect_map(self):
        for scene_episodes in self.scene_episodes_list:
            self.env.episodes = scene_episodes
            self.env.reset()
            scene_name = self.env.current_episode.scene_id.split('/')[-1].split('.')[0]
            print(">>> Starting scene {}".format(scene_name))
            sim = self.env.sim

            # Optionally drop the spawned scenario objects -> base-scene-only map.
            # Remove them from the render AND recompute the navmesh on the bare
            # stage (so sampling covers where objects used to be).
            if self.no_objects:
                import habitat_sim
                n = sim.get_rigid_object_manager().get_num_objects()
                sim.get_rigid_object_manager().remove_all_objects()
                ns = habitat_sim.NavMeshSettings(); ns.set_defaults()
                ns.agent_radius = self.agent_radius
                ns.agent_height = self.agent_height
                try:
                    ns.include_static_objects = False
                except Exception:
                    pass
                sim.recompute_navmesh(sim.pathfinder, ns)
                print(f">>> no_objects: removed {n} scenario objects, recomputed base-scene navmesh")

            # Single floor plane; count floors from the scene's semantic levels.
            self._scene_has_floors = False
            self.floor_info = [0.0]
            ss = sim.semantic_scene
            total_floor = (len(ss.levels) if ss is not None else 0) or 1
            print(">>> Total {} Floors".format(total_floor))

            for floor in range(total_floor):
                if self.only_floor != -1 and floor != self.only_floor:
                    continue
                print(">>> Floor {} | {} sample points x {} rotations".format(
                    floor, self.num_points, self.num_rotations))
                floor_h = self.floor_info[floor] if self._scene_has_floors else None
                occ_map = self._new_map(floor_height=floor_h)
                frames = []

                for _ in tqdm(range(self.num_points)):
                    p = self._sample_on_floor(floor)
                    for k in range(self.num_rotations):
                        yaw = 2.0 * np.pi * k / self.num_rotations
                        rot = q.from_euler_angles([0.0, yaw, 0.0])
                        obs = sim.get_observations_at(p, rot, keep_agent_at_new_pose=True)
                        depth_m = obs['depth'].squeeze() * self.max_depth
                        cam_pos = np.array(p, dtype=np.float64) + np.array([0.0, self.sensor_height, 0.0])
                        occ_map.update(depth_m, cam_pos, q.as_rotation_vector(rot),
                                       self.fx, self.fy, self.cx, self.cy, floor_y=float(p[1]))
                        if self.vis_video:
                            frames.append(self._video_frame(obs, occ_map))  # RGB

                if frames:
                    write_video(os.path.join(
                        self.save_dir, "{}_{}_explore.mp4".format(scene_name, floor)),
                        frames, fps=self.video_fps)
                    print(">>> Saved exploration video: {}_{}_explore.mp4".format(scene_name, floor))

                with gzip.open(os.path.join(
                        self.save_dir, "{}_{}_occ_map.dat.gz".format(scene_name, floor)), "wb") as f:
                    pickle.dump(occ_map.to_dict(), f)
                if self.vis:
                    plt.imsave(os.path.join(
                        self.save_dir, "{}_{}_occ.png".format(scene_name, floor)),
                        occ_map.visualize())
