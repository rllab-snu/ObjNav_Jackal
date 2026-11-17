import os
import sys
os.environ['GLOG_minloglevel'] = "3"
os.environ['MAGNUM_LOG'] = "quiet"
os.environ['HABITAT_SIM_LOG'] = "quiet"
sys.setrecursionlimit(10000)

import time
import argparse

from utils.config_utils import load_config
from utils.env_utils import make_env_and_dataset

from runners.collect_occupancy_map_runner import CollectOccupancyMapRunner


def get_args():
    parser = argparse.ArgumentParser()

    ## meta configs ##
    parser.add_argument("--config", type=str, default="daejeon_base.yaml",
                        help="config name relative to ./configs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--version", type=str, default="occupancy_map")

    ## running configs ##
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--num_points", type=int, default=300,
                        help="number of teleport sample points per floor (large -> fuller map)")
    parser.add_argument("--num_rotations", type=int, default=0,
                        help="rotations per point (0 = 360/turn_angle)")
    parser.add_argument("--scene_idx", type=int, default=-1, help="index of scene, -1 for all scenes")
    parser.add_argument("--floor", type=int, default=-1, help="floor index, -1 for all floors")
    parser.add_argument("--no_objects", action="store_true", default=False,
                        help="remove spawned scenario objects -> base-scene-only occ map")
    parser.add_argument("--obstacle_height_min", type=float, default=None,
                        help="lowest geometry (m above floor) counted as obstacle (default 0.2)")
    parser.add_argument("--obstacle_height_max", type=float, default=None,
                        help="highest geometry (m above floor) counted as obstacle "
                             "(default: agent height from the config)")

    ## log configs ##
    parser.add_argument("--vis", action="store_true", default=True)
    parser.add_argument("--vis_video", action="store_true", default=False,
                        help="save a video of the teleport+scan (can be large for many points)")
    parser.add_argument("--video_fps", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="output_maps/{}/{}")

    args = parser.parse_args()
    args.save_dir = args.save_dir.format(args.version, args.split)
    return args


def main(args):
    start_time = time.time()

    # Teleport-based collection samples observations directly (no env stepping
    # toward goals), so no need to raise max_episode_steps.
    config = load_config(args.config, overrides=[
        f"habitat.dataset.split={args.split}",
    ])

    env, dataset = make_env_and_dataset(config, seed=args.seed)

    runner = CollectOccupancyMapRunner(env, dataset, args, config)
    runner.collect_map()
    env.close()

    print(">>> Total Time: {:.2f}s".format(time.time() - start_time))


if __name__ == "__main__":
    args = get_args()
    main(args)
