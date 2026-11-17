"""Apply the manual base-map edits to a freshly collected occupancy map.

  python apply_occ_map_edits.py --src output_maps/occupancy_map/val/base_0_occ_map.dat.gz \
      --out occupancy_map/base_occ_map.dat.gz

Edits (occupancy_map/base_map_edits.json) are stored as world [x, z] cell centres, so they
apply to any map with the same resolution regardless of its grid origin/size:
  obstacle_add -> obstacle_map |= cell, seen_map |= cell   (real obstacles seen too sparsely)
  seen_add     -> seen_map |= cell
  keepout      -> keepout_map (planning-only no-go; navigation/global_occupancy_map.py removes it
                  from the FMM free space and particle seeding, but scan matching is unchanged)
Writes the robot map dict {obstacle_map, seen_map, keepout_map, map_params} + a PNG preview.
"""
import argparse
import gzip
import json
import os
import pickle

import cv2
import numpy as np


def _mask(cells, mp, shape):
    res = float(mp["resolution_m"])
    ox, oz = mp["origin_world_xz"]
    m = np.zeros(shape, bool)
    if not cells:
        return m, 0
    xz = np.asarray(cells, dtype=np.float64)
    cols = np.rint((xz[:, 0] - ox) / res).astype(int)
    rows = np.rint((xz[:, 1] - oz) / res).astype(int)
    ok = (rows >= 0) & (rows < shape[0]) & (cols >= 0) & (cols < shape[1])
    m[rows[ok], cols[ok]] = True
    return m, int((~ok).sum())


def apply_edits(d, edits):
    mp = d["map_params"]
    if not np.isclose(float(mp["resolution_m"]), float(edits["resolution_m"])):
        raise ValueError("map resolution {} != edits resolution {}".format(
            mp["resolution_m"], edits["resolution_m"]))
    obstacle = np.asarray(d["obstacle_map"]).astype(bool)
    seen = np.asarray(d["seen_map"]).astype(bool)
    keepout = np.asarray(d.get("keepout_map", np.zeros_like(obstacle))).astype(bool)
    dropped = 0
    add_obs, n = _mask(edits.get("obstacle_add", []), mp, obstacle.shape); dropped += n
    add_seen, n = _mask(edits.get("seen_add", []), mp, obstacle.shape); dropped += n
    add_keep, n = _mask(edits.get("keepout", []), mp, obstacle.shape); dropped += n
    if dropped:
        print(f">>> warning: {dropped} edit cells fall outside the map grid and were skipped")
    out = dict(d)
    out["obstacle_map"] = obstacle | add_obs
    out["seen_map"] = seen | add_obs | add_seen
    out["keepout_map"] = keepout | add_keep
    return out


def visualize(d):
    """Same scheme as OccupancyMap.visualize: white=unseen, grey=free, black=obstacle."""
    obstacle, seen = d["obstacle_map"].astype(bool), d["seen_map"].astype(bool)
    img = np.full(obstacle.shape + (3,), 255, np.uint8)
    img[seen] = (180, 180, 180)
    img[obstacle] = (0, 0, 0)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True, help="collected map (*_occ_map.dat.gz)")
    ap.add_argument("--out", required=True, help="output map path (.dat.gz); a .png is written next to it")
    ap.add_argument("--edits", default="occupancy_map/base_map_edits.json")
    args = ap.parse_args()

    with gzip.open(args.src, "rb") as f:
        d = pickle.load(f)
    with open(args.edits) as f:
        edits = json.load(f)
    out = apply_edits(d, edits)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with gzip.open(args.out, "wb") as f:
        pickle.dump(out, f)
    png = args.out[:-len(".dat.gz")] if args.out.endswith(".dat.gz") else os.path.splitext(args.out)[0]
    cv2.imwrite(png + ".png", visualize(out))      # grey/black/white -> RGB == BGR
    print(">>> wrote {} (+ .png) | obstacle {} | seen {} | keepout {}".format(
        args.out, int(out["obstacle_map"].sum()), int(out["seen_map"].sum()),
        int(out["keepout_map"].sum())))


if __name__ == "__main__":
    main()
