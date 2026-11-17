"""Build one Topology per prior (base, prior1, prior2, prior3) from the captured topo_data.

Each prior -> 6 nodes; each node -> 12 direction views (dir_00 = yaw 0) carrying the 4 raw PNG
bytes (main/sub x rgb/depth) exactly as captured, in order. Node placement / room type:

  node 1,2 -> room 'living_room'   pos = the SCAN's estimated (x, z)   (yaw ignored)
  node 3,4 -> room 'kitchen'       pos = the SCAN's estimated (x, z)
  node 5   -> room 'bedroom'       pos = (1.8, 1.8)   (fixed override)
  node 6   -> room 'bedroom'       pos = (0.2, 0.8)   (fixed override)

Saves  topo_data/<prior>_topology.pkl  (self-contained; the PNG bytes are embedded). Load with
    from planning.topology import Topology
    topo = Topology.load("topo_data/base_topology.pkl")

Run:  python build_topologies.py [--priors base]
"""
import glob
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from planning.topology import Topology, TopoNode, NodeView, IMAGE_KEYS

DATA_ROOT = "topo_data"
PRIORS = ["base", "prior1", "prior2", "prior3"]
NUM_DIRS = 12

ROOM_TYPE = {1: "living_room", 2: "living_room",
             3: "kitchen", 4: "kitchen",
             5: "bedroom", 6: "bedroom"}
# nodes 1-4 take the scan's estimated (x, z); 5 and 6 are placed by hand
FIXED_POS = {5: (1.8, 1.8), 6: (0.2, 0.8)}

# IMAGE_KEYS slot -> filename tag under dir_<ii>_deg<ddd>_<tag>.png
SLOT_TAG = {"main_rgb": "main_rgb", "main_depth": "main_depth",
            "sub_rgb": "sub_rgb", "sub_depth": "sub_depth"}


def _read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def _load_views(node_dir):
    """The 12 direction views for a node, each holding the 4 raw PNG bytes (dir_00 = yaw 0)."""
    views = []
    for i in range(NUM_DIRS):
        images = {}
        for slot, tag in SLOT_TAG.items():
            hits = sorted(glob.glob(os.path.join(node_dir, f"dir_{i:02d}_*_{tag}.png")))
            images[slot] = _read_bytes(hits[0]) if hits else None
        views.append(NodeView(index=i, images=images))
    return views


def _node_xz(prior, k):
    """(x, z) for node k: fixed override for 5/6, else the scan's estimated pose (x, z)."""
    if k in FIXED_POS:
        return FIXED_POS[k]
    man = json.load(open(os.path.join(DATA_ROOT, prior, f"node{k}", "scan.json")))
    fin = man.get("final") or (man["directions"][-1] if man.get("directions") else {})
    return float(fin["est_x"]), float(fin["est_z"])


def build_prior(prior):
    topo = Topology(name=prior, meta={"source": os.path.join(DATA_ROOT, prior)})
    for k in range(1, 7):
        node_dir = os.path.join(DATA_ROOT, prior, f"node{k}")
        x, z = _node_xz(prior, k)
        node = TopoNode(node_id=k, x=x, z=z, room_type=ROOM_TYPE[k],
                        views=_load_views(node_dir),
                        meta={"pos_source": "fixed" if k in FIXED_POS else "scan_estimate"})
        topo.add_node(node)
    return topo


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Build topo_data/<prior>_topology.pkl from captures.")
    ap.add_argument("--priors", default="base",
                    help="comma list of capture folders under topo_data/ (e.g. base or base,prior1)")
    args = ap.parse_args()
    for prior in [p.strip() for p in args.priors.split(",") if p.strip()]:
        topo = build_prior(prior)
        out = os.path.join(DATA_ROOT, f"{prior}_topology.pkl")
        topo.save(out)
        n_imgs = sum(1 for nd in topo for v in nd.views for k in IMAGE_KEYS if v.raw(k))
        size_mb = os.path.getsize(out) / 1e6
        print(f"[{prior}] {len(topo)} nodes, {n_imgs} images -> {out} ({size_mb:.0f} MB)")
        for nd in topo:
            print(f"    node{nd.id}: ({nd.x:6.2f},{nd.z:6.2f}) room={nd.room_type:12s} "
                  f"views={len(nd.views)} pos={nd.meta['pos_source']}")


if __name__ == "__main__":
    main()
