"""Render a saved Topology's nodes on the occupancy map (top-down).

Each node is a dot placed at its world (x, z), coloured by room type and labelled with its id;
a room-type legend and a faint 1 m grid are drawn for reference. (Edges are not implemented, so
none are drawn.) Works from a topology .pkl produced by build_topologies.py.

  # one topology
  python render_topology.py --topology topo_data/base_topology.pkl --out outputs/topo_base.png
  # all *_topology.pkl in topo_data/ -> outputs/topology_<name>.png
  python render_topology.py
"""
import argparse
import glob
import gzip
import os
import pickle
import sys

import numpy as np
import cv2

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from navigation.global_occupancy_map import GlobalOccupancyMap
from planning.topology import Topology
import utils.panels as P

ROOM_COLOR = {"living_room": (0, 220, 0), "kitchen": (255, 170, 0), "bedroom": (60, 120, 255)}
DEFAULT_COLOR = (200, 200, 200)
# per-node montage placement direction override (node_id -> (dx, dy); -y = up). Nodes not
# listed use the radial "away from map centre" direction. Node 3's callout reads better up-left.
DIR_HINT = {3: (-1.0, -1.3)}
# CANONICAL node world (x, z) used ONLY to lay out the montage boxes, so a node's montage lands
# in the SAME place in every prior (n1-4 vary slightly per scan; n5/n6 are fixed). The node dots
# and leader lines still follow each prior's own node positions.
CANON_POS = {1: (1.85, 4.85), 2: (-0.7, 5.5), 3: (-2.5, 2.85),
             4: (-5.4, 3.2), 5: (1.8, 1.8), 6: (0.2, 0.8)}


def load_occ_map(map_path, agent_radius_m=0.18):
    with gzip.open(map_path, "rb") as f:
        d = pickle.load(f)
    return GlobalOccupancyMap.from_occupancy_dict(d, agent_radius_m=agent_radius_m)


def _node_block(node, color, cols=6, rows=2, tw=84, th=63):
    """A titled `rows` x `cols` montage of the node's 12 direction images (main RGB, dir_00 =
    yaw 0). Each thumbnail is tagged with its direction index (0..11)."""
    thumbs = []
    for i in range(rows * cols):
        v = node.view(i)
        im = v.image("main_rgb") if v is not None else None
        im = (np.zeros((th, tw, 3), np.uint8) if im is None
              else cv2.resize(np.ascontiguousarray(im), (tw, th)))
        cv2.putText(im, str(i), (2, 12), cv2.FONT_HERSHEY_DUPLEX, 0.34, (60, 255, 255), 1, cv2.LINE_AA)
        thumbs.append(im)
    grid = np.concatenate([np.concatenate(thumbs[r * cols:(r + 1) * cols], axis=1)
                           for r in range(rows)], axis=0)
    band = 18
    bar = np.zeros((band, grid.shape[1], 3), np.uint8)
    cv2.putText(bar, f"node{node.id}  {node.room_type}  ({node.x:.1f},{node.z:.1f})",
                (4, band - 5), cv2.FONT_HERSHEY_DUPLEX, 0.42, color, 1, cv2.LINE_AA)
    return np.concatenate([bar, grid], axis=0)


def render_topology(topo, gmap, out_h=800, thumb_w=104, thumb_h=78, gap=14, pad=20, node_clear=20):
    """Occ map with each node's 12-direction montage (2x6 main RGB) placed as a CALLOUT next to
    the node (just outward of it; it may cover a little of the MAP -- that's fine). Montages
    NEVER overlap EACH OTHER: they are relaxed apart with a `gap`, and the canvas is grown to the
    bounding box of the map + all montages (no clamping) so there is no wasted margin. A leader
    line links each node to its montage. No coordinate grid. Returns an RGB image."""
    nodes = list(topo)
    img, scale = P.base_map(gmap, out_h)
    img = np.ascontiguousarray(img)
    map_h, map_w = img.shape[:2]
    cxm, cym = map_w / 2.0, map_h / 2.0                       # map coords; offset into canvas later
    ids = [nd.id for nd in nodes]

    color_of = {nd.id: ROOM_COLOR.get(nd.room_type, DEFAULT_COLOR) for nd in nodes}
    mont = {nd.id: _node_block(nd, color_of[nd.id], tw=thumb_w, th=thumb_h) for nd in nodes}
    # LAYOUT uses CANONICAL node pixels (fixed) so montage boxes are IDENTICAL in every prior;
    # dots + leaders below use each prior's ACTUAL node pixels.
    lay = {nid: P.px(gmap, CANON_POS.get(nid, (0.0, 0.0))[0], CANON_POS.get(nid, (0.0, 0.0))[1], scale)
           for nid in ids}
    node_xy = {nd.id: P.px(gmap, nd.x, nd.z, scale) for nd in nodes}

    # montage centre = just outward of its (canonical) node; radial unless a DIR_HINT is set
    mcx, mcy = {}, {}
    for nid in ids:
        nx, ny = lay[nid]
        mh, mw = mont[nid].shape[:2]
        dx, dy = DIR_HINT.get(nid, (nx - cxm, ny - cym))
        n = max(1e-6, (dx * dx + dy * dy) ** 0.5)
        ux, uy = dx / n, dy / n
        p = abs(ux) * (mw / 2) + abs(uy) * (mh / 2) + 14
        mcx[nid] = nx + ux * p
        mcy[nid] = ny + uy * p

    # relax so NO two montages overlap, and none covers another node's (canonical) marker
    for _ in range(6000):
        moved = False
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                i, j = ids[a], ids[b]
                mhi, mwi = mont[i].shape[:2]
                mhj, mwj = mont[j].shape[:2]
                ox = (mwi + mwj) / 2 + gap - abs(mcx[i] - mcx[j])
                oy = (mhi + mhj) / 2 + gap - abs(mcy[i] - mcy[j])
                if ox > 0 and oy > 0:
                    if ox <= oy:
                        s = ox / 2
                        if mcx[i] <= mcx[j]: mcx[i] -= s; mcx[j] += s
                        else: mcx[i] += s; mcx[j] -= s
                    else:
                        s = oy / 2
                        if mcy[i] <= mcy[j]: mcy[i] -= s; mcy[j] += s
                        else: mcy[i] += s; mcy[j] -= s
                    moved = True
        for i in ids:
            mhi, mwi = mont[i].shape[:2]
            for k in ids:
                if k == i:
                    continue
                nkx, nky = lay[k]
                ax = (mwi / 2 + node_clear) - abs(mcx[i] - nkx)
                ay = (mhi / 2 + node_clear) - abs(mcy[i] - nky)
                if ax > 0 and ay > 0:
                    if ax <= ay:
                        mcx[i] += (-ax if mcx[i] <= nkx else ax)
                    else:
                        mcy[i] += (-ay if mcy[i] <= nky else ay)
                    moved = True
        if not moved:
            break

    # montage boxes (top-left, map coords) -> bounding box of map + montages -> tight canvas
    boxes = {nid: (mcx[nid] - mont[nid].shape[1] / 2, mcy[nid] - mont[nid].shape[0] / 2,
                   mont[nid].shape[1], mont[nid].shape[0]) for nid in ids}
    bx_all = list(boxes.values())
    minx = min([0.0] + [b[0] for b in bx_all]) - pad
    miny = min([0.0] + [b[1] for b in bx_all]) - pad
    maxx = max([float(map_w)] + [b[0] + b[2] for b in bx_all]) + pad
    maxy = max([float(map_h)] + [b[1] + b[3] for b in bx_all]) + pad
    ox, oy = -minx, -miny
    Wc, Hc = int(round(maxx - minx)), int(round(maxy - miny))
    canvas = np.full((Hc, Wc, 3), 25, np.uint8)
    mx0, my0 = int(round(ox)), int(round(oy))
    canvas[my0:my0 + map_h, mx0:mx0 + map_w] = img

    def O(x, y):
        return int(round(x + ox)), int(round(y + oy))

    # leader lines (actual node -> fixed montage box), then montages, then node dots
    for nd in nodes:
        bx, by, mw, mh = boxes[nd.id]
        cv2.line(canvas, O(*node_xy[nd.id]), O(bx + mw / 2, by + mh / 2),
                 color_of[nd.id], 1, cv2.LINE_AA)
    for nid in ids:
        mb = mont[nid]
        x0, y0 = int(round(boxes[nid][0] + ox)), int(round(boxes[nid][1] + oy))
        x1, y1 = min(Wc, x0 + mb.shape[1]), min(Hc, y0 + mb.shape[0])
        x0, y0 = max(0, x0), max(0, y0)
        canvas[y0:y1, x0:x1] = mb[:y1 - y0, :x1 - x0]
    for nd in nodes:
        px_, py_ = O(*node_xy[nd.id])
        cv2.circle(canvas, (px_, py_), 8, color_of[nd.id], -1, cv2.LINE_AA)
        cv2.circle(canvas, (px_, py_), 8, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(canvas, str(nd.id), (px_ - 4, py_ + 4),
                    cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.putText(canvas, f"topology: {topo.name}  ({len(nodes)} nodes)", (10, 24),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)
    for i, (room, c) in enumerate(ROOM_COLOR.items()):
        yb = 46 + i * 20
        cv2.circle(canvas, (18, yb - 4), 6, c, -1, cv2.LINE_AA)
        cv2.putText(canvas, room, (30, yb), cv2.FONT_HERSHEY_DUPLEX, 0.45, c, 1, cv2.LINE_AA)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topology", default=None,
                    help="a topology .pkl; omit to render every topo_data/*_topology.pkl")
    ap.add_argument("--map", default="occupancy_map/base_occ_map.dat.gz")
    ap.add_argument("--out", default=None, help="output png (single-topology mode)")
    ap.add_argument("--out_h", type=int, default=800)
    args = ap.parse_args()

    gmap = load_occ_map(args.map)
    targets = ([args.topology] if args.topology
               else sorted(glob.glob("topo_data/*_topology.pkl")))
    os.makedirs("outputs", exist_ok=True)
    for pkl in targets:
        topo = Topology.load(pkl)
        img = render_topology(topo, gmap, out_h=args.out_h)
        out = (args.out if args.topology and args.out
               else os.path.join("outputs", f"topology_{topo.name}.png"))
        cv2.imwrite(out, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"[{topo.name}] {len(topo)} nodes -> {out}")


if __name__ == "__main__":
    main()
