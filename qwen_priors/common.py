"""Shared IO/util for the object-prior pipeline: object list, topo/occupancy loading, and per-view
furniture / caption / image extraction. Every scoring script builds on these instead of repeating
the decode+furniture boilerplate."""
import gzip, io, os, pickle
import numpy as np
from PIL import Image
from planning.topology import Topology
from utils.sam3_prompt import prompt as OBJ_PROMPTS   # {key: natural-language}

OBJKEYS = list(OBJ_PROMPTS.keys())
NLLIST = [OBJ_PROMPTS[k] for k in OBJKEYS]
ROOMNODES = {1: "living", 2: "living", 3: "kitchen", 4: "kitchen", 5: "bedroom", 6: "bedroom"}

# Canonical home for the FINAL scoring artifacts (single place so producers + consumers agree).
# Intermediate signals live in subfolders; the ensembled prior is SCORE_DIR/scores.json.
SCORE_DIR = os.path.join("topo_data", "score")
FINAL_SCORES = os.path.join(SCORE_DIR, "scores.json")

# Final prior fusion.  The caption branch is a weaker backoff for a VLM miss, while room
# affinity is a soft prior that can re-rank views but can never erase view evidence.
CAPTION_BACKOFF = 0.5
ROOM_AFFINITY_FLOOR = 0.6


def combine_prior_scores(vlm_image, caption_llm, room_affinity):
    """Fuse the three 0..1 prior signals into one score.

    direction = max(VLM_image, 0.5 * caption_LLM)
    room_gate = 0.6 + 0.4 * room_affinity
    final = direction * room_gate

    The non-zero room floor is intentional: room affinity is useful for ordering otherwise
    similar views, but portable objects and deliberately unusual placements make it unsafe as
    a hard probability gate.  A live update that has no caption passes ``caption_llm=0``.
    """
    direction = max(float(vlm_image), CAPTION_BACKOFF * float(caption_llm))
    room_gate = ROOM_AFFINITY_FLOOR + (1.0 - ROOM_AFFINITY_FLOOR) * float(room_affinity)
    return direction * room_gate


def load_topo(prior="base", images=True):
    """Image-carrying topology topo_data/<prior>_topology.pkl (rebuilt from the captures, see README).
    images=False: steps that only need node positions, view indices and annotations fall back to the
    image-free runtime copy topo_data/topo_graph/<prior>_topology.pkl when the full file is absent."""
    path = os.path.join("topo_data", f"{prior}_topology.pkl")
    if not images and not os.path.exists(path):
        path = os.path.join("topo_data", "topo_graph", f"{prior}_topology.pkl")
    return Topology.load(path)


def load_occ(path="occupancy_map/base_occ_map.dat.gz"):
    d = pickle.load(gzip.open(path, "rb")); mp = d["map_params"]
    return {"ob": np.asarray(d["obstacle_map"]).astype(bool),
            "seen": np.asarray(d.get("seen_map")).astype(bool),
            "res": mp["resolution_m"], "ox": mp["origin_world_xz"][0], "oz": mp["origin_world_xz"][1]}


def furniture_of(view):
    """Sorted union of furniture detected in the view's main + sub cameras."""
    fur = view.meta.get("furniture", {})
    return sorted(set(fur.get("main", {})) | set(fur.get("sub", {}))) if isinstance(fur, dict) else []


def caption_of(view):
    return view.meta.get("caption", "")


def images_of(view, cams=("main_rgb", "sub_rgb"), max_px=None):
    """Decode a view's camera frames to PIL RGB (optionally thumbnailed to max_px)."""
    out = []
    for c in cams:
        b = view.raw(c)
        if b is None:
            continue
        im = Image.open(io.BytesIO(b)).convert("RGB")
        if max_px:
            im.thumbnail((max_px, max_px))
        out.append(im)
    return out


def sorted_views(node):
    return sorted(node.views, key=lambda v: v.index)


# --- score layout: OBJECT-FIRST on disk {obj: {node: {dir: score}}} so a goal's prior loads
# directly (scores[obj]); compute loops build node-first, so transpose at the save/load boundary.
def to_object_first(node_first):
    """{node:{dir:{obj:score}}} -> {obj:{node:{dir:score}}}."""
    out = {}
    for nid, dirs in node_first.items():
        for dd, objs in dirs.items():
            for obj, sc in objs.items():
                out.setdefault(obj, {}).setdefault(nid, {})[dd] = sc
    return out


def to_node_first(object_first):
    """{obj:{node:{dir:score}}} -> {node:{dir:{obj:score}}} (for the heatmap renderer)."""
    out = {}
    for obj, nodes in object_first.items():
        for nid, dirs in nodes.items():
            for dd, sc in dirs.items():
                out.setdefault(nid, {}).setdefault(dd, {})[obj] = sc
    return out
