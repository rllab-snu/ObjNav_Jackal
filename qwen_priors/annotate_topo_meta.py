"""Annotate each topo_data prior's node views with the OBJECTS and FURNITURE that SAM 3
detects in every direction's main + sub image, under one consistent meta scheme.

Both detectors share the per-image SAM 3 encoding (set_image once, then all prompts), so a
single pass over topo_data/prior{1,2,3}_topology.pkl fills, per prior:

  view.meta["objects"]   = {"main": {cat: {pixels, score, instances}}, "sub": {...}}
  view.meta["furniture"] = {"main": {cat: {pixels, score, fill, instances}}, "sub": {...}}
  node.meta["objects"]   = {"all":[cats], "main":[cats], "sub":[cats],
                            "by_camera": {"main": {cat: best_info}, "sub": {...}}}
  node.meta["furniture"] = same shape as node objects
  topo.meta["detection_meta"] = {"objects": {prompts, min_pixels, confidence},
                                 "furniture": {prompts, min_pixels, confidence, iou_nms, min_fill}}

`pixels` is the SAM 3 instance MASK area (count of mask pixels after >0.5), NOT the bbox area;
`fill` = pixels / bbox_area (used to drop thin/sparse furniture like a bare TV floor-stand).

OBJECTS = small placed items (utils.sam3_prompt); FURNITURE = large context items an object
sits on/near (closed storage -- cabinet/shelf/wardrobe -- excluded). Furniture adds a
fill-ratio filter + cross-category NMS (so a tower AC is 'air_conditioner', not 'refrigerator').

Run in the sam3_complex env:
  conda run -n sam3_complex env PYTHONPATH=<repo>:third_party/sam3 python annotate_topo_meta.py
"""
import io
import os
import sys

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from planning.topology import Topology
from utils.sam3_prompt import prompt as OBJECTS

FURNITURE = {
    "sofa": "sofa", "coffee_table": "coffee table", "tv": "television",
    "tv_cabinet": "tv cabinet", "side_table": "small side table",
    "kitchen_counter": "kitchen countertop", "sink": "sink",
    "dining_table": "dining table", "dresser": "dressing table",
    "bed": "bed", "chair": "chair", "refrigerator": "refrigerator",
    "air_conditioner": "floor standing air conditioner",
}
BPE = "third_party/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
DEVICE = "cuda"
CONFIDENCE = 0.5
OBJ_MIN_PIX = 50
FUR_MIN_PIX = 400
FUR_IOU_NMS = 0.6
FUR_MIN_FILL = 0.35
PRIORS = ["prior1", "prior2", "prior3"]
CAMS = [("main", "main_rgb"), ("sub", "sub_rgb")]
# stale keys from earlier annotation passes, removed so only the unified scheme remains
OLD_VIEW_KEYS = ("sam3", "furniture_sam3")
OLD_NODE_KEYS = ("objects_detail", "furniture_detail")
OLD_TOPO_KEYS = ("sam3_scan", "furniture_scan")


def _iou(a, b):
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1]); x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--priors", default=",".join(PRIORS),
                    help="comma list of priors to annotate (e.g. 'base' or 'prior1,prior2')")
    ap.add_argument("--no-objects", action="store_true",
                    help="skip object detection (base scene has no placed objects -> objects empty)")
    args = ap.parse_args()
    priors = [p.strip() for p in args.priors.split(",") if p.strip()]
    do_objects = not args.no_objects
    os.chdir(PROJECT_ROOT)
    print(f">>> loading SAM 3 on {DEVICE} ... (priors={priors}, objects={do_objects})", flush=True)
    model = build_sam3_image_model(BPE, device=DEVICE)
    proc = Sam3Processor(model, device=DEVICE, confidence_threshold=CONFIDENCE)
    amp = torch.autocast("cuda", dtype=torch.bfloat16)
    obj_cats = list(OBJECTS.items())
    fur_cats = list(FURNITURE.items())
    print(f">>> {len(obj_cats)} object + {len(fur_cats)} furniture prompts", flush=True)

    def _instances(out):
        """(pixels[], scores[], boxes[]) for the current prompt's masks, or empty."""
        m = out["masks"]
        if not (m.numel() and m.shape[0] > 0):
            return None
        inst = (m.squeeze(1) > 0.5).reshape(m.shape[0], -1)
        px = inst.sum(1).float().cpu().numpy()
        bx = out["boxes"].float().cpu().numpy()
        sc = out["scores"].float().cpu().numpy().reshape(-1)
        return px, sc, bx

    def annotate(rgb):
        """-> (objects{cat:info}, furniture{cat:info}) for one RGB frame; encodes once."""
        img = Image.fromarray(np.ascontiguousarray(rgb[..., :3]).astype(np.uint8))
        objs, fur_raw = {}, []
        with torch.no_grad(), amp:
            st = proc.set_image(img)
            # small placed objects: pixel-gated, best instance per category (skipped for base)
            for cat, nlp in (obj_cats if do_objects else []):
                r = _instances(proc.set_text_prompt(state=st, prompt=nlp))
                if r is None:
                    continue
                px, sc, _ = r
                keep = px >= OBJ_MIN_PIX
                if keep.any():
                    bi = int(np.argmax(np.where(keep, px, -1)))
                    objs[cat] = {"pixels": int(px[bi]),
                                 "score": float(sc[bi]) if bi < len(sc) else None,
                                 "instances": int(keep.sum())}
            # furniture: pixel + fill-ratio gate, then cross-category NMS
            for cat, nlp in fur_cats:
                r = _instances(proc.set_text_prompt(state=st, prompt=nlp))
                if r is None:
                    continue
                px, sc, bx = r
                for j in range(len(px)):
                    box = [float(t) for t in bx[j]]
                    area = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
                    fill = float(px[j]) / area
                    if px[j] >= FUR_MIN_PIX and fill >= FUR_MIN_FILL:
                        fur_raw.append((cat, float(sc[j]) if j < len(sc) else 0.0,
                                        int(px[j]), fill, box))
        fur_raw.sort(key=lambda d: -d[1])
        kept = []
        for d in fur_raw:
            if all(_iou(d[4], k[4]) < FUR_IOU_NMS or d[0] == k[0] for k in kept):
                kept.append(d)
        fur = {}
        for cat, sc, px, fill, _ in kept:
            prev = fur.get(cat)
            if prev is None or px > prev["pixels"]:
                fur[cat] = {"pixels": px, "score": sc, "fill": round(fill, 3),
                            "instances": sum(1 for k in kept if k[0] == cat)}
        return objs, fur

    def aggregate(per_cam):
        """per_cam = {"main": {cat: best_info}, "sub": {...}} -> node-level summary dict."""
        return {"all": sorted(set(per_cam["main"]) | set(per_cam["sub"])),
                "main": sorted(per_cam["main"]), "sub": sorted(per_cam["sub"]),
                "by_camera": per_cam}

    for prior in priors:
        path = os.path.join("topo_data", f"{prior}_topology.pkl")
        topo = Topology.load(path)
        for k in OLD_TOPO_KEYS:
            topo.meta.pop(k, None)
        print(f"\n===== {prior}: {len(topo)} nodes =====", flush=True)
        for node in topo.nodes:
            for k in OLD_NODE_KEYS:
                node.meta.pop(k, None)
            obj_cam = {"main": {}, "sub": {}}
            fur_cam = {"main": {}, "sub": {}}
            for view in node.views:
                for k in OLD_VIEW_KEYS:
                    view.meta.pop(k, None)
                vobj, vfur = {}, {}
                for cam, key in CAMS:
                    b = view.raw(key)
                    if b is None:
                        continue
                    rgb = np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))
                    o, f = annotate(rgb)
                    if o:
                        vobj[cam] = o
                        for c, info in o.items():
                            if c not in obj_cam[cam] or info["pixels"] > obj_cam[cam][c]["pixels"]:
                                obj_cam[cam][c] = info
                    if f:
                        vfur[cam] = f
                        for c, info in f.items():
                            if c not in fur_cam[cam] or info["pixels"] > fur_cam[cam][c]["pixels"]:
                                fur_cam[cam][c] = info
                view.meta["objects"] = vobj
                view.meta["furniture"] = vfur
            node.meta["objects"] = aggregate(obj_cam)
            node.meta["furniture"] = aggregate(fur_cam)
            print(f"  node {node.id}: objects={node.meta['objects']['all']}", flush=True)
            print(f"           furniture={node.meta['furniture']['all']}", flush=True)
        topo.meta["detection_meta"] = {
            "objects": {"prompts": dict(OBJECTS), "min_pixels": OBJ_MIN_PIX,
                        "confidence": CONFIDENCE},
            "furniture": {"prompts": dict(FURNITURE), "min_pixels": FUR_MIN_PIX,
                          "confidence": CONFIDENCE, "iou_nms": FUR_IOU_NMS,
                          "min_fill": FUR_MIN_FILL},
        }
        tmp = path + ".tmp"
        topo.save(tmp)
        os.replace(tmp, path)
        print(f">>> updated {path}", flush=True)
    print("\n>>> DONE", flush=True)


if __name__ == "__main__":
    main()
