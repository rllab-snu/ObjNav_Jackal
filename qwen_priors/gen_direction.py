"""Generate a per-(node,dir,object) DIRECTION signal for the two-stage / ensemble prior, with the
sharpened per-object prompt. The current ensemble combines these signals with a VLM-led caption
backoff rather than averaging them:

  --signal vlm      Qwen3-VL-8B-4bit scores from the view IMAGES        -> topo_data/score/Qwen3-VL-8B_dir/
  --signal caption  Qwen3-4B LLM scores from the stored CAPTION (text)  -> topo_data/score/caption_llm_dir/
                    (also writes topo_data/score/room_affinity.json, the cross-node room signal)

  PYTHONPATH=.:third_party/sam3 python qwen_priors/gen_direction.py --signal vlm
  PYTHONPATH=.:third_party/sam3 python qwen_priors/gen_direction.py --signal caption
Run in sam3_complex (VLM: export PYTORCH_ALLOC_CONF=expandable_segments:True).
"""
import argparse, json, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
from common import (OBJKEYS, SCORE_DIR, load_topo, load_occ, furniture_of, images_of, caption_of,
                    sorted_views, to_object_first)
from utils.sam3_prompt import prompt as OBJ_PROMPTS
from qwen_score import load_model
from scorers import SharpenedScorer, RoomAffinityScorer
from viz import render

DEFAULTS = {"vlm": ("Qwen/Qwen3-VL-8B-Instruct", "4bit", "Qwen3-VL-8B_dir"),
            "caption": ("Qwen/Qwen3-4B-Instruct-2507", "bf16", "caption_llm_dir")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal", choices=["vlm", "caption"], required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--load", default=None)
    ap.add_argument("--prior", default="base")
    ap.add_argument("--render", action="store_true", help="also render 25 heatmaps of this signal")
    args = ap.parse_args()
    dmodel, dload, dfolder = DEFAULTS[args.signal]
    model_id, load = args.model or dmodel, args.load or dload
    mode = "vlm" if args.signal == "vlm" else "llm"
    outdir = os.path.join(SCORE_DIR, dfolder); os.makedirs(outdir, exist_ok=True)
    topo = load_topo(args.prior)

    model, proc = load_model(model_id, mode, load)
    sc = SharpenedScorer(model, proc, mode)
    print(f">>> {dfolder}: {model_id} [{mode}/{load}]", flush=True)

    if args.signal == "caption":                       # room-level affinity gate (cross-node)
        aff = RoomAffinityScorer(model, proc, "llm"); affinity = {}; t0 = time.perf_counter(); n = 0
        for node in topo.nodes:
            fur_all = node.meta.get("furniture", {}).get("all", [])
            for ok in OBJKEYS:
                affinity[f"{node.id}|{ok}"] = aff.score(OBJ_PROMPTS[ok], node.room_type, fur_all); n += 1
        json.dump(affinity, open(os.path.join(SCORE_DIR, "room_affinity.json"), "w"))
        print(f">>> room_affinity {n} calls, {(time.perf_counter()-t0)/n:.2f}s/call", flush=True)

    scores = {str(n.id): {} for n in topo.nodes}; t1 = time.perf_counter(); m = 0
    for node in topo.nodes:
        for v in sorted_views(node):
            fur = furniture_of(v)
            imgs = images_of(v) if args.signal == "vlm" else None
            cap = caption_of(v) if args.signal == "caption" else None
            scores[str(node.id)][str(v.index)] = {
                ok: sc.score(OBJ_PROMPTS[ok], node.room_type, fur, images=imgs, caption=cap)
                for ok in OBJKEYS}
            m += 1
        print(f"  node {node.id} done ({m} dirs, {(time.perf_counter()-t1)/m:.1f}s/dir)", flush=True)
    scores = to_object_first(scores)                       # {obj:{node:{dir:score}}}
    json.dump(scores, open(f"{outdir}/scores.json", "w"))
    print(f">>> saved {outdir}/scores.json", flush=True)
    if args.render:
        occ = load_occ()
        for tgt in OBJKEYS:
            render(tgt, scores, topo, occ, f"{outdir}/{tgt}.png")
        print(">>> rendered 25", flush=True)
    print(">>> DONE", flush=True)


if __name__ == "__main__":
    main()
