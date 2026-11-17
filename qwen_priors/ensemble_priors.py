"""FINAL prior (recommended):
  direction = max(VLM_image_score, 0.5 * caption_LLM_score)
  room_gate = 0.6 + 0.4 * room_affinity_LLM
  final(node,dir,obj) = direction * room_gate
Reads the three precomputed signals (gen_direction.py --signal vlm / --signal caption), renders 25
heatmaps, and prints metrics vs the previous hard-gated ensemble.
  PYTHONPATH=.:third_party/sam3 python qwen_priors/ensemble_priors.py
Outputs topo_data/score/scores.json and topo_data/score/renders/<object>.png. Run in sam3_complex.
"""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
from common import (OBJKEYS, SCORE_DIR, FINAL_SCORES, combine_prior_scores,
                    load_topo, load_occ, sorted_views)
from viz import render
from evals import compare


def main():
    topo = load_topo("base", images=False); occ = load_occ()
    vlm = json.load(open(os.path.join(SCORE_DIR, "Qwen3-VL-8B_dir", "scores.json")))
    cap = json.load(open(os.path.join(SCORE_DIR, "caption_llm_dir", "scores.json")))
    aff = json.load(open(os.path.join(SCORE_DIR, "room_affinity.json")))

    os.makedirs(SCORE_DIR, exist_ok=True)
    scores = {}                                    # object-first {obj:{node:{dir:score}}}
    for ok in OBJKEYS:
        scores[ok] = {}
        for node in topo.nodes:
            nid = str(node.id); scores[ok][nid] = {}
            for v in sorted_views(node):
                dd = str(v.index)
                dv = float(vlm.get(ok, {}).get(nid, {}).get(dd, 0))
                dc = float(cap.get(ok, {}).get(nid, {}).get(dd, 0))
                ar = float(aff.get(f"{node.id}|{ok}", 0.0))
                scores[ok][nid][dd] = combine_prior_scores(dv, dc, ar)
    json.dump(scores, open(FINAL_SCORES, "w"))
    viz_dir = os.path.join(SCORE_DIR, "renders"); os.makedirs(viz_dir, exist_ok=True)
    for tgt in OBJKEYS:
        render(tgt, scores, topo, occ, os.path.join(viz_dir, f"{tgt}.png"))
    print(f">>> saved {FINAL_SCORES} + 25 heatmaps -> {viz_dir}/")
    # Metrics of the new prior; add the previous hard-gated ensemble side-by-side if present.
    paths = [SCORE_DIR]
    old = "topo_data/score_old"
    if os.path.exists(os.path.join(old, "scores.json")):
        paths.append(old)
    compare(paths)


if __name__ == "__main__":
    main()
