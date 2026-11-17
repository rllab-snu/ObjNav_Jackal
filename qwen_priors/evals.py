"""Metrics used to judge good/bad across prior-scoring versions, and a table printer to compare
scores.json files. All evaluated on the object-free base scene with a known room layout.

  python qwen_priors/evals.py topo_data/score topo_data/score_old ...
"""
import json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
from common import OBJKEYS, ROOMNODES, load_topo

# clear-affinity diagnostic objects -> expected room
AFFIN = {"mug": "kitchen", "bowl": "kitchen", "coffee_machine": "kitchen", "lemon": "kitchen",
         "orange": "kitchen", "pringles": "kitchen", "spoon": "kitchen", "sponge": "kitchen",
         "toaster": "kitchen", "clock": "bedroom", "comb": "bedroom", "dryer": "bedroom",
         "remote_controller": "living"}


def empty_views(topo):
    """(node,dir) views whose furniture list is empty (wall/curtain/floor)."""
    from common import furniture_of
    out = set()
    for n in topo.nodes:
        for v in n.views:
            if not furniture_of(v):
                out.add((str(n.id), str(v.index)))
    return out


def metrics(scores, empties):
    # object-first: scores[obj][node][dir] = score
    def g(obj, n, k):
        return scores.get(obj, {}).get(str(n), {}).get(str(k), 0)
    objs = OBJKEYS
    empty = np.mean([g(o, n, d) for (n, d) in empties for o in objs]) if empties else 0.0
    ct = []
    for obj, room in AFFIN.items():
        cor = [g(obj, n, k) for n in ROOMNODES for k in range(12) if ROOMNODES[n] == room]
        oth = [g(obj, n, k) for n in ROOMNODES for k in range(12) if ROOMNODES[n] != room]
        ct.append(np.mean(cor) - np.mean(oth))
    d5 = [g("dryer", 5, k) for k in range(12)]
    dryer = np.mean([d5[5], d5[6], d5[7]]) - np.mean([d5[k] for k in range(12) if k not in (5, 6, 7)])
    hit = []
    for obj, room in AFFIN.items():
        cells = [(n, k, g(obj, n, k)) for n in ROOMNODES for k in range(12)]
        top = sorted(cells, key=lambda x: -x[2])[:3]
        hit.append(np.mean([ROOMNODES[n] == room for n, k, _ in top]))
    wr = [np.std([g(o, n, k) for k in range(12)]) for n in ROOMNODES for o in objs]
    distinct = len(set(round(g(o, n, k), 4) for o in objs for n in ROOMNODES for k in range(12)))
    return dict(empty=empty, room_ctr=np.mean(ct), dryer=dryer, hit=np.mean(hit),
                dir_res=np.mean(wr), distinct=distinct)


def compare(paths):
    topo = load_topo("base", images=False); empties = empty_views(topo)
    hdr = f"{'config':30s} {'empty↓':>7} {'room_ctr↑':>9} {'dryer↑':>7} {'hit↑':>6} {'dir_res↑':>8} {'distinct':>8}"
    print(hdr); print("-" * len(hdr))
    for p in paths:
        sp = p if p.endswith(".json") else os.path.join(p, "scores.json")
        m = metrics(json.load(open(sp)), empties)
        name = os.path.basename(p.rstrip("/")).replace("scores.json", "")
        print(f"{name:30s} {m['empty']:7.3f} {m['room_ctr']:+9.2f} {m['dryer']:+7.2f} "
              f"{m['hit']*100:5.0f}% {m['dir_res']:8.3f} {m['distinct']:8d}")


if __name__ == "__main__":
    args = sys.argv[1:] or ["outputs/priors/ensemble", "outputs/priors/twostage"]
    compare(args)
