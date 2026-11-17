"""Add a per-view CAPTION to the topo: a VLM (Qwen3-VL-8B-4bit) describes, in one sentence, what
surfaces/furniture are visible in each direction -- or says explicitly when only a bare wall / curtain
/ floor is visible. This closes the ambiguity of an empty furniture list (missing != "nothing here")
and recovers surfaces SAM 3 missed. Stored as view.meta["caption"], computed once, reused by scoring.
  PYTHONPATH=.:third_party/sam3 python annotate_captions.py --priors base
"""
import argparse, io, os, sys, time
import torch
from PIL import Image
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
from planning.topology import Topology
from qwen_score import load_model

SYS = ("Describe ONLY what a robot camera sees in this view, in ONE short sentence, for deciding where "
       "objects could be placed. Name the visible furniture/surfaces (sofa, small coffee table, kitchen "
       "counter, dresser, bed, tv unit...). If only a bare wall, curtain, window, door or empty floor is "
       "visible with no usable surface, say exactly that. Do not guess objects that are not visible.")


def caption(model, proc, imgs):
    from qwen_vl_utils import process_vision_info
    content = [{"type": "image", "image": im} for im in imgs] + [{"type": "text", "text": "Describe this view:"}]
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": content}]
    t = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    im, _ = process_vision_info(msgs)
    inp = proc(text=[t], images=im, return_tensors="pt").to(model.device)
    with torch.no_grad():
        o = model.generate(**inp, max_new_tokens=48, do_sample=False)
    return proc.batch_decode(o[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--priors", default="base")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--load", default="4bit")
    args = ap.parse_args()
    model, proc = load_model(args.model, "vlm", args.load)
    print(f">>> caption model loaded | GPU {torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)
    for prior in [p.strip() for p in args.priors.split(",") if p.strip()]:
        path = f"topo_data/{prior}_topology.pkl"
        topo = Topology.load(path)
        n = 0; t0 = time.perf_counter()
        for node in topo.nodes:
            for v in sorted(node.views, key=lambda v: v.index):
                imgs = [Image.open(io.BytesIO(v.raw(c))).convert("RGB")
                        for c in ("main_rgb", "sub_rgb") if v.raw(c)]
                v.meta["caption"] = caption(model, proc, imgs) if imgs else ""
                n += 1
            print(f"  node {node.id}: " + " | ".join(
                f"d{v.index}:{v.meta['caption'][:42]}" for v in sorted(node.views, key=lambda v: v.index)[:2]), flush=True)
        tmp = path + ".tmp"; topo.save(tmp); os.replace(tmp, path)
        print(f">>> {prior}: {n} captions saved -> {path} ({(time.perf_counter()-t0)/n:.1f}s/view)", flush=True)
    print(">>> DONE", flush=True)


if __name__ == "__main__":
    main()
