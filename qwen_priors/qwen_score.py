"""Score topo nodes for a target object with a Qwen model -- shared VLM/LLM scoring used by
the CLI here and by bench.py. Reads the SAM3-annotated topo (node.meta['furniture'|
'objects']); no SAM3/detector needed.

  python qwen_score.py --mode llm --model Qwen/Qwen2.5-7B-Instruct    --target mug --prior prior2
  python qwen_score.py --mode vlm --model Qwen/Qwen2.5-VL-3B-Instruct --target mug --load 4bit
"""
import argparse, io, os, re, sys, time
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from planning.topology import Topology

SYS = ("You estimate where a target object is in a home, for a robot. Given a location's "
       "visible furniture / previously-seen objects (and maybe an image), rate how likely the "
       "TARGET is at that location. Homes are dynamic: past objects are a hint, not a guarantee; "
       "reason from furniture/room context (mug -> kitchen counter/sink; pillow -> bed). "
       "Reply EXACTLY one line: 'SCORE: <0-1> | REASON: <short clause>'.")


def montage(node, cams=("main_rgb",), cols=4, cell=(320, 240)):
    ims = []
    for v in sorted(node.views, key=lambda v: v.index):
        for c in cams:
            b = v.raw(c)
            if b is not None:
                ims.append(Image.open(io.BytesIO(b)).convert("RGB").resize(cell))
    rows = (len(ims) + cols - 1) // cols
    W, H = cell; canvas = Image.new("RGB", (cols * W, rows * H), (20, 20, 20))
    for k, im in enumerate(ims):
        r, c = divmod(k, cols); canvas.paste(im, (c * W, r * H))
    return canvas


def ctx_text(node):
    fur = node.meta.get("furniture", {}).get("all", [])
    obj = node.meta.get("objects", {}).get("all", [])
    return f"room_type={node.room_type}; visible_furniture={fur}; previously_seen_objects={obj}"


def parse_score(text):
    m = re.search(r"SCORE:\s*([01](?:\.\d+)?)", text)
    return (float(m.group(1)) if m else None), " ".join(text.split())[:200]


def load_model(model_id, mode, load="bf16"):
    kw = {"device_map": "cuda"}
    if load == "4bit":
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    else:
        kw["dtype"] = torch.bfloat16
    if mode == "vlm":
        from transformers import AutoProcessor, AutoModelForImageTextToText
        proc = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForImageTextToText.from_pretrained(model_id, **kw)
    else:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        proc = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    model.eval()
    return model, proc


def _apply_template(proc, msgs):
    try:                                   # Qwen3 thinking models: disable to keep output short
        return proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                        enable_thinking=False)
    except TypeError:
        return proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def score_node(model, proc, mode, target, node, max_new_tokens=96):
    user = (f"TARGET: {target}\nLOCATION (node {node.id}): {ctx_text(node)}\n"
            "How likely is the target at THIS location?")
    if mode == "vlm":
        from qwen_vl_utils import process_vision_info
        msgs = [{"role": "system", "content": SYS},
                {"role": "user", "content": [{"type": "image", "image": montage(node)},
                                             {"type": "text", "text": user}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        imgs, vids = process_vision_info(msgs)
        inputs = proc(text=[text], images=imgs, return_tensors="pt").to(model.device)
    else:
        msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": user}]
        inputs = proc([_apply_template(proc, msgs)], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen = out[:, inputs["input_ids"].shape[1]:]
    txt = (proc.batch_decode(gen, skip_special_tokens=True)[0] if mode == "vlm"
           else proc.decode(gen[0], skip_special_tokens=True))
    return parse_score(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["llm", "vlm"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--prior", default="prior2")
    ap.add_argument("--load", choices=["bf16", "4bit"], default="bf16")
    args = ap.parse_args()
    t0 = time.perf_counter()
    model, proc = load_model(args.model, args.mode, args.load)
    print(f">>> loaded {args.model} ({args.load}) in {time.perf_counter()-t0:.1f}s | "
          f"GPU {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)
    topo = Topology.load(os.path.join("topo_data", f"{args.prior}_topology.pkl"))
    res = [(n.id,) + score_node(model, proc, args.mode, args.target, n) for n in topo.nodes]
    print(f"\n=== {args.prior} | target='{args.target}' | {args.mode} {args.model} ({args.load}) ===")
    for nid, sc, reason in sorted(res, key=lambda r: -(r[1] or -1)):
        print(f"  node {nid}: score={sc}  | {reason}", flush=True)


if __name__ == "__main__":
    main()
