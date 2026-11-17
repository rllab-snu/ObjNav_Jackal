"""The scoring 'versions' tried, as one Scorer class hierarchy (shared generate/parse skeleton;
each subclass differs only in prompt + parsing). All take a loaded (model, proc) from
qwen_score.load_model and a mode in {'vlm','llm'}.

  Scorer                     base: _gen(system,user,images) + parse_score
   +- BatchedScorer          v1  25 objects -> one JSON            (predict_object_priors, no --cot)
   +- CoTScorer              v2  per-object reason -> SCORE        (predict_object_priors, --cot)
   +- AnchoredRankScorer     v4  12 dirs -> one call, per-line score (rank_object_priors)
   +- SharpenedScorer        v5/6 per-object tiered, wall->0        (gen_direction --signal vlm/caption)
   +- RoomAffinityScorer     room-level affinity signal             (two_stage / gen_direction caption)
"""
import json, re
import torch
import prompts
from common import OBJKEYS, NLLIST
from utils.sam3_prompt import prompt as OBJ_PROMPTS


class Scorer:
    def __init__(self, model, proc, mode):
        self.model, self.proc, self.mode = model, proc, mode

    def _gen(self, system, user, images=None, max_new_tokens=64):
        proc = self.proc
        if self.mode == "vlm":
            from qwen_vl_utils import process_vision_info
            content = [{"type": "image", "image": im} for im in (images or [])]
            content.append({"type": "text", "text": user})
            msgs = [{"role": "system", "content": system}, {"role": "user", "content": content}]
            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            imgs, _ = process_vision_info(msgs)
            inp = proc(text=[text], images=imgs, return_tensors="pt").to(self.model.device)
        else:
            msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            try:
                text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                enable_thinking=False)
            except TypeError:
                text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inp = proc([text], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
        gen = out[:, inp["input_ids"].shape[1]:]
        return (proc.batch_decode(gen, skip_special_tokens=True)[0] if self.mode == "vlm"
                else proc.decode(gen[0], skip_special_tokens=True))

    @staticmethod
    def parse_score(txt):
        m = re.search(r"SCORE:\s*([01](?:\.\d+)?)", txt)
        return float(m.group(1)) if m else 0.0


class CoTScorer(Scorer):
    """v2 per-object CoT: VLM uses main+sub images; LLM uses furniture/room text."""
    def score(self, obj_nl, room, furniture, images=None):
        if self.mode == "vlm":
            user = (f"Context (from an object-free scan of this spot): room_type={room}; "
                    f"detected_furniture={furniture}.\nObject: {obj_nl}")
            txt = self._gen(prompts.COT, user, images=images, max_new_tokens=128)
        else:
            user = f"visible_furniture: {furniture}; room_type: {room}\nObject: {obj_nl}"
            txt = self._gen(prompts.COT.replace("the photo(s) of ", ""), user, max_new_tokens=128)
        return self.parse_score(txt)


class BatchedScorer(Scorer):
    """v1 batched JSON: rate every object in one call (kept for the record; it goes lazy)."""
    def score_all(self, room, furniture, image=None):
        user = (f"room_type: {room}\nvisible_furniture: {furniture}\n"
                f"Items: {NLLIST}\nJSON scores (0-1) for every item:")
        imgs = [image] if (self.mode == "vlm" and image is not None) else None
        txt = self._gen(prompts.BATCHED, user, images=imgs, max_new_tokens=400)
        return self._parse_batch(txt)

    @staticmethod
    def _parse_batch(text):
        out = {}
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                for k, v in json.loads(m.group(0)).items():
                    out[k] = float(v)
            except Exception:
                pass
        if not out:
            for nl in NLLIST:
                mm = re.search(re.escape(nl) + r"\"?\s*[:=]\s*([01](?:\.\d+)?)", text)
                if mm:
                    out[nl] = float(mm.group(1))
        can = {}
        for k in OBJKEYS:
            for cand in (k, OBJ_PROMPTS[k]):
                if cand in out:
                    can[k] = max(0.0, min(1.0, out[cand])); break
            else:
                can[k] = 0.0
        return can


class SharpenedScorer(Scorer):
    """v5/v6 sharpened per-object (specific furniture->0.8-1.0, generic->0.3-0.5, wall/empty->0).
    VLM scores from images+furniture; LLM scores from the stored caption+furniture."""
    def score(self, obj_nl, room, furniture, images=None, caption=None):
        if self.mode == "vlm":
            user = f"Context: room_type={room}; detected_furniture={furniture}.\nObject: {obj_nl}"
            txt = self._gen(prompts.SHARP_VLM, user, images=images, max_new_tokens=64)
        else:
            user = f"Visible: {caption}\nfurniture={furniture}; room={room}\nObject: {obj_nl}"
            txt = self._gen(prompts.SHARP_LLM, user, max_new_tokens=24)
        return self.parse_score(txt)


class RoomAffinityScorer(Scorer):
    """LLM room-level affinity signal (hard gate in two-stage; soft prior in current ensemble)."""
    def score(self, obj_nl, room, furniture_all):
        user = f"room={room}; furniture={furniture_all}\nObject: {obj_nl}"
        return self.parse_score(self._gen(prompts.ROOM, user, max_new_tokens=24))


class Captioner(Scorer):
    """VLM one-sentence view caption (annotate_captions prompt) -- feeds the caption-LLM branch of
    the ensemble when recomputing priors in-process."""
    def caption(self, images):
        if not images:
            return ""
        return self._gen(prompts.CAPTION, "Describe this view:", images=images, max_new_tokens=48).strip()


class AnchoredRankScorer(Scorer):
    """v4 per-node relative ranking: all 12 direction views in one call, a score anchored to each
    reasoning line. `views` = list of (dir, room, furniture, image)."""
    def rank(self, obj_nl, views):
        from qwen_vl_utils import process_vision_info
        content = []
        for i, (d, room, fur, im) in enumerate(views):
            content.append({"type": "text", "text": f"[spot {i}] room={room}; furniture={fur}"})
            content.append({"type": "image", "image": im})
        content.append({"type": "text", "text": f"TARGET: {obj_nl}\nPer-spot lines, then the JSON:"})
        msgs = [{"role": "system", "content": prompts.RANK}, {"role": "user", "content": content}]
        text = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        imgs, _ = process_vision_info(msgs)
        inp = self.proc(text=[text], images=imgs, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inp, max_new_tokens=700, do_sample=False)
        txt = self.proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        s = {}
        for i in range(len(views)):
            mm = re.search(rf"spot {i}\s*:.*?=\s*([01](?:\.\d+)?)", txt)  # anchored per-spot score
            if mm:
                s[i] = float(mm.group(1))
        if not s:                                                        # fallback: trailing JSON
            m = re.findall(r"\{.*\}", txt, re.S)
            if m:
                try:
                    for k, v in json.loads(m[-1]).items():
                        s[int(k)] = max(0.0, min(1.0, float(v)))
                except Exception:
                    pass
        return s
