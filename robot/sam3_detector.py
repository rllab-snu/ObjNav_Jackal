"""In-process SAM 3 open-vocab detector for the real robot (NO server).

On the robot there is no habitat (py3.9) / SAM 3 (py3.12) split, so SAM 3 can be
loaded directly in the same process -- the localhost server (`sam3_bridge`) that the
habitat pipeline needs is unnecessary here. Same `segment(rgb, prompt)` interface as
`sam3_bridge.client.Sam3Client`, so the navigator's detection code is identical.

torch / PIL / sam3 are imported lazily in __init__, so `robot/` still imports fine on
a machine without SAM 3 (nav-only deployment).
"""
import numpy as np
from threading import RLock

DEFAULT_BPE = "third_party/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"


class Sam3Detector:
    def __init__(self, bpe_path=DEFAULT_BPE, device="cuda", confidence_threshold=0.3):
        import torch
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        self._torch = torch
        self._model = build_sam3_image_model(bpe_path, device=device)
        self._proc = Sam3Processor(self._model, device=device,
                                   confidence_threshold=confidence_threshold)
        self._amp = (torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda"
                     else torch.autocast("cpu"))
        self.device = device
        # The capture writer can run in a background thread.  SAM3's processor
        # mutates per-inference state, so calls sharing this model must serialize.
        self._segment_lock = RLock()

    def segment(self, rgb, prompt):
        """rgb (H,W,3) uint8, prompt text -> {"mask": bool HxW union, "masks": bool
        (N,H,W) per-instance, "boxes", "scores"}. Same contract as the server client:
        `mask` is the union (for viz), `masks` are the individual detections of `prompt`."""
        from PIL import Image
        rgb = np.ascontiguousarray(np.asarray(rgb)[..., :3], dtype=np.uint8)
        h, w = rgb.shape[:2]
        img = Image.fromarray(rgb)
        with self._segment_lock:
            with self._torch.no_grad(), self._amp:
                st = self._proc.set_image(img)
                out = self._proc.set_text_prompt(state=st, prompt=str(prompt))
        masks = out["masks"]                               # [N,1,H,W]
        if masks.numel() and masks.shape[0] > 0:
            inst = (masks.squeeze(1) > 0.5).cpu().numpy().astype(bool)   # [N,H,W]
            boxes = out["boxes"].float().cpu().tolist()
        else:
            inst = np.zeros((0, h, w), bool)
            boxes = []
        union = inst.any(0) if inst.shape[0] else np.zeros((h, w), bool)
        return {"mask": union, "masks": inst, "boxes": boxes,
                "scores": out["scores"].float().cpu().tolist()}
