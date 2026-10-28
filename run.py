"""Track 1 inference entry point (reference implementation: the EURS organizer baseline).

The organizers call   main(model_dir, input_dir, output_dir, device)   and nothing else.

  model_dir   the weights/ folder of this repository (your checkpoint(s) are in there)
  input_dir   a folder of degraded images (.png/.jpg), all at their original size
  output_dir  write here, one file per input image, same file stem:
                  output_dir/restored/<stem>.png    RGB restored image
                  output_dir/seg_masks/<stem>.png   single-channel mask, labels 0..4
              both resized to EXACTLY the scoring size: longest side 1024 px, aspect ratio kept
              (downscale only: an input at or below 1024 px keeps its size); the evaluator rejects any other size
  device      "cuda" or "cpu"

Rules for this file (see README.md): inference only; every output must come from its own input
image alone; no network access; deterministic (fixed seeds, model.eval()).
The organizers add this folder and src/ to the import path, so `from src.xxx import ...` works.
"""
import os

import numpy as np
import torch
from PIL import Image

from src.joint_model import JointModel

RES = (256, 256)          # (H, W) the baseline network runs at
NUM_CLASSES = 5           # 0 background, 1 coral, 2 fish, 3 elasmobranch, 4 other
MAX_SIDE = 1024           # scoring size cap


def _scoring_size(w, h):
    s = MAX_SIDE / max(w, h)
    return (w, h) if s >= 1 else (max(1, round(w * s)), max(1, round(h * s)))


def main(model_dir, input_dir, output_dir, device):
    torch.set_grad_enabled(False)
    model = JointModel(resolution=RES, num_classes=NUM_CLASSES,
                       encoder_channels=(16, 32, 64, 128),
                       seg_decoder_channels=(64, 32, 16),
                       rest_decoder_channels=(64, 32, 16)).to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, "best.pt"), map_location=device, weights_only=True))
    model.eval()

    os.makedirs(os.path.join(output_dir, "restored"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "seg_masks"), exist_ok=True)
    H, W = RES
    for name in sorted(os.listdir(input_dir)):
        stem, ext = os.path.splitext(name)
        if ext.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        with Image.open(os.path.join(input_dir, name)) as im:
            im = im.convert("RGB")
            tw, th = _scoring_size(*im.size)
            small = im.resize((W, H), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(small, dtype=np.float32).transpose(2, 0, 1) / 255.0).unsqueeze(0).to(device)
        restored, seg_logits = model(x)
        r = restored[0].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
        # NOTE: copy YOUR OWN conversion to 8 bit here (this baseline truncates; many methods use .round()):
        Image.fromarray((r * 255).astype(np.uint8)).resize((tw, th), Image.BILINEAR).save(
            os.path.join(output_dir, "restored", stem + ".png"))
        m = seg_logits[0].argmax(dim=0).byte().cpu().numpy()
        Image.fromarray(m).resize((tw, th), Image.NEAREST).save(os.path.join(output_dir, "seg_masks", stem + ".png"))
