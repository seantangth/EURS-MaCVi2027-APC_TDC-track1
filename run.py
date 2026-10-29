"""Track 1 inference: reproduces EURS_t1_sub046 (leaderboard submission #19441), team APC_TDC.

The organizers call   main(model_dir, input_dir, output_dir, device).

Per image, one image at a time, using that image only:
  1. Scale the full-resolution input to the scoring size (longest side 1024, PIL bilinear, same formula as the official
     resize_for_submission.py) and convert to RGB. Our leaderboard pipeline cached exactly this image as a PNG.
  2. Restoration: NAFNet conditioned on the water type and level read from the END of the file name
     (`<anything>_<type>_<level>`), bfloat16 autocast on CUDA, clamp, round to 8 bit, saved as JPEG (quality 98, 4:4:4).
  3. Segmentation: three teachers, each on the scaled image stretched to its own input size with horizontal-flip TTA:
     DINOv3 ViT-L + LoRA + FPN at 1024x1360, the same family trained with partial external labels at 768x1024, and
     Mask2Former Swin-L at 768x1024. Each argmax mask is resized back with NEAREST; the three masks are combined
     by a per-pixel majority vote, ties go to the first teacher.
This is exactly our leaderboard pipeline (it ran as separate batch scripts; here the same functions run per image).
"""
import os
import sys

# No network and a read-only root at run time: keep every library offline and its caches in /tmp.
for _k, _v in (("HF_HUB_OFFLINE", "1"), ("TRANSFORMERS_OFFLINE", "1"), ("HF_HOME", "/tmp/hf_home"),
               ("XDG_CACHE_HOME", "/tmp/.cache"), ("TORCH_HOME", "/tmp/torch_home")):
    os.environ.setdefault(_k, _v)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from eurs.restorer import load_restorer  # noqa: E402
from eurs.teacher import aspect_prep, aspect_undo_mask, load_teacher  # noqa: E402

MAX_SIDE = 1024
# (file in weights/, model input (H, W)); the first teacher breaks ties in the vote
TEACHERS = [("teacher_exp066b.pt", (1024, 1360)),
            ("teacher_exp083.pt", (768, 1024)),
            ("teacher_exp078_m2f.pt", (768, 1024))]
RESTORER = "restorer_exp034.pt"
JPEG_KW = dict(quality=98, subsampling=0, optimize=True)


def _scoring_size(w, h):
    s = MAX_SIDE / max(w, h)
    return (w, h) if s >= 1 else (max(1, round(w * s)), max(1, round(h * s)))


def _autocast(device):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda"))


def _vote(masks):
    st = np.stack(masks)                                                         # (K,H,W), st[0] = tie-breaker
    cnt = np.stack([(st == c).sum(0) for c in range(5)])                         # (5,H,W)
    score = cnt * 10 + (np.arange(5)[:, None, None] == st[0][None]).astype(np.int64)
    return score.argmax(0).astype(np.uint8)


def main(model_dir, input_dir, output_dir, device):
    torch.manual_seed(0)
    np.random.seed(0)
    with torch.no_grad():
        _main(model_dir, input_dir, output_dir, torch.device(device))


def _main(model_dir, input_dir, output_dir, dev):
    teachers = [(load_teacher(os.path.join(model_dir, f), dev)[0], hw) for f, hw in TEACHERS]
    restorer = load_restorer(os.path.join(model_dir, RESTORER), dev)
    os.makedirs(os.path.join(output_dir, "restored"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "seg_masks"), exist_ok=True)

    for name in sorted(os.listdir(input_dir)):
        stem, ext = os.path.splitext(name)
        if ext.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        with Image.open(os.path.join(input_dir, name)) as im0:
            tw, th = _scoring_size(*im0.size)
            im = im0 if im0.size == (tw, th) else im0.resize((tw, th), Image.BILINEAR)
            im = im.convert("RGB")

        # restoration (exp034 Meta NAFNet)
        _, a, b = stem.rsplit("_", 2)          # water type and level: the end of the name only
        a, b = int(a), int(b)
        x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)[None]
        with _autocast(dev):
            out = restorer(x.to(dev), torch.tensor([a], device=dev), torch.tensor([b], device=dev)).float().clamp(0, 1)
        J = (out[0].permute(1, 2, 0).cpu().numpy() * 255).round() / 255.0
        Image.fromarray((J * 255).round().astype(np.uint8)).save(os.path.join(output_dir, "restored", stem + ".jpg"), "JPEG", **JPEG_KW)

        # segmentation (three teachers, hflip TTA, majority vote)
        masks = []
        for model, (H, W) in teachers:
            imr, info = aspect_prep(im, H, W, False)
            xt = torch.from_numpy(np.asarray(imr, dtype=np.float32) / 255.0).permute(2, 0, 1)[None].to(dev).to(memory_format=torch.channels_last)
            with _autocast(dev):
                logit = model(xt).float()
                logit = 0.5 * (logit + torch.flip(model(torch.flip(xt, [3])).float(), [3]))
                prob = logit.softmax(1)
            lab = prob.argmax(1)[0].cpu().numpy().astype(np.uint8)
            masks.append(np.asarray(aspect_undo_mask(lab, info, im.size)))
        Image.fromarray(_vote(masks)).save(os.path.join(output_dir, "seg_masks", stem + ".png"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else ("cuda" if torch.cuda.is_available() else "cpu"))
