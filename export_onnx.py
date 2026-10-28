"""Track 2 only: regenerate the ONNX model you submitted, from your weights.

The organizers call   main(model_dir, out_onnx)   and compare the result with the ONNX file on the
leaderboard. Delete this file if you only entered Track 1.

Contract of the ONNX file: static input shape (1, 3, H, W) with values in [0, 1]; outputs named
`restored_image` and `seg_logits`; no dynamic axes; at most 12 GFLOPs and 1.0 M parameters.
"""
import os

import torch

from src.joint_model import JointModel

RES = (256, 256)
NUM_CLASSES = 5


def main(model_dir, out_onnx):
    model = JointModel(resolution=RES, num_classes=NUM_CLASSES,
                       encoder_channels=(16, 32, 64, 128),
                       seg_decoder_channels=(64, 32, 16),
                       rest_decoder_channels=(64, 32, 16))
    model.load_state_dict(torch.load(os.path.join(model_dir, "best.pt"), map_location="cpu", weights_only=True))
    model.eval()
    os.makedirs(os.path.dirname(out_onnx) or ".", exist_ok=True)
    H, W = RES
    torch.onnx.export(model, torch.randn(1, 3, H, W), out_onnx,
                      input_names=["input"], output_names=["restored_image", "seg_logits"],
                      opset_version=12, do_constant_folding=True, dynamic_axes=None)
