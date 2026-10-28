"""Lightweight joint restoration + segmentation model.

Design constraints (from MaCVi_Hardware_Benchmarking.md / Dataset_Prep_Plan.md):
  - Target: Jetson Nano (4GB), JetPack 4.6.x, TensorRT 8.2, ONNX opset 11-13.
  - Only ops with solid TensorRT 8.2 support: Conv2d, BatchNorm2d, ReLU6, bilinear
    Resize with an explicit static size, Sigmoid, Concat. No LayerNorm/GroupNorm,
    no attention, no deformable conv, no dynamic shapes.
  - Single shared encoder + two lightweight decoder heads, one forward pass ->
    (restored_image, seg_logits). This matches the single-ONNX/two-named-outputs
    submission format decided for the challenge, and reuses one feature extraction
    pass for both tasks (important on 4GB of shared memory).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DepthwiseSeparableConv(nn.Module):
    """Depthwise 3x3 + pointwise 1x1 -- standard MobileNet-style block. Exports
    cleanly to ONNX opset<=13 as plain Conv2d/BatchNorm/ReLU6, well supported by
    TensorRT 8.2."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.depthwise = ConvBNReLU(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch)
        self.pointwise = ConvBNReLU(in_ch, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class Encoder(nn.Module):
    """4-stage lightweight encoder, downsamples input by 2x per stage (16x total).
    Returns every stage's feature map for U-Net-style skip connections."""

    def __init__(self, in_ch=3, channels=(16, 32, 64, 128)):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.stem = ConvBNReLU(in_ch, c1, kernel_size=3, stride=2, padding=1)   # H/2
        self.stage1 = DepthwiseSeparableConv(c1, c1, stride=1)                  # H/2
        self.stage2 = DepthwiseSeparableConv(c1, c2, stride=2)                  # H/4
        self.stage3 = DepthwiseSeparableConv(c2, c3, stride=2)                  # H/8
        self.stage4 = DepthwiseSeparableConv(c3, c4, stride=2)                  # H/16

    def forward(self, x):
        f1 = self.stage1(self.stem(x))   # H/2
        f2 = self.stage2(f1)             # H/4
        f3 = self.stage3(f2)             # H/8
        f4 = self.stage4(f3)             # H/16
        return f1, f2, f3, f4


class DecoderBlock(nn.Module):
    """Upsample to an explicit static target size (not scale_factor) + concat skip
    + conv. Explicit sizes avoid the Shape/Gather/Concat pattern that
    scale_factor-based interpolation can emit in the ONNX graph, which is safer
    for TensorRT 8.2's parser than a runtime-computed target size."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = DepthwiseSeparableConv(in_ch + skip_ch, out_ch, stride=1)

    def forward(self, x, skip, target_size):
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class RestorationDecoder(nn.Module):
    """Reconstructs a full-resolution RGB image. Sigmoid keeps output in [0, 1]."""

    def __init__(self, channels=(16, 32, 64, 128), decoder_channels=(64, 32, 16)):
        super().__init__()
        c1, c2, c3, c4 = channels
        d3, d2, d1 = decoder_channels
        self.up3 = DecoderBlock(c4, c3, d3)   # H/16 -> H/8
        self.up2 = DecoderBlock(d3, c2, d2)   # H/8  -> H/4
        self.up1 = DecoderBlock(d2, c1, d1)   # H/4  -> H/2
        self.out_conv = nn.Conv2d(d1, 3, kernel_size=1)

    def forward(self, f1, f2, f3, f4, sizes):
        # sizes = [(H/8,W/8), (H/4,W/4), (H/2,W/2), (H,W)]
        x = self.up3(f4, f3, sizes[0])
        x = self.up2(x, f2, sizes[1])
        x = self.up1(x, f1, sizes[2])
        x = F.interpolate(x, size=sizes[3], mode='bilinear', align_corners=False)
        return torch.sigmoid(self.out_conv(x))


class SegmentationDecoder(nn.Module):
    """Lighter decoder than restoration -- segmentation needs less high-frequency
    detail. Outputs raw per-class logits; argmax is left to the evaluation harness
    (matches the LaRS-style post-processing convention in Dataset_Prep_Plan.md),
    keeping ArgMax -- a historically flaky op on older TensorRT parsers -- out of
    the exported graph."""

    def __init__(self, channels=(16, 32, 64, 128), decoder_channels=(64, 32, 16), num_classes=6):
        super().__init__()
        c1, c2, c3, c4 = channels
        d3, d2, d1 = decoder_channels
        self.up3 = DecoderBlock(c4, c3, d3)
        self.up2 = DecoderBlock(d3, c2, d2)
        self.up1 = DecoderBlock(d2, c1, d1)
        self.out_conv = nn.Conv2d(d1, num_classes, kernel_size=1)

    def forward(self, f1, f2, f3, f4, sizes):
        x = self.up3(f4, f3, sizes[0])
        x = self.up2(x, f2, sizes[1])
        x = self.up1(x, f1, sizes[2])
        x = F.interpolate(x, size=sizes[3], mode='bilinear', align_corners=False)
        return self.out_conv(x)


class JointModel(nn.Module):
    """Single shared encoder -> two lightweight decoder heads. One forward pass
    produces both outputs, matching the challenge's 'single ONNX graph, two named
    outputs' submission format."""

    def __init__(self, resolution, num_classes=6, encoder_channels=(16, 32, 64, 128),
                 seg_decoder_channels=(64, 32, 16), rest_decoder_channels=(64, 32, 16)):
        super().__init__()
        H, W = resolution
        # Static target sizes for each decoder stage, baked into the graph as
        # concrete ints at export time rather than derived from input shape at runtime.
        self.sizes = [(H // 8, W // 8), (H // 4, W // 4), (H // 2, W // 2), (H, W)]
        self.encoder = Encoder(3, encoder_channels)
        self.restoration_head = RestorationDecoder(encoder_channels, rest_decoder_channels)
        self.segmentation_head = SegmentationDecoder(encoder_channels, seg_decoder_channels, num_classes)

    def forward(self, x):
        f1, f2, f3, f4 = self.encoder(x)
        restored = self.restoration_head(f1, f2, f3, f4, self.sizes)
        seg_logits = self.segmentation_head(f1, f2, f3, f4, self.sizes)
        return restored, seg_logits
