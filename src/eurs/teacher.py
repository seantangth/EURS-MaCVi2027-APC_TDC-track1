"""分割 teacher（act007／exp012）：ImageNet 預訓練骨幹（timm）＋ FPN 解碼器，不受 12 GFLOPs 限制。
用途：(a) leave-scene-out CV 量新場景泛化；(b) 對 dev／外部影像產偽標籤（硬標籤 PNG ＋ 1/4 解析 fp16 softmax 供蒸餾）；(c) t1 提交的遮罩。
輸入與學生一致：0–1 RGB、拉伸到靜態 (H,W)；ImageNet 正規化在模型內部。

用法（箱上）：
  python -I eurs/teacher.py train --root ~/eurs/processed --side 768 --fold 0 --hw 768 1024 --epochs 20 --bs 6 \
      --backbone convnext_small.fb_in22k_ft_in1k_384 --out ~/eurs/runs/EURS_exp012_f0 --exp exp012
  python -I eurs/teacher.py predict --ckpt ~/eurs/runs/EURS_exp012_f-1/best.pt --hw 768 1024 \
      --degraded-dir ~/eurs/processed/cache1024/dev/degraded --out-masks ~/eurs/pseudo/dev_masks --out-logits ~/eurs/pseudo/dev_logits --tta
exp078（Mask2Former，HF transformers；預設權重 facebook/mask2former-swin-large-ade-semantic，整體微調、原生集合預測損失）：
  python -I eurs/teacher.py train --root ~/eurs/processed --side 1024 --fold -1 --hw 768 1024 --backbone m2f_swin_l --epochs 16 --bs 4 \
      --lr 1e-4 --lr-backbone 1e-5 --clip-grad 0.01 --ext-root ~/eurs/processed/external --ext-share 0.35 --ext-rare-boost 4 \
      --out ~/eurs/runs/EURS_exp078_f-1 --exp exp078        （OOM 時改 --bs 2 --accum 2：每 epoch 的 optimizer step 數不變）
  predict 不變（checkpoint 內存 HF config，load_teacher 不必連網即可重建）。
"""
import argparse
import copy
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eurs import folds as F_
from eurs.data import EURSPairs, ExtSeg, mix_weights, scene_balanced_sampler_weights
from eurs.losses import SegLoss
from eurs.metrics import Evaluator

IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD = (0.229, 0.224, 0.225)


class ConvBNAct(nn.Sequential):
    def __init__(self, cin, cout, k=3):
        super().__init__(nn.Conv2d(cin, cout, k, padding=k // 2, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class LoRAQKV(nn.Module):
    """融合 qkv Linear 的 LoRA，只加在 q 與 v（rank r）：qkv(x)＋[B_q·A_q·x, 0, B_v·A_v·x]·alpha/r。B 初始為 0，起點＝原模型。"""

    def __init__(self, qkv, r=8, alpha=16):
        super().__init__()
        self.qkv, self.scale = qkv, alpha / r
        d_in, d_out = qkv.in_features, qkv.out_features // 3
        self.a_q, self.b_q = nn.Linear(d_in, r, bias=False), nn.Linear(r, d_out, bias=False)
        self.a_v, self.b_v = nn.Linear(d_in, r, bias=False), nn.Linear(r, d_out, bias=False)
        for m in (self.a_q, self.a_v):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        for m in (self.b_q, self.b_v):
            nn.init.zeros_(m.weight)

    def forward(self, x):
        dq = self.b_q(self.a_q(x)) * self.scale
        dv = self.b_v(self.a_v(x)) * self.scale
        return self.qkv(x) + torch.cat([dq, torch.zeros_like(dq), dv], -1)


def vit_pyramid(c):
    """ViT 四層 stride-16 特徵 → strides 4/8/16/32（DPT／ViTDet 式重組），回傳 (模組, 各層通道)。"""
    return nn.ModuleList([
        nn.Sequential(nn.ConvTranspose2d(c, c // 2, 2, 2), nn.GELU(), nn.ConvTranspose2d(c // 2, c // 4, 2, 2)),
        nn.ConvTranspose2d(c, c // 2, 2, 2),
        nn.Identity(),
        nn.MaxPool2d(2)]), [c // 4, c // 2, c, c]


class SegTeacher(nn.Module):
    """timm features_only 骨幹（strides 4/8/16/32）→ FPN（256ch）→ 四層上採樣到 stride 4 後 concat → 分類頭 → bilinear 回輸入解析。
    ViT 骨幹（四層皆 stride 16，--out-indices 指定取哪幾個 block）先經 vit_pyramid 重組成四個尺度。
    freeze＝骨幹不更新（只訓 FPN／頭）；lora_r>0＝凍結骨幹上對每個 attention 的 q、v 加 LoRA。"""

    def __init__(self, backbone='convnext_small.fb_in22k_ft_in1k_384', num_classes=5, fpn_ch=256, pretrained=True, drop_path=0.1,
                 out_indices=None, freeze=False, lora_r=0):
        super().__init__()
        import timm
        self.backbone = timm.create_model(backbone, pretrained=pretrained, features_only=True, out_indices=tuple(out_indices or (0, 1, 2, 3)),
                                          drop_path_rate=0.0 if freeze else drop_path)
        chs = self.backbone.feature_info.channels()
        self.pyr = None
        if len(set(self.backbone.feature_info.reduction())) == 1:
            self.pyr, chs = vit_pyramid(chs[0])
        self.freeze = freeze or lora_r > 0
        if self.freeze:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        if lora_r > 0:  # 凍結之後才加，LoRA 參數保持可訓練
            for mod in list(self.backbone.modules()):
                if isinstance(getattr(mod, 'qkv', None), nn.Linear):
                    mod.qkv = LoRAQKV(mod.qkv, lora_r)
        self.no_grad_backbone = freeze and lora_r == 0
        self.lateral = nn.ModuleList([nn.Conv2d(c, fpn_ch, 1) for c in chs])
        self.smooth = nn.ModuleList([ConvBNAct(fpn_ch, fpn_ch) for _ in chs])
        self.fuse = ConvBNAct(fpn_ch * len(chs), fpn_ch)
        self.head = nn.Sequential(ConvBNAct(fpn_ch, fpn_ch), nn.Dropout2d(0.1), nn.Conv2d(fpn_ch, num_classes, 1))
        self.register_buffer('mean', torch.tensor(IMNET_MEAN).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(IMNET_STD).view(1, 3, 1, 1))

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:  # 凍結的骨幹固定在 eval（drop_path／dropout 關閉）
            self.backbone.eval()
        return self

    def forward(self, x):
        H, W = x.shape[-2:]
        xn = (x - self.mean) / self.std
        if self.no_grad_backbone:
            with torch.no_grad():
                feats = self.backbone(xn)
        else:
            feats = self.backbone(xn)
        if self.pyr is not None:
            feats = [p(f) for p, f in zip(self.pyr, feats)]
        lat = [l(f) for l, f in zip(self.lateral, feats)]
        for i in range(len(lat) - 1, 0, -1):  # top-down
            lat[i - 1] = lat[i - 1] + F.interpolate(lat[i], size=lat[i - 1].shape[-2:], mode='bilinear', align_corners=False)
        outs = [s(l) for s, l in zip(self.smooth, lat)]
        tgt = outs[0].shape[-2:]
        outs = [outs[0]] + [F.interpolate(o, size=tgt, mode='bilinear', align_corners=False) for o in outs[1:]]
        y = self.head(self.fuse(torch.cat(outs, 1)))  # stride 4 logits
        return F.interpolate(y, size=(H, W), mode='bilinear', align_corners=False)

    def param_groups(self, lr_backbone, lr_head, wd):
        bb = [p for p in self.backbone.parameters() if p.requires_grad]
        hd = [p for n, p in self.named_parameters() if not n.startswith('backbone.') and p.requires_grad]
        return [{'params': bb, 'lr': lr_backbone, 'weight_decay': wd}, {'params': hd, 'lr': lr_head, 'weight_decay': wd}]


EPFL_DPT_URL = 'https://huggingface.co/EPFL-ECEO/coralscapesv2-dinov3-vitl-lora-dpt/resolve/main/model.safetensors'
# exp067：Coralscapes 39 類（id 0＝void）→ EURS 5 類（0 bg、1 coral、2 fish、3 elasmo、4 other）；只用來初始化 5 類分類層，之後整體再訓
CORALSCAPES_TO_EURS = {
    0: [1, 2, 5, 7, 8, 13, 14, 15],                     # seagrass、trash、sand、human、transect tools、background、dark、transect line
    1: [3, 4, 6, 10, 12, 16, 17, 18, 19, 20, 21, 22, 23, 25, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37],   # 各類珊瑚、藻覆底質、硬底質、碎礫、海綿、海葵
    2: [9],                                              # fish
    3: [9],                                              # elasmo：Coralscapes 沒有，先借 fish 的權重
    4: [11, 24, 26, 35, 38, 39],                         # other animal、clam、sea cucumber、sea urchin、crown of thorn、dead clam
}


class LoRALinear(nn.Module):
    """一般 nn.Linear 的 LoRA（HF DINOv3 的 q_proj／v_proj 是分開的 Linear）：base(x)＋B·A·x·alpha/r，B 初始 0＝起點等於原模型。"""

    def __init__(self, base, r=8, alpha=16):
        super().__init__()
        self.base, self.scale = base, alpha / r
        self.a, self.b = nn.Linear(base.in_features, r, bias=False), nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5)); nn.init.zeros_(self.b.weight)

    def forward(self, x):
        return self.base(x) + self.b(self.a(x)) * self.scale


class EpflDPTTeacher(nn.Module):
    """exp067：EPFL CoralscapesV2 的 DINOv3 ViT-L/16（LoRA 已合併）＋DPT 解碼器（HF transformers 的 DPTNeck／DPTSemanticSegmentationHead），
    39 類分類層依 CORALSCAPES_TO_EURS 初始化成 5 類，骨幹凍結後在 q／v 加 rank-r LoRA。介面同 SegTeacher：吃 0–1 影像、回傳原解析 logits、
    param_groups(lr_backbone, lr_head, wd)、train() 時骨幹固定 eval（RoPE 的訓練期增強不啟用）。結構與 EPFL 權重的 553 個張量逐一核對過（10-25）。"""
    LAYERS = (0, 8, 15, 23)   # EPFL 的 backbone_layer_indices＝None → linspace(0, 23, 4) 四捨五入

    def __init__(self, num_classes=5, lora_r=8, epfl_ckpt=''):
        super().__init__()
        from transformers import DINOv3ViTConfig, DINOv3ViTModel
        from transformers.models.dpt.configuration_dpt import DPTConfig
        from transformers.models.dpt.modeling_dpt import DPTNeck, DPTSemanticSegmentationHead
        cfg = DINOv3ViTConfig(hidden_size=1024, intermediate_size=4096, num_hidden_layers=24, num_attention_heads=16, patch_size=16, num_register_tokens=4,
                              use_gated_mlp=False, key_bias=False, query_bias=True, value_bias=True, proj_bias=True, mlp_bias=True, rope_theta=100.0,
                              layer_norm_eps=1e-5)
        cfg._attn_implementation = 'sdpa'
        self.backbone = DINOv3ViTModel(cfg)
        dcfg = DPTConfig(hidden_size=1024, num_labels=40, patch_size=16, image_size=224, num_channels=3, is_hybrid=False, readout_type='project',
                         neck_hidden_sizes=(96, 192, 384, 768), reassemble_factors=(4, 2, 1, 0.5), fusion_hidden_size=256, head_in_index=-1,
                         use_auxiliary_head=False, use_batch_norm_in_fusion_residual=True, semantic_classifier_dropout=0.1)
        self.neck, self.seg_head = DPTNeck(dcfg), DPTSemanticSegmentationHead(dcfg)
        if epfl_ckpt:
            self._load_epfl(epfl_ckpt)
        old = self.seg_head.head[4]
        new = nn.Conv2d(old.in_channels, num_classes, 1)
        with torch.no_grad():
            for c, ids in CORALSCAPES_TO_EURS.items():
                if c < num_classes:
                    new.weight[c] = old.weight[ids].mean(0); new.bias[c] = old.bias[ids].mean() + math.log(len(ids))
        self.seg_head.head[4] = new
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.freeze, self.lora_r = True, lora_r
        if lora_r > 0:
            for layer in self.backbone.model.layer:
                layer.attention.q_proj = LoRALinear(layer.attention.q_proj, lora_r)
                layer.attention.v_proj = LoRALinear(layer.attention.v_proj, lora_r)
        self.register_buffer('mean', torch.tensor(IMNET_MEAN).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(IMNET_STD).view(1, 3, 1, 1))

    def _load_epfl(self, path):
        if path.startswith('http'):
            import urllib.request
            local = os.path.join(os.path.expanduser('~/.cache/eurs'), 'epfl_coralscapesv2_dinov3_vitl_lora_dpt.safetensors')
            if not os.path.exists(local):
                os.makedirs(os.path.dirname(local), exist_ok=True)
                urllib.request.urlretrieve(path, local + '.part'); os.replace(local + '.part', local)
            path = local
        from safetensors.torch import load_file
        sd = {('backbone.' + k[len('encoder.'):] if k.startswith('encoder.') else k): v for k, v in load_file(path).items()}
        missing, unexpected = self.load_state_dict(sd, strict=False)
        assert not missing and not unexpected, f'EPFL 權重對不上：missing={missing[:5]} unexpected={unexpected[:5]}'
        print(f'loaded EPFL CoralscapesV2 DINOv3-L DPT weights: {len(sd)} tensors from {path}', flush=True)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x):
        H, W = x.shape[-2:]
        xn = (x - self.mean) / self.std
        hs = self.backbone(pixel_values=xn, output_hidden_states=True).hidden_states
        off = 1 if len(hs) == len(self.backbone.model.layer) + 1 else 0
        feats = [torch.cat([hs[i + off][:, :1], hs[i + off][:, 1 + 4:]], 1) for i in self.LAYERS]   # 去掉 4 個 register token，保留 CLS
        logits = self.seg_head(self.neck(feats, patch_height=H // 16, patch_width=W // 16))
        return F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False) if logits.shape[-2:] != (H, W) else logits

    def param_groups(self, lr_backbone, lr_head, wd):
        bb = [p for p in self.backbone.parameters() if p.requires_grad]
        hd = [p for n, p in self.named_parameters() if not n.startswith('backbone.') and p.requires_grad]
        return [{'params': bb, 'lr': lr_backbone, 'weight_decay': wd}, {'params': hd, 'lr': lr_head, 'weight_decay': wd}]


M2F_HUB = {'m2f_swin_l': 'facebook/mask2former-swin-large-ade-semantic',   # exp078：Swin-L（IN21k）→ ADE20K 語意分割
           'm2f_swin_t': 'facebook/mask2former-swin-tiny-ade-semantic'}    # 只供本機 CPU 煙霧測試
EURS_LABELS = ('background', 'coral', 'fish', 'elasmobranch', 'other')


def _hf_tensors(ckpt, names):
    """從 HF checkpoint（hub id 或本機資料夾；model.safetensors 優先，否則 pytorch_model.bin）讀出指定張量；讀不到回傳 None。
    hub id 只查本機快取（from_pretrained 剛下載過），不連網、不會另外下載第二份權重。"""
    from huggingface_hub import hf_hub_download
    for fname in ('model.safetensors', 'pytorch_model.bin'):
        try:
            path = os.path.join(ckpt, fname) if os.path.isdir(ckpt) else hf_hub_download(ckpt, fname, local_files_only=True)
        except Exception:
            continue
        if not os.path.exists(path):
            continue
        if fname.endswith('.safetensors'):
            from safetensors import safe_open
            with safe_open(path, framework='pt') as f:
                return [f.get_tensor(n) for n in names]
        sd = torch.load(path, map_location='cpu', weights_only=True)
        return [sd[n] for n in names]
    return None


class M2FTeacher(nn.Module):
    """exp078：HF transformers 的 Mask2Former（query 式 mask classification，ADE20K 語意分割預訓練 Swin），整體微調。
    分類層換成 5 類＋no-object（no-object 列沿用 ADE 權重，5 類列隨機）。輸入同其他 teacher：0–1 影像，ImageNet 正規化在模型內部（與 HF 前處理同值）。
    forward(x) → (B,5,H,W) 逐像素語意分數的 log：同 detectron2 semantic_inference，mask logits 先 bilinear 回輸入解析再 sigmoid，
    與去掉 no-object 欄的類別 softmax 做 einsum；log 之後的 softmax＝分數正規化，既有 predict／TTA／evaluate 不必改。
    loss(x, m)：Mask2Former 原生集合預測損失（Hungarian 匹配＋類別 CE＋point-sampled BCE／Dice，含 9 層輔助），一律 fp32；
    每張圖的每個出現類別一個二值遮罩，255（ignore）不屬於任何遮罩（同 detectron2 語意 mapper）。"""
    is_m2f = True

    def __init__(self, ckpt, num_classes=5, pretrained=True, hf_config=None):
        super().__init__()
        from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation
        id2label = {i: (EURS_LABELS[i] if i < len(EURS_LABELS) else f'class{i}') for i in range(num_classes)}
        label2id = {v: k for k, v in id2label.items()}
        if hf_config:   # load_teacher：用 checkpoint 內存的 HF config 重建骨架（不連網），權重之後由 state_dict 載入
            self.m2f = Mask2FormerForUniversalSegmentation(Mask2FormerConfig.from_dict(json.loads(hf_config)))
        elif pretrained:
            self.m2f = Mask2FormerForUniversalSegmentation.from_pretrained(ckpt, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)
            old = _hf_tensors(ckpt, ['class_predictor.weight', 'class_predictor.bias'])
            if old is not None:   # no-object 列（最後一列）沿用預訓練值：起點的「此 query 無物件」判斷是校準過的
                with torch.no_grad():
                    self.m2f.class_predictor.weight[-1].copy_(old[0][-1]); self.m2f.class_predictor.bias[-1].copy_(old[1][-1])
            print(f'loaded Mask2Former {ckpt}: class head {"?" if old is None else old[0].shape[0]} -> {num_classes}+1, no-object row '
                  f'{"copied" if old is not None else "random"}, backbone drop_path {self.m2f.config.backbone_config.drop_path_rate}', flush=True)
        else:
            self.m2f = Mask2FormerForUniversalSegmentation(Mask2FormerConfig.from_pretrained(ckpt, id2label=id2label, label2id=label2id))
        assert self.m2f.config.num_labels == num_classes, f'num_labels={self.m2f.config.num_labels}'
        self.num_classes = num_classes
        self.register_buffer('mean', torch.tensor(IMNET_MEAN).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(IMNET_STD).view(1, 3, 1, 1))

    @property
    def backbone(self):   # cmd_train 印參數量用；property 不會在 state_dict 裡重複註冊
        return self.m2f.model.pixel_level_module.encoder

    def hf_config_json(self):
        return self.m2f.config.to_json_string(use_diff=False)

    def _run(self, x, aux=False):
        return self.m2f(pixel_values=((x - self.mean) / self.std).contiguous(), output_auxiliary_logits=aux)

    def forward(self, x):
        H, W = x.shape[-2:]
        out = self._run(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            cls = out.class_queries_logits.float().softmax(-1)[..., :-1]   # (B,Q,C)，去掉 no-object
            msk = F.interpolate(out.masks_queries_logits.float(), size=(H, W), mode='bilinear', align_corners=False).sigmoid_()   # (B,Q,H,W)
            sem = torch.einsum('bqc,bqhw->bchw', cls, msk)
        return sem.clamp_min(1e-6).log()

    def loss(self, x, m):
        """x：(B,3,H,W) 0–1；m：(B,H,W) int64，0–4，255＝ignore。呼叫端可包 bf16 autocast（只影響網路本體）；匹配與損失在 fp32。"""
        out = self._run(x, aux=True)
        keep, mask_labels, class_labels = [], [], []
        for i in range(m.shape[0]):
            cl = torch.unique(m[i]); cl = cl[cl < self.num_classes]   # 255 不成遮罩
            if cl.numel():
                keep.append(i); class_labels.append(cl); mask_labels.append((m[i][None] == cl[:, None, None]).float())
        if not keep:   # 整批只有 ignore（極少見）：回傳帶梯度的 0
            return out.masks_queries_logits.sum() * 0 + out.class_queries_logits.sum() * 0
        idx = torch.tensor(keep, device=x.device) if len(keep) < m.shape[0] else None   # 沒有任何類別的影像不進損失
        pick = lambda t: (t if idx is None else t.index_select(0, idx)).float()
        with torch.autocast(device_type=x.device.type, enabled=False):
            aux = [{k: pick(v) for k, v in d.items()} for d in out.auxiliary_logits]
            ld = self.m2f.get_loss_dict(pick(out.masks_queries_logits), pick(out.class_queries_logits), mask_labels, class_labels, aux)
            return self.m2f.get_loss(ld)

    def param_groups(self, lr_backbone, lr_head, wd):
        """同 detectron2 Mask2Former：骨幹 lr 較低；norm 層、embedding、relative_position_bias_table、level_embed 不做 weight decay。"""
        norm_like = (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d, nn.Embedding)
        groups, seen = {}, set()
        for mn, mod in self.m2f.named_modules():
            for pn, p in mod.named_parameters(recurse=False):
                if not p.requires_grad or id(p) in seen:
                    continue
                seen.add(id(p))
                bb = (mn + '.').startswith('model.pixel_level_module.encoder.')
                decay = not (isinstance(mod, norm_like) or pn in ('relative_position_bias_table', 'level_embed', 'absolute_pos_embed'))
                groups.setdefault((bb, decay), []).append(p)
        assert len(seen) == sum(1 for p in self.parameters() if p.requires_grad)
        return [{'params': ps, 'lr': lr_backbone if bb else lr_head, 'weight_decay': wd if decay else 0.0} for (bb, decay), ps in sorted(groups.items(), reverse=True)]


def build_teacher(backbone, pretrained=True, drop_path=0.1, out_indices=None, freeze=False, lora_r=0, epfl_ckpt='', m2f_ckpt='', hf_config=None):
    """backbone＝'epfl_dpt' → EpflDPTTeacher（exp067）；'m2f_*' → M2FTeacher（exp078，HF Mask2Former）；其他 → SegTeacher（timm 骨幹＋FPN）。"""
    if backbone == 'epfl_dpt':
        return EpflDPTTeacher(lora_r=lora_r, epfl_ckpt=epfl_ckpt if pretrained else '')
    if backbone.startswith('m2f'):
        if not (m2f_ckpt or backbone in M2F_HUB):
            raise ValueError(f'未知的 {backbone}：可用 {sorted(M2F_HUB)}，或以 --m2f-ckpt 指定 HF 權重')
        return M2FTeacher(m2f_ckpt or M2F_HUB[backbone], pretrained=pretrained, hf_config=hf_config)
    return SegTeacher(backbone, pretrained=pretrained, drop_path=drop_path, out_indices=out_indices, freeze=freeze, lora_r=lora_r)


class EMA:
    def __init__(self, model, decay=0.999):
        self.m = copy.deepcopy(model).eval(); self.d = decay
        for p in self.m.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k, v in self.m.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.d).add_(msd[k].detach(), alpha=1 - self.d)
            else:
                v.copy_(msd[k])


def _args(a):
    """argparse Namespace → 可序列化 dict（去掉 subparser 掛的函式，否則 torch.load weights_only 會拒收）。"""
    return {k: v for k, v in vars(a).items() if k != 'fn'}


def _ck_extra(model):
    """exp078：M2F 的 checkpoint 多存 HF config（JSON 字串，weights_only 可讀），load_teacher 不必連網即可重建；其他 teacher 回傳 {}（checkpoint 不變）。"""
    return {'hf_config': model.hf_config_json()} if getattr(model, 'is_m2f', False) else {}


def _autocast(device):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == 'cuda'))


def aspect_prep(im, H, W, aspect_norm):
    """把 PIL 影像變成模型輸入 (W,H)。aspect_norm=False：直接拉伸（與訓練、Track 2 伺服器一致）。
    aspect_norm=True（exp017）：直幅先逆時針轉 90°；轉完若比 W:H 寬，上下反射補邊到 W:H，再縮放（不變形）；比 W:H 窄則照舊拉伸。
    回傳 (縮放後影像, 還原資訊)。"""
    rot, top, bot = False, 0, 0
    if aspect_norm:
        if im.height > im.width:
            im = im.transpose(Image.ROTATE_90); rot = True
        w, h = im.size
        th = int(round(w * H / W))
        if th > h:
            top = (th - h) // 2; bot = th - h - top
            im = Image.fromarray(np.pad(np.asarray(im), ((top, bot), (0, 0), (0, 0)), mode='reflect'))
    return im.resize((W, H), Image.BILINEAR), (rot, top, bot, im.size)


def aspect_undo_mask(lab, info, size):
    """模型網格的標籤 (H,W) uint8 → 原影像尺寸 size=(w,h) 的 PIL 標籤（nearest）。"""
    rot, top, bot, (pw, ph) = info
    m = Image.fromarray(lab, mode='L').resize((pw, ph), Image.NEAREST)
    if top or bot:
        m = m.crop((0, top, pw, ph - bot))
    if rot:
        m = m.transpose(Image.ROTATE_270)
    return m if m.size == size else m.resize(size, Image.NEAREST)


def aspect_undo_prob(prob, info, out_hw):
    """模型網格的機率 (C,H,W) → 原影像拉伸網格 out_hw 的機率（供蒸餾，與學生的拉伸輸入對齊）。"""
    rot, top, bot, (pw, ph) = info
    p = F.interpolate(prob[None], size=(ph, pw), mode='bilinear', align_corners=False)[0]
    if top or bot:
        p = p[:, top:ph - bot]
    if rot:
        p = torch.rot90(p, -1, dims=(1, 2))
    return F.interpolate(p[None], size=out_hw, mode='area')[0]


@torch.no_grad()
def evaluate(model, loader, device, max_batches=None):
    """只量分割（復原欄位以 identity 填入，不可比）。"""
    model.eval(); ev = Evaluator(with_nr=False, nr_every=1)
    for bi, batch in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        x = batch['x'].to(device, non_blocking=True)
        with _autocast(device):
            seg = model(x)
        pm = seg.float().argmax(1).cpu().numpy()
        xi = batch['x'].permute(0, 2, 3, 1).numpy(); gy = batch['y'].permute(0, 2, 3, 1).numpy(); gm = batch['m'].numpy()
        for i in range(x.shape[0]):
            ev.add(xi[i], gy[i], pm[i], gm[i])
    return ev.summary()


def cmd_train(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log = open(os.path.join(a.out, 'train.log'), 'a')
    def P(*s):
        msg = ' '.join(str(t) for t in s); print(msg, flush=True); log.write(msg + '\n'); log.flush()
    P(f'[{time.strftime("%F %T")}] start TEACHER exp={a.exp} fold={a.fold} backbone={a.backbone} hw={a.hw} device={device} args={_args(a)}')

    tr_scenes, va_scenes = F_.split(a.fold)
    ds_tr = EURSPairs(a.root, a.side, tr_scenes, hw=a.hw, train=True)
    rare = {s: a.rare_boost for s in ['ray1', 'sharkFish', 'seahorse1', 'starfish2', 'turtles1']}
    w = scene_balanced_sampler_weights(ds_tr.files, rare); n_samples = len(ds_tr)
    if a.ext_root:  # act006 外部分割資料（255＝ignore），佔總抽樣權重 ext_share
        # exp060：--ext-pseudo-dir 給 exp051 的 teacher 偽標籤 → 外部樣本的硬標籤改用 teacher 遮罩（稀有類仍以外部遮罩為準，ExtSeg 內處理）；teacher 不吃軟標籤
        ds_ext = ExtSeg(a.ext_root, hw=a.hw, train=True, datasets=a.ext_datasets, pseudo_dir=a.ext_pseudo_dir, drop_cls=a.ext_drop_cls)
        w, mult = mix_weights(w, ds_ext.sampler_weights(a.ext_rare_boost), a.ext_share)
        ds_tr = ConcatDataset([ds_tr, ds_ext]); n_samples = int(n_samples * mult)
        P(f'ext imgs={len(ds_ext)} (rare {int(ds_ext.rare.sum())}) from {a.ext_root} share={a.ext_share} rare_boost={a.ext_rare_boost} datasets={a.ext_datasets} drop_cls={a.ext_drop_cls}')
    sampler = WeightedRandomSampler(w, num_samples=n_samples, replacement=True)
    dl_tr = DataLoader(ds_tr, batch_size=a.bs, sampler=sampler, num_workers=a.workers, pin_memory=True, drop_last=True, persistent_workers=a.workers > 0)
    dl_va = None
    if va_scenes:
        abf = (lambda aa, bb: bb % (5 * a.val_ab_stride) == 0) if a.val_ab_stride > 1 else None
        ds_va = EURSPairs(a.root, a.side, va_scenes, hw=a.hw, train=False, ab_filter=abf)
        dl_va = DataLoader(ds_va, batch_size=max(1, a.bs), shuffle=False, num_workers=max(1, a.workers // 2), pin_memory=True)
        P(f'train imgs={len(ds_tr)} scenes={tr_scenes}\nval imgs={len(ds_va)} scenes={va_scenes}')
    else:
        P(f'train imgs={len(ds_tr)} (no val)')

    model = build_teacher(a.backbone, pretrained=not a.no_pretrained, drop_path=a.drop_path, out_indices=a.out_indices,
                          freeze=a.freeze_backbone, lora_r=a.lora_r, epfl_ckpt=a.epfl_ckpt, m2f_ckpt=a.m2f_ckpt).to(device).to(memory_format=torch.channels_last)
    is_m2f = getattr(model, 'is_m2f', False)
    if is_m2f:
        P(f'm2f: Mask2Former 原生集合預測損失、整體微調（--class-weights／--w-dice／--w-lovasz／--freeze-backbone／--lora-r／--out-indices／--drop-path 不作用）clip_grad={a.clip_grad}')
    n_all = sum(p.numel() for p in model.parameters()); n_bb = sum(p.numel() for p in model.backbone.parameters())
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    P(f'params total={n_all/1e6:.2f}M backbone={n_bb/1e6:.2f}M trainable={n_tr/1e6:.2f}M freeze={a.freeze_backbone} lora_r={a.lora_r} out_indices={a.out_indices}')
    seg_loss = SegLoss(a.class_weights, a.w_dice, a.w_lovasz).to(device)
    opt = torch.optim.AdamW(model.param_groups(a.lr_backbone, a.lr, a.wd), betas=(0.9, 0.999))
    steps_per_epoch = len(dl_tr) // a.accum; total = a.epochs * steps_per_epoch; warm = min(300, total // 20 + 1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total)))))
    ema = EMA(model, a.ema)
    start_ep, best = 0, -1.0
    if a.resume and os.path.exists(a.resume):
        ck = torch.load(a.resume, map_location='cpu'); model.load_state_dict(ck['model']); ema.m.load_state_dict(ck['ema'])
        opt.load_state_dict(ck['opt']); start_ep = ck['epoch'] + 1; best = ck.get('best', -1.0); P(f'resumed from {a.resume} epoch {start_ep}')
    elif a.init_ckpt:  # exp085：從既有 teacher 的 EMA 權重起始（新 optimizer／排程），用於 EURS-only 鞏固
        ck = torch.load(a.init_ckpt, map_location='cpu'); model.load_state_dict(ck['ema']); ema.m.load_state_dict(ck['ema'])
        P(f'init from {a.init_ckpt} (ema, epoch {ck.get("epoch")})')

    end_ep = min(a.epochs, a.stop_epoch) if a.stop_epoch > 0 else a.epochs
    for ep in range(start_ep, end_ep):
        model.train(); t0 = time.time(); agg = {'seg': 0.0, 'n': 0}
        opt.zero_grad(set_to_none=True)
        for bi, batch in enumerate(dl_tr):
            x, m = batch['x'].to(device, non_blocking=True).to(memory_format=torch.channels_last), batch['m'].to(device, non_blocking=True)
            if is_m2f:   # exp078：Mask2Former 原生損失（網路本體 bf16，匹配與損失 fp32）
                with _autocast(device):
                    loss = model.loss(x, m) / a.accum
            else:
                with _autocast(device):
                    seg = model(x)
                loss = seg_loss(seg.float(), m) / a.accum
            loss.backward()
            if (bi + 1) % a.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip_grad)
                opt.step(); opt.zero_grad(set_to_none=True); sched.step(); ema.update(model)
                if is_m2f and device.type == 'cuda' and ep == start_ep and bi + 1 == 2 * a.accum:   # 第 2 個 step（Adam 狀態已配置）後的峰值，監控時讀這行
                    P(f'm2f: peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB allocated / {torch.cuda.max_memory_reserved() / 2**30:.1f} GiB reserved (bs {a.bs}, accum {a.accum})')
            agg['seg'] += loss.item() * a.accum; agg['n'] += 1
            if bi % 50 == 0:
                P(f'ep {ep} it {bi}/{len(dl_tr)} loss_seg {loss.item()*a.accum:.4f} lr {sched.get_last_lr()[0]:.2e} {(bi+1)/(time.time()-t0+1e-6):.2f} it/s')
            if a.dry and bi >= 1:
                break
        rec = {'exp': a.exp, 'fold': a.fold, 'epoch': ep, 'loss_seg': agg['seg'] / max(1, agg['n']), 'time_s': time.time() - t0}
        if dl_va is not None:
            s = evaluate(ema.m, dl_va, device, max_batches=(2 if a.dry else (a.val_max_batches or None)))
            rec.update({f'val_{k}': v for k, v in s.items() if k not in ('present',)})
            score = s['mIoU']
            if score > best:
                best = score; torch.save({'model': model.state_dict(), 'ema': ema.m.state_dict(), 'epoch': ep, 'best': best, 'args': _args(a), **_ck_extra(model)}, os.path.join(a.out, 'best.pt'))
            P(f'[val] ep {ep} mIoU {s["mIoU"]:.4f} F1 {s["F1"]:.4f} Qseg {s["Qseg"]:.4f} cov {s["coverage"]:.2f} IoU/cls {s["IoU_per_class"]} best_mIoU {best:.4f}')
        torch.save({'model': model.state_dict(), 'ema': ema.m.state_dict(), 'opt': opt.state_dict(), 'epoch': ep, 'best': best, 'args': _args(a), **_ck_extra(model)}, os.path.join(a.out, 'last.pt'))
        if dl_va is None:
            torch.save({'model': model.state_dict(), 'ema': ema.m.state_dict(), 'epoch': ep, 'best': best, 'args': _args(a), **_ck_extra(model)}, os.path.join(a.out, 'best.pt'))
        with open(os.path.join(a.out, 'metrics.jsonl'), 'a') as f:
            f.write(json.dumps(rec) + '\n')
        P(f'[epoch {ep} done] {json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items() if not isinstance(v, list)})}')
        if a.dry:
            break
    P(f'[{time.strftime("%F %T")}] DONE best_mIoU={best:.4f}')


def load_teacher(ckpt, device, use_ema=True):
    ck = torch.load(ckpt, map_location='cpu')
    args = ck['args']
    model = build_teacher(args['backbone'], pretrained=False, drop_path=0.0, out_indices=args.get('out_indices'),
                          freeze=args.get('freeze_backbone', False), lora_r=args.get('lora_r', 0),
                          m2f_ckpt=args.get('m2f_ckpt', ''), hf_config=ck.get('hf_config'))
    model.load_state_dict(ck['ema' if use_ema else 'model'])
    return model.to(device).eval(), args


@torch.no_grad()
def cmd_predict(a):
    """對任一資料夾的 PNG／JPG 產偽標籤：硬標籤 PNG（原尺寸，nearest 縮回）＋可選 1/4 解析 fp16 softmax（.npy，供蒸餾）。"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = [load_teacher(p, device)[0] for p in a.ckpt.split(',')]   # exp057：逗號分隔多個 checkpoint → softmax 平均（集成）
    model = models[0]
    print(f'loaded {len(models)} teacher checkpoint(s)', flush=True)
    H, W = a.hw
    os.makedirs(a.out_masks, exist_ok=True)
    if a.out_logits:
        os.makedirs(a.out_logits, exist_ok=True)
    files = sorted(f for f in os.listdir(a.degraded_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg')))
    sk, sn = (int(v) for v in a.shard.split('/'))   # exp051：多台機器分片（第 k 片取 files[k::n]）
    files = files[sk::sn]
    rest_of = {}
    if a.avg_dir:  # exp039：同名復原圖（任意副檔名）的機率與原圖機率加權平均
        rest_of = {os.path.splitext(f)[0]: f for f in os.listdir(a.avg_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))}
        missing = [f for f in files if os.path.splitext(f)[0] not in rest_of]
        if missing:
            raise FileNotFoundError(f'{len(missing)} 張缺復原圖，例如 {missing[0]}')

    def run(x):
        prob = None
        with _autocast(device):
            for m in models:
                logit = m(x).float()
                if a.tta:
                    logit = 0.5 * (logit + torch.flip(m(torch.flip(x, [3])).float(), [3]))
                p = logit.softmax(1)
                prob = p if prob is None else prob + p
        return prob / len(models)

    def run_tiled(xb, n):
        """exp064：xb＝整張拉伸到 (k·H, k·W) 的 batch → 切 n×n 個 (H,W) 等距重疊窗各跑 run → 機率平均回大網格 → bilinear 縮回 (H,W)。"""
        BH, BW = xb.shape[-2:]
        acc = torch.zeros(xb.shape[0], 5, BH, BW, device=device); cnt = torch.zeros(1, 1, BH, BW, device=device)
        ys = np.linspace(0, BH - H, n).round().astype(int); xs_ = np.linspace(0, BW - W, n).round().astype(int)
        for y0 in ys:
            for x0 in xs_:
                acc[:, :, y0:y0 + H, x0:x0 + W] += run(xb[:, :, y0:y0 + H, x0:x0 + W].contiguous(memory_format=torch.channels_last))
                cnt[:, :, y0:y0 + H, x0:x0 + W] += 1
        return F.interpolate(acc / cnt, size=(H, W), mode='bilinear', align_corners=False)

    if a.tile2:
        assert not a.aspect_norm, '--tile2 只支援直接拉伸（不與 --aspect-norm 併用）'
    t0 = time.time(); stats = {'n': 0, 'px_per_class': np.zeros(5, dtype=np.int64)}
    for i in range(0, len(files), a.bs):
        names = files[i:i + a.bs]; xs, rs, xbs, sizes, infos = [], [], [], [], []
        for f in names:
            im = Image.open(os.path.join(a.degraded_dir, f)).convert('RGB'); sizes.append(im.size)
            imr, info = aspect_prep(im, H, W, a.aspect_norm); infos.append(info)
            xs.append(torch.from_numpy(np.asarray(imr, dtype=np.float32) / 255.0).permute(2, 0, 1))
            if a.tile2:
                xbs.append(torch.from_numpy(np.asarray(im.resize((a.tile2 * W, a.tile2 * H), Image.BILINEAR), dtype=np.float32) / 255.0).permute(2, 0, 1))
            if a.avg_dir:
                rim = Image.open(os.path.join(a.avg_dir, rest_of[os.path.splitext(f)[0]])).convert('RGB')
                rim = rim if rim.size == im.size else rim.resize(im.size, Image.BILINEAR)
                rs.append(torch.from_numpy(np.asarray(aspect_prep(rim, H, W, a.aspect_norm)[0], dtype=np.float32) / 255.0).permute(2, 0, 1))
        prob = run(torch.stack(xs).to(device).to(memory_format=torch.channels_last))
        if a.tile2:  # exp064：單次機率與 k 倍切窗機率加權平均
            prob = (1.0 - a.tile_w) * prob + a.tile_w * run_tiled(torch.stack(xbs).to(device), a.tile_n)
        if a.avg_dir:
            prob = (1.0 - a.avg_w) * prob + a.avg_w * run(torch.stack(rs).to(device).to(memory_format=torch.channels_last))
        pm = prob.argmax(1).cpu().numpy().astype(np.uint8)
        if a.out_logits:
            if a.aspect_norm:
                small = torch.stack([aspect_undo_prob(prob[j], infos[j], (H // 4, W // 4)) for j in range(len(names))]).half().cpu().numpy()
            else:
                small = F.interpolate(prob, scale_factor=0.25, mode='area').half().cpu().numpy()
        for j, f in enumerate(names):
            stem = os.path.splitext(f)[0]
            aspect_undo_mask(pm[j], infos[j], sizes[j]).save(os.path.join(a.out_masks, stem + '.png'))
            if a.out_logits:
                np.save(os.path.join(a.out_logits, stem + '.npy'), small[j])
            stats['n'] += 1; stats['px_per_class'] += np.bincount(pm[j].ravel(), minlength=5)
        if (i // a.bs) % 20 == 0:
            print(f'{stats["n"]}/{len(files)} {(time.time()-t0):.0f}s', flush=True)
    frac = stats['px_per_class'] / max(1, stats['px_per_class'].sum())
    summary = {'n': stats['n'], 'class_pixel_frac': [round(float(v), 4) for v in frac], 'tta': a.tta, 'aspect_norm': a.aspect_norm, 'hw': a.hw, 'ckpt': a.ckpt,
               'avg_dir': a.avg_dir, 'avg_w': a.avg_w if a.avg_dir else None, 'tile2': a.tile2, 'tile_n': a.tile_n if a.tile2 else None, 'tile_w': a.tile_w if a.tile2 else None}
    summary['shard'] = a.shard
    print(json.dumps(summary)); json.dump(summary, open(os.path.join(a.out_masks, '_summary.json' if sn == 1 else f'_summary_shard{sk}of{sn}.json'), 'w'))


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('train')
    t.add_argument('--root', required=True); t.add_argument('--side', type=int, default=768)
    t.add_argument('--fold', type=int, default=0); t.add_argument('--hw', type=int, nargs=2, default=[768, 1024])
    t.add_argument('--backbone', default='convnext_small.fb_in22k_ft_in1k_384'); t.add_argument('--no-pretrained', action='store_true')
    t.add_argument('--drop-path', type=float, default=0.1)
    t.add_argument('--out-indices', type=int, nargs='*', default=None, help='timm features_only 的 out_indices（預設 0 1 2 3＝ConvNeXt 四個 stage；ViT-L 用 5 11 17 23）')
    t.add_argument('--freeze-backbone', action='store_true', help='骨幹不更新，只訓 FPN／頭（exp038 候選）')
    t.add_argument('--lora-r', type=int, default=0, help='>0：凍結骨幹，對 attention 的 q、v 加 rank r 的 LoRA（lr 用 --lr-backbone）')
    t.add_argument('--epochs', type=int, default=20); t.add_argument('--stop-epoch', type=int, default=0, help='>0：排程仍按 --epochs，只跑到第 N 個 epoch 就停'); t.add_argument('--bs', type=int, default=6); t.add_argument('--accum', type=int, default=1)
    t.add_argument('--lr', type=float, default=1e-3, help='解碼器／頭'); t.add_argument('--lr-backbone', type=float, default=1e-4)
    t.add_argument('--wd', type=float, default=0.05); t.add_argument('--workers', type=int, default=12)
    t.add_argument('--w-dice', type=float, default=1.0); t.add_argument('--w-lovasz', type=float, default=0.0)
    t.add_argument('--class-weights', type=float, nargs=5, default=[1.0, 1.0, 1.5, 3.0, 2.0])
    t.add_argument('--rare-boost', type=float, default=2.0)
    t.add_argument('--ext-root', default=''); t.add_argument('--ext-share', type=float, default=0.35)
    t.add_argument('--ext-rare-boost', type=float, default=4.0); t.add_argument('--ext-datasets', nargs='*', default=None)
    t.add_argument('--ext-drop-cls', type=int, nargs='*', default=[], help='exp086：外部遮罩中這些類別改 ignore（0 1＝丟外部 bg／coral，只留 fish／elasmo／other 實例監督）')
    t.add_argument('--epfl-ckpt', default=EPFL_DPT_URL, help='exp067：--backbone epfl_dpt 時的 EPFL CoralscapesV2 DINOv3-L DPT 權重（本機路徑或 URL，URL 會下載到 ~/.cache/eurs）')
    t.add_argument('--m2f-ckpt', default='', help='exp078：--backbone m2f_* 時的 HF Mask2Former 權重（hub id 或本機 snapshot 資料夾）；空白＝m2f_swin_l→facebook/mask2former-swin-large-ade-semantic、m2f_swin_t→tiny')
    t.add_argument('--clip-grad', type=float, default=3.0, help='梯度全模型 L2 範數上限（預設 3.0＝既有行為；exp078 Mask2Former 用 0.01，同原論文配方）')
    t.add_argument('--ext-pseudo-dir', default='', help='exp060：exp051 的外部偽標籤根目錄（ext_masks/<ds>/、ext_logits/<ds>/）；外部樣本的硬標籤改用 teacher 遮罩，稀有類仍用外部遮罩')
    t.add_argument('--val-ab-stride', type=int, default=2); t.add_argument('--val-max-batches', type=int, default=0)
    t.add_argument('--ema', type=float, default=0.999); t.add_argument('--out', required=True); t.add_argument('--exp', default='expXXX')
    t.add_argument('--seed', type=int, default=0); t.add_argument('--resume', default=''); t.add_argument('--init-ckpt', default='', help='exp085：從既有 teacher checkpoint 的 EMA 權重起始（optimizer 與排程重來）'); t.add_argument('--dry', action='store_true')
    t.set_defaults(fn=cmd_train)
    p = sub.add_parser('predict')
    p.add_argument('--ckpt', required=True); p.add_argument('--hw', type=int, nargs=2, default=[768, 1024])
    p.add_argument('--degraded-dir', required=True); p.add_argument('--out-masks', required=True); p.add_argument('--out-logits', default='')
    p.add_argument('--bs', type=int, default=4); p.add_argument('--tta', action='store_true')
    p.add_argument('--aspect-norm', action='store_true', help='exp017：直幅先轉 90°、過寬補邊到 W:H 再推論（預設關＝直接拉伸）')
    p.add_argument('--avg-dir', default='', help='exp039：同名復原圖資料夾；給了就把復原圖的機率與原圖機率加權平均（預設關）')
    p.add_argument('--avg-w', type=float, default=0.5, help='復原圖機率的權重（1.0＝只用復原圖）')
    p.add_argument('--shard', default='0/1', help='exp051：k/n，只處理排序後第 k 片（files[k::n]），多台機器分攤外部 3.2 萬張')
    p.add_argument('--tile2', type=int, default=0, help='exp064：另把整張拉到 k 倍（k·H, k·W）切 tile-n×tile-n 個 (H,W) 重疊窗推論，機率與單次平均（0＝關）')
    p.add_argument('--tile-n', type=int, default=3, help='切窗每邊個數（預設 3，2 倍時窗距 384／512）')
    p.add_argument('--tile-w', type=float, default=0.5, help='切窗機率的權重（1.0＝只用切窗）')
    p.set_defaults(fn=cmd_predict)
    a = ap.parse_args(); a.fn(a)


if __name__ == '__main__':
    main()
