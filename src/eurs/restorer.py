"""t1 專用大型復原網（act007／exp020）：NAFNet（Chen et al., ECCV 2022；官方實作 MIT 授權，此為依論文重寫）＋可選每影像標準化輸入。
不受 12 GFLOPs 限制；在評分尺寸（cache1024）上以隨機裁切訓練、全解析驗證與推論，輸出與輸入同尺寸（＝官方評分尺寸，不再縮放）。
驗證口徑：leave-scene-out 折的 holdout，b∈{10,…,100}（同 exp012／exp017 的子集）；PSNR／SSIM／sharpness 用 metrics.py（同 Evaluator）。

用法（箱上）：
  python -I eurs/restorer.py train --root ~/eurs/processed --fold 0 --iters 40000 --bs 4 --crop 512 --out ~/eurs/runs/EURS_exp020_f0 --exp exp020
  python -I eurs/restorer.py eval --kind student --ckpt exp006_f0_best.pt --root ~/eurs/processed --fold 0 --out base.json   # 同口徑基準
  python -I eurs/restorer.py predict --ckpt ~/eurs/runs/EURS_exp020_f-1/best.pt --degraded-dir ~/eurs/processed/cache1024/test/degraded --out-dir restored/
"""
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eurs import folds as F_
from eurs.data import list_degraded, parse_name
from eurs.losses import restoration_loss
from eurs.metrics import psnr, sharpness_ratio, ssim


class LayerNorm2d(nn.Module):
    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c)); self.bias = nn.Parameter(torch.zeros(c)); self.eps = eps

    def forward(self, x):
        x32 = x.float(); mu = x32.mean(1, keepdim=True); var = (x32 - mu).pow(2).mean(1, keepdim=True)
        y = (x32 - mu) * torch.rsqrt(var + self.eps)
        return (y * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)).to(x.dtype)


class NAFBlock(nn.Module):
    def __init__(self, c, dw_expand=2, ffn_expand=2):
        super().__init__()
        dw, ffn = c * dw_expand, c * ffn_expand
        self.norm1, self.norm2 = LayerNorm2d(c), LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, dw, 1); self.conv2 = nn.Conv2d(dw, dw, 3, padding=1, groups=dw); self.conv3 = nn.Conv2d(dw // 2, c, 1)
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dw // 2, dw // 2, 1))
        self.conv4 = nn.Conv2d(c, ffn, 1); self.conv5 = nn.Conv2d(ffn // 2, c, 1)
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1)); self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    @staticmethod
    def _gate(x):
        a, b = x.chunk(2, 1)
        return a * b

    def forward(self, inp):
        x = self._gate(self.conv2(self.conv1(self.norm1(inp))))
        x = self.conv3(x * self.sca(x))
        y = inp + x * self.beta
        x = self.conv5(self._gate(self.conv4(self.norm2(y))))
        return y + x * self.gamma


A_MAX, B_MAX = 10, 100   # 檔名 {scene}_{a}_{b}：a＝Jerlov 水型 1–10，b＝深度 5–100（步長 5）


class NAFNet(nn.Module):
    """U 形 NAFNet；in_norm=True 時第一層另吃每影像標準化的輸入（(x−μ)/σ，逐通道），幫助極暗影像。輸出＝輸入＋殘差。
    meta=True（exp034）：以檔名 (a,b) 查表得每組退化的 3×4 線性反轉換 J0＝M_ab·[x;1]（訓練場景最小平方擬合，存成 buffer 隨權重保存），
    第一層另吃 J0 與 a／b 常數平面，輸出改為 J0＋殘差。"""

    def __init__(self, width=32, enc=(2, 2, 4, 8), mid=12, dec=(2, 2, 2, 2), in_norm=True, meta=False):
        super().__init__()
        self.in_norm, self.meta = in_norm, meta
        if meta:
            eye = torch.zeros(A_MAX + 1, B_MAX // 5 + 1, 3, 4); eye[..., :, :3] = torch.eye(3)
            self.register_buffer('ab_table', eye)                                   # 未擬合前＝恆等
        self.intro = nn.Conv2d((6 if in_norm else 3) + (5 if meta else 0), width, 3, padding=1)
        self.ending = nn.Conv2d(width, 3, 3, padding=1)
        self.encoders, self.downs, self.ups, self.decoders = nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        c = width
        for n in enc:
            self.encoders.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))
            self.downs.append(nn.Conv2d(c, 2 * c, 2, stride=2)); c *= 2
        self.middle = nn.Sequential(*[NAFBlock(c) for _ in range(mid)])
        for n in dec:
            self.ups.append(nn.Sequential(nn.Conv2d(c, 2 * c, 1, bias=False), nn.PixelShuffle(2))); c //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))
        self.mult = 2 ** len(enc)
        if meta:  # 起點＝J0（殘差從 0 開始學）
            nn.init.zeros_(self.ending.weight); nn.init.zeros_(self.ending.bias)

    def ab_prior(self, x, a, b):
        """x (B,3,H,W)、a／b (B,) long → J0＝clamp(M_ab·[x;1])。"""
        M = self.ab_table[a, b // 5].to(x.dtype)                                    # (B,3,4)
        return (torch.einsum('bij,bjhw->bihw', M[:, :, :3], x) + M[:, :, 3, None, None]).clamp(0, 1)

    def forward(self, inp, a=None, b=None):
        H, W = inp.shape[-2:]
        ph, pw = (-H) % self.mult, (-W) % self.mult
        x0 = F.pad(inp, (0, pw, 0, ph), mode='replicate') if (ph or pw) else inp
        feats = [x0]
        if self.in_norm:
            mu = x0.mean((2, 3), keepdim=True); sd = x0.std((2, 3), keepdim=True).clamp_min(1e-3)
            feats.append((x0 - mu) / sd)
        base = x0
        if self.meta:
            base = self.ab_prior(x0, a, b)
            planes = torch.stack([a.to(x0.dtype) / A_MAX, b.to(x0.dtype) / B_MAX], 1)[:, :, None, None].expand(-1, -1, *x0.shape[-2:])
            feats += [base, planes]
        x = self.intro(torch.cat(feats, 1))
        skips = []
        for e, d in zip(self.encoders, self.downs):
            x = e(x); skips.append(x); x = d(x)
        x = self.middle(x)
        for u, dcd, s in zip(self.ups, self.decoders, skips[::-1]):
            x = dcd(u(x) + s)
        return (self.ending(x) + base)[..., :H, :W]


PRESETS = {'naf32': dict(width=32, enc=(2, 2, 4, 8), mid=12, dec=(2, 2, 2, 2)),
           'naf48': dict(width=48, enc=(2, 2, 4, 8), mid=12, dec=(2, 2, 2, 2)),   # exp056 候選：約 2.25× 算力
           'naf64': dict(width=64, enc=(2, 2, 4, 8), mid=12, dec=(2, 2, 2, 2)),   # exp056：NAFNet 寬 64（約 4× 算力、約 116M 參數）
           'naf16': dict(width=16, enc=(1, 1, 1, 1), mid=1, dec=(1, 1, 1, 1))}  # naf16 只供 CPU 煙霧測試


def build(preset='naf32', in_norm=True, meta=False):
    return NAFNet(**PRESETS[preset], in_norm=in_norm, meta=meta)


def _ab_stats(args):
    """單張退化圖 → 該 (a,b) 的最小平方累加量 XᵀX (4×4)、XᵀY (4×3)；X＝[I_r,I_g,I_b,1]、Y＝GT。"""
    path, gt_path, n_px, seed = args
    x = Image.open(path).convert('RGB'); y = Image.open(gt_path).convert('RGB')
    if y.size != x.size:
        y = y.resize(x.size, Image.BILINEAR)
    x = np.asarray(x, dtype=np.float64).reshape(-1, 3) / 255.0; y = np.asarray(y, dtype=np.float64).reshape(-1, 3) / 255.0
    idx = np.random.default_rng(seed).choice(len(x), size=min(n_px, len(x)), replace=False)
    X = np.concatenate([x[idx], np.ones((len(idx), 1))], 1); Y = y[idx]
    _, a, b = parse_name(path)
    return a, b, X.T @ X, X.T @ Y


def fit_ab_table(root, scenes, n_px=8192, workers=16, ridge=1e-4):
    """訓練場景上逐 (a,b) 擬合 J≈M·[I;1]（每通道吃三個通道＋偏置，直接最小化 GT 空間誤差＝PSNR 的口徑）。回傳 (11,21,3,4) tensor。"""
    from multiprocessing import Pool
    files = list_degraded(root, 1024, 'train', scenes)
    gt = lambda p: os.path.join(root, 'cache1024', 'train', 'gt_images', parse_name(p)[0] + '.png')  # noqa: E731
    acc = {}
    with Pool(workers) as pool:
        for a, b, xtx, xty in pool.imap_unordered(_ab_stats, [(p, gt(p), n_px, i) for i, p in enumerate(files)], chunksize=8):
            s = acc.setdefault((a, b), [np.zeros((4, 4)), np.zeros((4, 3))]); s[0] += xtx; s[1] += xty
    T = torch.zeros(A_MAX + 1, B_MAX // 5 + 1, 3, 4); T[..., :, :3] = torch.eye(3)
    for (a, b), (xtx, xty) in acc.items():
        M = np.linalg.solve(xtx + ridge * np.trace(xtx) / 4 * np.eye(4), xty)       # (4,3)
        T[a, b // 5] = torch.from_numpy(M.T.astype(np.float32))
    return T, len(acc)


class RestorePairs(Dataset):
    """cache1024 的退化圖與場景 GT（GT 縮到退化圖尺寸，同 data.EURSPairs）；訓練：隨機裁切 crop×crop＋水平翻轉。"""

    def __init__(self, root, scenes, crop=512, train=True, ab_filter=None):
        self.root, self.crop, self.train = root, crop, train
        self.files = list_degraded(root, 1024, 'train', scenes, ab_filter)
        self._gt = {}

    def __len__(self):
        return len(self.files)

    def gt(self, scene, size):
        if scene not in self._gt:
            self._gt[scene] = Image.open(os.path.join(self.root, 'cache1024', 'train', 'gt_images', scene + '.png')).convert('RGB')
        g = self._gt[scene]
        return g if g.size == size else g.resize(size, Image.BILINEAR)

    def __getitem__(self, i):
        path = self.files[i]; scene, a, b = parse_name(path)
        x = Image.open(path).convert('RGB'); y = self.gt(scene, x.size)
        x = np.asarray(x, dtype=np.float32) / 255.0; y = np.asarray(y, dtype=np.float32) / 255.0
        if self.train:
            h, w = x.shape[:2]; ch, cw = min(self.crop, h), min(self.crop, w)
            t, l = random.randint(0, h - ch), random.randint(0, w - cw)
            x, y = x[t:t + ch, l:l + cw], y[t:t + ch, l:l + cw]
            if random.random() < 0.5:
                x, y = x[:, ::-1], y[:, ::-1]
        return {'x': torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1), 'y': torch.from_numpy(np.ascontiguousarray(y)).permute(2, 0, 1),
                'name': os.path.splitext(os.path.basename(path))[0], 'a': a, 'b': b}


class ExtRestorePairs(Dataset):
    """exp056：外部乾淨水下影像（act006 的 ext_index.csv）→ 以 exp008 的 (a,b) 仿射再退化成訓練配對（GT＝再退化前的外部影像）。
    只取 a ≤ amax 的 (a,b) 格子：exp008 顯示 a≤3 的退化近乎全域仿射，redegrade() 才像真的；a≥4 的空間變化背散射它模擬不了。
    原生解析度隨機裁切 crop×crop（短邊不足 crop 時先等比放大到 crop）＋水平翻轉。"""

    def __init__(self, ext_root, crop=512, amax=3, datasets=None):
        import csv
        from eurs.data import load_ab_table
        rows = list(csv.DictReader(open(os.path.join(ext_root, 'ext_index.csv'))))
        if datasets:
            rows = [r for r in rows if r['ds'] in datasets]
        self.root, self.crop = ext_root, crop
        self.files = [r['path_img'] for r in rows]
        self.ab = [r for r in load_ab_table() if 1 <= r[0] <= amax]
        if not self.files or not self.ab:
            raise ValueError(f'ExtRestorePairs: files={len(self.files)} ab_rows={len(self.ab)} (amax={amax})')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        from eurs.data import redegrade
        im = Image.open(os.path.join(self.root, self.files[i])).convert('RGB')
        w, h = im.size; s = self.crop / max(1, min(w, h))
        if s > 1:
            im = im.resize((max(self.crop, round(w * s)), max(self.crop, round(h * s))), Image.BICUBIC)
        y = np.asarray(im, dtype=np.float32) / 255.0
        H, W = y.shape[:2]; ch, cw = min(self.crop, H), min(self.crop, W)
        t, l = random.randint(0, H - ch), random.randint(0, W - cw)
        y = y[t:t + ch, l:l + cw]
        if random.random() < 0.5:
            y = y[:, ::-1]
        yt = torch.from_numpy(np.ascontiguousarray(y)).permute(2, 0, 1)
        row = self.ab[random.randrange(len(self.ab))]
        xt = redegrade(yt, row, np.random.default_rng(random.getrandbits(32)))
        return {'x': xt.contiguous(), 'y': yt, 'name': 'ext_' + os.path.splitext(os.path.basename(self.files[i]))[0], 'a': row[0], 'b': row[1]}


class MixedPairs(Dataset):
    """exp056：以 ext_share 的機率改抽一筆外部再退化配對（隨機索引），其餘回傳 dev 配對；長度＝dev，使 iters 口徑與 exp034 相同。
    DataLoader worker 的 python random 由 torch 逐 worker 播種，各 worker 抽樣獨立。"""

    def __init__(self, dev, ext, share):
        self.dev, self.ext, self.share = dev, ext, share

    def __len__(self):
        return len(self.dev)

    def __getitem__(self, i):
        if self.ext is not None and self.share > 0 and random.random() < self.share:
            return self.ext[random.randrange(len(self.ext))]
        return self.dev[i]


def _autocast(device):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == 'cuda'))


def load_restorer(ckpt, device, use_ema=True):
    ck = torch.load(ckpt, map_location='cpu')
    args = ck['args']
    model = build(args['preset'], in_norm=not args.get('no_in_norm', False), meta=args.get('meta', False))
    model.load_state_dict(ck['ema' if use_ema else 'model'])
    return model.to(device).eval()


def student_restore_fn(ckpt, preset, device, hw=(768, 1024)):
    """exp006／exp009 學生的 t1 路徑（同 infer_t1.py）：拉伸到 hw → J → bilinear 縮回原尺寸。"""
    from eurs.model import build_from_ckpt
    ck = torch.load(ckpt, map_location='cpu'); m = build_from_ckpt(ck, preset); m.load_state_dict(ck['ema']); m.to(device).eval()
    H, W = hw

    def fn(x, name=None):  # x (1,3,h,w) 0–1（由 uint8 而來，可無損轉回 PIL）
        h, w = x.shape[-2:]
        im = Image.fromarray((x[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).resize((W, H), Image.BILINEAR)
        xs = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)[None].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
            J, _ = m(xs)
        J = J.float().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
        return np.asarray(Image.fromarray((J * 255).round().astype(np.uint8)).resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0
    return fn


@torch.no_grad()
def evaluate(fn, ds, max_n=0):
    """fn：(1,3,h,w) tensor、name → (h,w,3) numpy 0–1。回傳 PSNR／SSIM／sharpness 平均（逐張）、PSNRnorm（對平均 PSNR，hi=34）與逐場景 PSNR。"""
    P, S, Sh, sc = [], [], [], []
    for i in range(len(ds) if not max_n else min(max_n, len(ds))):
        it = ds[i]; x = it['x'][None]; y = it['y'].permute(1, 2, 0).numpy()
        out = fn(x, it['name'])
        P.append(psnr(out, y)); S.append(ssim(out, y)); Sh.append(sharpness_ratio(out, y)); sc.append(it['name'].rsplit('_', 2)[0])
    p = float(np.mean(P))
    per_scene = {s: round(float(np.mean([v for v, t in zip(P, sc) if t == s])), 2) for s in dict.fromkeys(sc)}
    return {'n': len(P), 'PSNR': p, 'SSIM': float(np.mean(S)), 'sharpness': float(np.mean(Sh)),
            'PSNRnorm': float(np.clip((p - 15) / 19, 0, 1)), 'rest_term': 0.45 * float(np.clip((p - 15) / 19, 0, 1)) + 0.45 * float(np.mean(S)) + 0.10 * float(np.mean(Sh)),
            'PSNR_per_scene': per_scene}


def _ab_of(name, device):
    _, a, b = parse_name(name + '.png')
    return torch.tensor([a], device=device), torch.tensor([b], device=device)


def model_fn(model, device):
    def fn(x, name=None):
        ab = _ab_of(name, device) if getattr(model, 'meta', False) else (None, None)
        with _autocast(device):
            out = model(x.to(device), *ab).float().clamp(0, 1)
        return (out[0].permute(1, 2, 0).cpu().numpy() * 255).round() / 255.0  # 與存成 8-bit 影像後評分一致
    return fn


def ab_prior_fn(table, device):
    """只用 (a,b) 查表的線性反轉換（無網路）＝ Meta 條件化的地板讀數。"""
    m = NAFNet(width=8, enc=(), mid=0, dec=(), in_norm=False, meta=True).to(device); m.ab_table.copy_(table)

    def fn(x, name=None):
        out = m.ab_prior(x.to(device), *_ab_of(name, device))
        return (out[0].permute(1, 2, 0).cpu().numpy() * 255).round() / 255.0
    return fn


class EMA:
    def __init__(self, model, decay):
        import copy
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


def cmd_train(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log = open(os.path.join(a.out, 'train.log'), 'a')

    def P(*s):
        msg = ' '.join(str(t) for t in s); print(msg, flush=True); log.write(msg + '\n'); log.flush()
    args = {k: v for k, v in vars(a).items() if k != 'fn'}
    P(f'[{time.strftime("%F %T")}] start RESTORER exp={a.exp} fold={a.fold} device={device} args={args}')
    tr, va = F_.split(a.fold)
    ds_dev = RestorePairs(a.root, tr, crop=a.crop, train=True)
    ds_ext = ExtRestorePairs(a.ext_root, crop=a.crop, amax=a.ext_amax, datasets=a.ext_datasets) if a.ext_root else None
    ds_tr = MixedPairs(ds_dev, ds_ext, a.ext_share) if ds_ext is not None else ds_dev
    dl = DataLoader(ds_tr, batch_size=a.bs, shuffle=True, num_workers=a.workers, pin_memory=True, drop_last=True, persistent_workers=a.workers > 0)
    ds_va = RestorePairs(a.root, va, train=False, ab_filter=lambda aa, bb: bb % (5 * a.val_ab_stride) == 0) if va else None
    model = build(a.preset, in_norm=not a.no_in_norm, meta=a.meta).to(device).to(memory_format=torch.channels_last)
    P(f'train imgs={len(ds_dev)} scenes={tr}; val imgs={len(ds_va) if ds_va else 0} scenes={va}; params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')
    P(f'ext imgs={len(ds_ext) if ds_ext is not None else 0} share={a.ext_share if ds_ext is not None else 0} amax={a.ext_amax} ab_rows={len(ds_ext.ab) if ds_ext is not None else 0} root={a.ext_root or "-"}')
    if a.meta:
        t1 = time.time(); table, n_ab = fit_ab_table(a.root, tr, workers=a.workers); model.ab_table.copy_(table.to(device))
        P(f'ab table fitted on {len(tr)} train scenes: {n_ab} (a,b) cells in {time.time() - t1:.0f}s')
        if ds_va is not None:
            rec = {'it': 0, 'kind': 'ab_prior_only', **evaluate(ab_prior_fn(table, device), ds_va)}
            P('[baseline ab_prior_only] ' + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}))
            with open(os.path.join(a.out, 'metrics.jsonl'), 'a') as f:
                f.write(json.dumps(rec) + '\n')
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.9), weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: min(1.0, (it + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(it, a.iters) / a.iters)))
    ema = EMA(model, a.ema)
    best, it, t0 = -1.0, 0, time.time()
    while it < a.iters:
        for b in dl:
            model.train()
            x = b['x'].to(device, non_blocking=True).to(memory_format=torch.channels_last); y = b['y'].to(device, non_blocking=True)
            ab = (b['a'].to(device), b['b'].to(device)) if a.meta else (None, None)
            with _autocast(device):
                out = model(x, *ab)
            loss = restoration_loss(out.float(), y, w_ssim=a.w_ssim)
            opt.zero_grad(set_to_none=True); loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
            opt.step(); sched.step(); ema.update(model); it += 1
            if it % 100 == 0:
                mem = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == 'cuda' else 0.0
                P(f'it {it}/{a.iters} loss {loss.item():.4f} gn {float(gn):.3f} lr {sched.get_last_lr()[0]:.2e} {(time.time() - t0) / it:.3f}s/it mem {mem:.1f}G')
            last = it >= a.iters
            if it % a.val_every == 0 or last or (a.dry and it >= 2):
                rec = {'it': it, 'loss': round(loss.item(), 5), 'elapsed_min': round((time.time() - t0) / 60, 1)}
                if ds_va is not None:
                    rec.update(evaluate(model_fn(ema.m, device), ds_va, max_n=0 if last else a.val_max))
                score = rec.get('PSNR', it)  # 無驗證折：取最後的 EMA
                ck = {'model': model.state_dict(), 'ema': ema.m.state_dict(), 'it': it, 'args': args, 'val': rec}
                torch.save(ck, os.path.join(a.out, 'last.pt'))
                if score > best:
                    best = score; torch.save(ck, os.path.join(a.out, 'best.pt'))
                with open(os.path.join(a.out, 'metrics.jsonl'), 'a') as f:
                    f.write(json.dumps(rec) + '\n')
                P(f'[val it {it}] ' + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}))
            if last or (a.dry and it >= 2):
                break
        if a.dry and it >= 2:
            break
    P(f'[{time.strftime("%F %T")}] DONE best={best:.4f}')


def cmd_eval(a):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, va = F_.split(a.fold)
    ds = RestorePairs(a.root, va, train=False, ab_filter=lambda aa, bb: bb % (5 * a.val_ab_stride) == 0)
    if a.kind == 'identity':
        fn = lambda x, name=None: x[0].permute(1, 2, 0).numpy()  # noqa: E731
    elif a.kind == 'ab_linear':  # 第 0 折訓練場景擬合 (a,b) 線性反轉換 → holdout（exp034 的分析讀數）
        table, _ = fit_ab_table(a.root, F_.split(a.fold)[0]); fn = ab_prior_fn(table, device)
    elif a.kind == 'student':
        fn = student_restore_fn(a.ckpt, a.preset, device)
    else:
        fn = model_fn(load_restorer(a.ckpt, device), device)
    res = {'kind': a.kind, 'ckpt': a.ckpt, 'fold': a.fold, 'holdout': va, **evaluate(fn, ds, max_n=a.max_n)}
    print(json.dumps(res)); json.dump(res, open(a.out, 'w'), indent=1)


@torch.no_grad()
def cmd_predict(a):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fn = model_fn(load_restorer(a.ckpt, device), device)
    os.makedirs(a.out_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(a.degraded_dir) if f.lower().endswith('.png'))
    t0 = time.time()
    for i, f in enumerate(files):
        x = torch.from_numpy(np.asarray(Image.open(os.path.join(a.degraded_dir, f)).convert('RGB'), dtype=np.float32) / 255.0).permute(2, 0, 1)[None]
        Image.fromarray((fn(x, os.path.splitext(f)[0]) * 255).round().astype(np.uint8)).save(os.path.join(a.out_dir, f))
        if i % 100 == 0:
            print(f'{i}/{len(files)} {time.time() - t0:.0f}s', flush=True)
    print(f'wrote {len(files)} → {a.out_dir}')


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('train')
    t.add_argument('--root', required=True); t.add_argument('--fold', type=int, default=0)
    t.add_argument('--preset', default='naf32'); t.add_argument('--no-in-norm', action='store_true')
    t.add_argument('--meta', action='store_true', help='exp034：以檔名 (a,b) 查表的線性反轉換 J0 為底＋a／b 常數平面輸入（看板 Meta 欄須勾選並揭露）')
    t.add_argument('--iters', type=int, default=40000); t.add_argument('--bs', type=int, default=4); t.add_argument('--crop', type=int, default=512)
    t.add_argument('--lr', type=float, default=5e-4); t.add_argument('--wd', type=float, default=0.0); t.add_argument('--warmup', type=int, default=1000)
    t.add_argument('--w-ssim', type=float, default=0.5); t.add_argument('--clip', type=float, default=1.0); t.add_argument('--ema', type=float, default=0.999)
    t.add_argument('--workers', type=int, default=10); t.add_argument('--val-every', type=int, default=4000)
    t.add_argument('--val-ab-stride', type=int, default=2, help='驗證取 b %% (5×stride)==0（預設 2＝b∈{10,…,100}，同 exp012）')
    t.add_argument('--val-max', type=int, default=150, help='中途驗證只算前 N 張（最後一次算全部）')
    t.add_argument('--out', required=True); t.add_argument('--exp', default='expXXX'); t.add_argument('--seed', type=int, default=0)
    t.add_argument('--ext-root', default='', help='exp056：外部乾淨影像根目錄（含 ext_index.csv）；給了就以 --ext-share 機率混入 a≤--ext-amax 的再退化配對')
    t.add_argument('--ext-share', type=float, default=0.35); t.add_argument('--ext-amax', type=int, default=3)
    t.add_argument('--ext-datasets', nargs='*', default=None)
    t.add_argument('--dry', action='store_true')
    t.set_defaults(fn=cmd_train)
    e = sub.add_parser('eval')
    e.add_argument('--kind', choices=['restorer', 'student', 'identity', 'ab_linear'], required=True); e.add_argument('--ckpt', default='')
    e.add_argument('--preset', default='base768'); e.add_argument('--root', required=True); e.add_argument('--fold', type=int, default=0)
    e.add_argument('--val-ab-stride', type=int, default=2); e.add_argument('--max-n', type=int, default=0); e.add_argument('--out', required=True)
    e.set_defaults(fn=cmd_eval)
    p = sub.add_parser('predict')
    p.add_argument('--ckpt', required=True); p.add_argument('--degraded-dir', required=True); p.add_argument('--out-dir', required=True)
    p.set_defaults(fn=cmd_predict)
    a = ap.parse_args(); a.fn(a)


if __name__ == '__main__':
    main()
