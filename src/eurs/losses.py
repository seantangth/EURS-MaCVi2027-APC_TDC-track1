"""損失：復原 Charbonnier＋(1−SSIM)；分割 CE（類別加權）＋Dice／Lovász-softmax。"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def charbonnier(pred, target, eps=1e-3):
    return torch.sqrt((pred - target) ** 2 + eps * eps).mean()


def _gauss(win=11, sigma=1.5, device='cpu', dtype=torch.float32):
    g = torch.arange(win, dtype=dtype, device=device) - win // 2
    g = torch.exp(-(g ** 2) / (2 * sigma ** 2)); g = g / g.sum()
    return (g[:, None] @ g[None, :])


def ssim(pred, target, win=11, sigma=1.5, data_range=1.0):
    """逐張 SSIM（高斯窗、每通道後平均），回傳 (N,) 張量。"""
    C = pred.shape[1]
    k = _gauss(win, sigma, pred.device, pred.dtype).expand(C, 1, win, win).contiguous()
    mu_x = F.conv2d(pred, k, padding=win // 2, groups=C); mu_y = F.conv2d(target, k, padding=win // 2, groups=C)
    sxx = F.conv2d(pred * pred, k, padding=win // 2, groups=C) - mu_x ** 2
    syy = F.conv2d(target * target, k, padding=win // 2, groups=C) - mu_y ** 2
    sxy = F.conv2d(pred * target, k, padding=win // 2, groups=C) - mu_x * mu_y
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    s = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))
    return s.flatten(1).mean(1)


def restoration_loss(pred, target, w_ssim=0.5):
    return charbonnier(pred, target) + w_ssim * (1.0 - ssim(pred, target).mean())


def _lap_var(J):
    """灰階（0.299／0.587／0.114，同 metrics.sharpness_ratio）4 鄰 Laplacian 的逐張變異數（內部像素），回傳 (N,)。"""
    g = 0.299 * J[:, 0] + 0.587 * J[:, 1] + 0.114 * J[:, 2]
    lap = -4 * g[:, 1:-1, 1:-1] + g[:, :-2, 1:-1] + g[:, 2:, 1:-1] + g[:, 1:-1, :-2] + g[:, 1:-1, 2:]
    return lap.flatten(1).var(1)


def rest_score(pred, target):
    """exp088：競賽的復原分數（逐張）＝0.45·clip((PSNR−15)/20, 0, 1)＋0.45·SSIM＋0.10·min(1, LapVar(out)/LapVar(gt))，回傳 (N,)。可微分（clip 與 min 外為 0 梯度）。"""
    mse = (pred - target).pow(2).flatten(1).mean(1).clamp_min(1e-8)
    pn = ((-10.0 * torch.log10(mse) - 15.0) / 20.0).clamp(0.0, 1.0)
    sh = (_lap_var(pred) / _lap_var(target).clamp_min(1e-8)).clamp(max=1.0)
    return 0.45 * pn + 0.45 * ssim(pred, target) + 0.10 * sh


IGNORE = 255  # act006 外部資料的未知／邊界像素


def dice_loss(logits, target, num_classes=5, eps=1.0):
    p = logits.softmax(1)
    valid = target != IGNORE
    if not valid.all():
        p = p * valid[:, None].to(p.dtype); target = target.masked_fill(~valid, 0)
    oh = F.one_hot(target, num_classes).permute(0, 3, 1, 2).to(p.dtype) * valid[:, None].to(p.dtype)
    inter = (p * oh).sum((0, 2, 3)); denom = p.sum((0, 2, 3)) + oh.sum((0, 2, 3))
    present = oh.sum((0, 2, 3)) > 0
    d = 1 - (2 * inter + eps) / (denom + eps)
    return d[present].mean() if present.any() else d.mean() * 0


def lovasz_grad(gt_sorted):
    p = len(gt_sorted); gts = gt_sorted.sum()
    inter = gts - gt_sorted.float().cumsum(0); union = gts + (1 - gt_sorted).float().cumsum(0)
    j = 1.0 - inter / union
    if p > 1:
        j[1:p] = j[1:p] - j[0:-1]
    return j


def lovasz_softmax(logits, target, classes='present'):
    """Lovász-softmax（Berman et al.），逐批平攤。"""
    p = logits.softmax(1)
    N, C = p.shape[:2]
    losses = []
    for n in range(N):
        pn = p[n].permute(1, 2, 0).reshape(-1, C); tn = target[n].reshape(-1)
        keep = tn != IGNORE
        pn, tn = pn[keep], tn[keep]
        for c in range(C):
            fg = (tn == c).float()
            if classes == 'present' and fg.sum() == 0:
                continue
            errors = (fg - pn[:, c]).abs()
            errors_sorted, perm = torch.sort(errors, 0, descending=True)
            losses.append(torch.dot(errors_sorted, lovasz_grad(fg[perm])))
    return torch.stack(losses).mean() if losses else logits.sum() * 0


class SegLoss(nn.Module):
    def __init__(self, class_weights=(1.0, 1.0, 1.5, 3.0, 2.0), w_dice=1.0, w_lovasz=0.0):
        super().__init__()
        self.register_buffer('cw', torch.tensor(class_weights, dtype=torch.float32))
        self.w_dice, self.w_lovasz = w_dice, w_lovasz

    def forward(self, logits, target):
        if not (target != IGNORE).any():
            return logits.sum() * 0
        loss = F.cross_entropy(logits, target, weight=self.cw.to(logits.dtype), ignore_index=IGNORE)
        if self.w_dice:
            loss = loss + self.w_dice * dice_loss(logits, target, logits.shape[1])
        if self.w_lovasz:
            loss = loss + self.w_lovasz * lovasz_softmax(logits.float(), target)
        return loss
