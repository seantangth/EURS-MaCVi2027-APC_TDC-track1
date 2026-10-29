"""本地 composite 複刻（act003）。官方 evaluator 未公開，以下口徑：
- PSNR／SSIM：skimage，data_range=1，逐張；PSNRnorm 兩種口徑都算：對平均 PSNR 正規化（以看板數據回歸支持）與逐張 clip 後平均。
- sharpness_ratio：min(1, Var(Laplacian(gray_out)) / Var(Laplacian(gray_gt)))（推定公式；榜上最大值 1.000）。
- UIQM（Panetta 2016）、UCIQE（Yang 2015）：標準公式；CCF（Wang 2018）為近似實作（colorfulness＋contrast＋fog 的加權）。
  NRquality 正規化範圍未知：預設 (lo, hi) 可調，用 t1 探針校準；在校準前只用於相對比較。
- Qseg：整個集合像素累加（官方澄清）：mIoU＝出現類別 IoU 平均、F1＝pooled P/R 的逐類 F1 平均、
  coverage＝min(1, 最差出現類別 IoU／0.02)。
composite = 0.6 Qseg + 0.3 (0.45 PSNRnorm + 0.45 SSIM + 0.10 sharp) + 0.1 NR。
"""
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as _psnr, structural_similarity as _ssim
from skimage.color import rgb2lab, rgb2hsv

NR_RANGES = {'uiqm': (0.0, 2.27), 'uciqe': (0.3, 0.7), 'ccf': (0.0, 0.23)}   # 推定；寬度取自看板回歸


def psnr(out, gt):
    mse = np.mean((out.astype(np.float64) - gt.astype(np.float64)) ** 2)
    return 99.0 if mse == 0 else float(10 * np.log10(1.0 / mse))


def ssim(out, gt):
    return float(_ssim(gt, out, channel_axis=2, data_range=1.0))


def _gray(x):
    return 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]


def _laplacian_var(g):
    lap = (-4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:])
    return float(lap.var())


def sharpness_ratio(out, gt):
    vg = _laplacian_var(_gray(gt))
    if vg <= 1e-12:
        return 1.0
    return float(min(1.0, _laplacian_var(_gray(out)) / vg))


def uiqm(img):
    """img float 0–1 RGB。c1=0.0282, c2=0.2953, c3=3.5753。"""
    x = img.astype(np.float64)
    R, G, B = x[..., 0], x[..., 1], x[..., 2]
    rg, yb = R - G, 0.5 * (R + G) - B
    def mu_alpha(v, a=0.1):
        v = np.sort(v.ravel()); n = len(v); lo, hi = int(a * n), int(n - a * n)
        return v[lo:hi].mean() if hi > lo else v.mean()
    def sig2(v, m):
        return ((v - m) ** 2).mean()
    mrg, myb = mu_alpha(rg), mu_alpha(yb)
    uicm = -0.0268 * np.sqrt(mrg ** 2 + myb ** 2) + 0.1586 * np.sqrt(sig2(rg, mrg) + sig2(yb, myb))
    def eme(ch, k=8):
        h, w = ch.shape; bh, bw = h // k, w // k
        if bh == 0 or bw == 0:
            return 0.0
        vals = []
        for i in range(k):
            for j in range(k):
                blk = ch[i * bh:(i + 1) * bh, j * bw:(j + 1) * bw]
                mx, mn = blk.max(), blk.min()
                if mn > 0 and mx > 0:
                    vals.append(np.log(mx / mn))
        return 2.0 / (k * k) * float(np.sum(vals)) if vals else 0.0
    def sobel(ch):
        gx = np.abs(np.diff(ch, axis=1, prepend=ch[:, :1])); gy = np.abs(np.diff(ch, axis=0, prepend=ch[:1, :]))
        return gx + gy
    uism = 0.299 * eme(sobel(R) * R) + 0.587 * eme(sobel(G) * G) + 0.114 * eme(sobel(B) * B)
    def logamee(ch, k=8):
        h, w = ch.shape; bh, bw = h // k, w // k
        if bh == 0 or bw == 0:
            return 0.0
        s = 0.0; cnt = 0
        for i in range(k):
            for j in range(k):
                blk = ch[i * bh:(i + 1) * bh, j * bw:(j + 1) * bw]
                mx, mn = blk.max(), blk.min()
                top, bot = mx - mn, mx + mn
                if bot > 1e-6 and top > 1e-6:
                    s += (top / bot) * np.log(top / bot); cnt += 1
        return s / max(cnt, 1)
    uiconm = logamee(_gray(x))
    return float(0.0282 * uicm + 0.2953 * uism + 3.5753 * uiconm)


def uciqe(img):
    lab = rgb2lab(np.clip(img, 0, 1))
    L, a, b = lab[..., 0] / 100.0, lab[..., 1], lab[..., 2]
    chroma = np.sqrt(a ** 2 + b ** 2) / 100.0
    sigma_c = float(chroma.std())
    con_l = float(np.percentile(L, 99) - np.percentile(L, 1))
    hsv = rgb2hsv(np.clip(img, 0, 1)); mu_s = float(hsv[..., 1].mean())
    return float(0.4680 * sigma_c + 0.2745 * con_l + 0.2576 * mu_s)


def ccf(img):
    """近似 CCF：colorfulness（Hasler-Süsstrunk，正規化）、RMS contrast、fog density proxy（暗通道均值的反向）。"""
    x = img.astype(np.float64)
    R, G, B = x[..., 0], x[..., 1], x[..., 2]
    rg, yb = R - G, 0.5 * (R + G) - B
    colorfulness = np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    contrast = _gray(x).std()
    fog = 1.0 - float(np.minimum(np.minimum(R, G), B).mean())   # 暗通道越低 → 霧越少（粗略）
    return float(0.17593 * colorfulness + 0.11718 * contrast + 0.02 * fog)


def nr_quality(img, ranges=NR_RANGES):
    vals = {'uiqm': uiqm(img), 'uciqe': uciqe(img), 'ccf': ccf(img)}
    norm = {k: float(np.clip((v - ranges[k][0]) / (ranges[k][1] - ranges[k][0]), 0, 1)) for k, v in vals.items()}
    return vals, norm, float(np.mean(list(norm.values())))


class PooledSeg:
    """整個集合層級的混淆矩陣累加（官方口徑）。"""
    def __init__(self, num_classes=5, floor=0.02):
        self.C, self.floor = num_classes, floor
        self.cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    def add(self, pred, gt):
        p, g = pred.astype(np.int64).ravel(), gt.astype(np.int64).ravel()
        ok = (g >= 0) & (g < self.C)
        self.cm += np.bincount(g[ok] * self.C + p[ok], minlength=self.C * self.C).reshape(self.C, self.C)

    def summary(self):
        cm = self.cm; tp = np.diag(cm).astype(np.float64)
        fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
        present = cm.sum(1) > 0
        iou = tp / np.maximum(tp + fp + fn, 1); f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
        miou = float(iou[present].mean()) if present.any() else 0.0
        mf1 = float(f1[present].mean()) if present.any() else 0.0
        worst = float(iou[present].min()) if present.any() else 0.0
        cov = float(min(1.0, worst / self.floor))
        return {'mIoU': miou, 'F1': mf1, 'coverage': cov, 'Qseg': mf1 * miou * cov,
                'IoU_per_class': iou.round(4).tolist(), 'present': present.tolist()}


def composite(qseg, psnr_mean, ssim_mean, sharp_mean, nr_mean, psnr_norm_per_image=None):
    pn_mean = float(np.clip((psnr_mean - 15.0) / 20.0, 0, 1))
    rest = 0.45 * pn_mean + 0.45 * ssim_mean + 0.10 * sharp_mean
    out = {'composite': 0.6 * qseg + 0.3 * rest + 0.1 * nr_mean, 'PSNRnorm_meanPSNR': pn_mean}
    if psnr_norm_per_image is not None:
        rest2 = 0.45 * psnr_norm_per_image + 0.45 * ssim_mean + 0.10 * sharp_mean
        out['composite_perimagePSNR'] = 0.6 * qseg + 0.3 * rest2 + 0.1 * nr_mean
    return out


class Evaluator:
    """累加整個驗證集：restored／GT 影像（float 0–1，HWC）與遮罩（HW int）。"""
    def __init__(self, with_nr=True, nr_every=1):
        self.seg = PooledSeg(); self.psnr = []; self.ssim = []; self.sharp = []; self.nr = []; self.with_nr = with_nr; self.nr_every = nr_every; self._i = 0

    def add(self, restored, gt_img, pred_mask, gt_mask):
        self.psnr.append(psnr(restored, gt_img)); self.ssim.append(ssim(restored, gt_img)); self.sharp.append(sharpness_ratio(restored, gt_img))
        if self.with_nr and (self._i % self.nr_every == 0):
            self.nr.append(nr_quality(restored)[2])
        self.seg.add(pred_mask, gt_mask); self._i += 1

    def summary(self):
        s = self.seg.summary()
        p = np.array(self.psnr); pn_img = float(np.clip((p - 15) / 20, 0, 1).mean()) if len(p) else 0.0
        nr = float(np.mean(self.nr)) if self.nr else 0.0
        c = composite(s['Qseg'], float(p.mean()) if len(p) else 0.0, float(np.mean(self.ssim)) if self.ssim else 0.0,
                      float(np.mean(self.sharp)) if self.sharp else 0.0, nr, pn_img)
        return {**s, 'PSNR': float(p.mean()) if len(p) else 0.0, 'SSIM': float(np.mean(self.ssim)) if self.ssim else 0.0,
                'sharpness': float(np.mean(self.sharp)) if self.sharp else 0.0, 'NR': nr, 'PSNRnorm_perimage': pn_img, **c, 'n': int(len(p))}
