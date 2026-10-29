"""資料集：讀 Lambda 前處理產生的快取（cache768／cache1024），路徑格式
  <root>/cache<side>/train/degraded/<scene>_<a>_<b>.png
  <root>/cache<side>/train/gt_images/<scene>.png
  <root>/cache<side>/train/gt_masks/<scene>.png
  <root>/cache<side>/dev/degraded/<scene>_<a>_<b>.png

訓練輸入一律「拉伸」到靜態 (H, W)（與主辦方 Track 2 的 PIL bilinear 直接拉伸一致）；
增強：水平翻轉、隨機 90° 旋轉（讓直幅／橫幅都見過）、隨機裁切（長寬比抖動）、輕度 gamma 抖動（只動輸入）。
遮罩用 nearest。回傳 float32 tensor：x (3,H,W) 0–1、y (3,H,W) 0–1、m (H,W) int64。"""
import os
import random
import re

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

PAT = re.compile(r'^(.+)_(\d+)_(\d+)\.png$')
PORTRAIT_P = 0.0     # 直幅增強（10-27 審查第 3 點，未開訓）：train.py --aug-portrait 設定；以此機率改做「正立直幅裁切（寬高比 0.62–0.78）→ 拉伸到 (H,W)」，模擬 Jetson 把直幅 test 影像直接拉成橫幅輸入（t1 test 6 場景有 4 個直幅）；此模式下 90° 旋轉只取 0／90／270（不再上下顛倒）
WITH_CLEAN = False   # exp068：train.py --fd-mode clean 時設 True，三種資料集都多回傳 xc（乾淨圖）與 has_clean（DataLoader 以 fork 繼承）


def parse_name(fname):
    m = PAT.match(os.path.basename(fname))
    return m.group(1), int(m.group(2)), int(m.group(3))


def list_degraded(root, side, split, scenes=None, ab_filter=None):
    d = os.path.join(root, f'cache{side}', split, 'degraded')
    out = []
    for f in sorted(os.listdir(d)):
        if not f.endswith('.png'):
            continue
        sc, a, b = parse_name(f)
        if scenes is not None and sc not in scenes:
            continue
        if ab_filter is not None and not ab_filter(a, b):
            continue
        out.append(os.path.join(d, f))
    return out


class EURSPairs(Dataset):
    def __init__(self, root, side, scenes, hw=(768, 1024), train=True, ab_filter=None, with_gt=True, split='train', with_soft=False):
        self.root, self.side, self.hw, self.train, self.with_gt, self.split = root, side, tuple(hw), train, with_gt, split
        self.with_soft = with_soft  # 與 PseudoPairs 混用時補上 has_gt／soft 欄位，讓 default_collate 的鍵一致
        self.files = list_degraded(root, side, split, scenes, ab_filter)
        self._gt_cache = {}

    def __len__(self):
        return len(self.files)

    def _gt(self, scene):
        if scene not in self._gt_cache:
            base = os.path.join(self.root, f'cache{self.side}', 'train')
            img = Image.open(os.path.join(base, 'gt_images', scene + '.png')).convert('RGB')
            msk = Image.open(os.path.join(base, 'gt_masks', scene + '.png'))
            self._gt_cache[scene] = (img, msk)
        return self._gt_cache[scene]

    def __getitem__(self, i):
        path = self.files[i]
        scene, a, b = parse_name(path)
        x = Image.open(path).convert('RGB')
        H, W = self.hw
        if self.with_gt:
            y, m = self._gt(scene)
            if y.size != x.size:
                y = y.resize(x.size, Image.BILINEAR)
            if m.size != x.size:
                m = m.resize(x.size, Image.NEAREST)
        else:
            y = m = None
        if self.train:
            # 隨機裁切（面積 60–100%，長寬比 3/4–4/3）→ 之後拉伸到 (H,W) 製造長寬比抖動
            portrait = PORTRAIT_P > 0 and random.random() < PORTRAIT_P
            if portrait:  # 直幅增強（10-27 審查第 3 點，未開訓）：正立直幅裁切、不旋轉
                box = portrait_box(*x.size)
                x = x.crop(box)
                if y is not None:
                    y, m = y.crop(box), m.crop(box)
            elif random.random() < 0.7:
                w0, h0 = x.size
                area = random.uniform(0.6, 1.0) * w0 * h0
                ar = random.uniform(0.75, 1.333)
                cw, ch = int(min(w0, (area * ar) ** 0.5)), int(min(h0, (area / ar) ** 0.5))
                left, top = random.randint(0, w0 - cw), random.randint(0, h0 - ch)
                box = (left, top, left + cw, top + ch)
                x = x.crop(box)
                if y is not None:
                    y, m = y.crop(box), m.crop(box)
            k = 0 if portrait else rot_k()  # 90° 旋轉：直幅／橫幅互換
            if k:
                x = x.rotate(90 * k, expand=True)
                if y is not None:
                    y, m = y.rotate(90 * k, expand=True), m.rotate(90 * k, expand=True)
            if random.random() < 0.5:
                x = x.transpose(Image.FLIP_LEFT_RIGHT)
                if y is not None:
                    y, m = y.transpose(Image.FLIP_LEFT_RIGHT), m.transpose(Image.FLIP_LEFT_RIGHT)
        # 拉伸到靜態 (H,W)：與 Track 2 評分伺服器一致（PIL bilinear），遮罩 nearest
        x = x.resize((W, H), Image.BILINEAR)
        xt = torch.from_numpy(np.asarray(x, dtype=np.float32) / 255.0).permute(2, 0, 1).contiguous()
        if self.train and random.random() < 0.3:  # 輕度 gamma 抖動（只動輸入）
            g = random.uniform(0.85, 1.18)
            xt = xt.clamp(0, 1) ** g
        item = {'x': xt, 'name': os.path.splitext(os.path.basename(path))[0], 'scene': scene, 'a': a, 'b': b}
        if y is not None:
            y = y.resize((W, H), Image.BILINEAR)
            m = m.resize((W, H), Image.NEAREST)
            item['y'] = torch.from_numpy(np.asarray(y, dtype=np.float32) / 255.0).permute(2, 0, 1).contiguous()
            item['m'] = torch.from_numpy(np.asarray(m, dtype=np.int64).copy()).clamp_(0, 4)
        if self.with_soft:
            item['has_gt'] = 1
            item['has_soft'] = 0
            item['soft'] = torch.zeros(5, H // 4, W // 4)
        if WITH_CLEAN:  # exp068 b 臂：特徵蒸餾的 teacher 吃乾淨圖（官方 train 的 GT 影像）
            item['xc'] = item['y'] if 'y' in item else xt
            item['has_clean'] = int('y' in item)
        return item


class PseudoPairs(Dataset):
    """無 GT 的退化影像（預設 dev）＋ teacher 偽標籤：硬遮罩 PNG（任意尺寸，nearest 對齊）＋ 1/4 解析 fp16 softmax .npy
    （對應「原方向拉伸到 (H,W)」的網格）。增強只做 90° 旋轉與水平翻轉（不裁切），同步套在軟標籤上以保持對齊。
    回傳 has_gt=0、y=x（占位，訓練時不算復原損失）、m=硬偽遮罩、soft=(5,H/4,W/4) 機率。"""

    def __init__(self, root, side, pseudo_dir, hw=(768, 1024), train=True, split='dev', scenes=None, masks_sub='dev_masks', logits_sub='dev_logits'):
        self.hw, self.train = tuple(hw), train
        self.files = list_degraded(root, side, split, scenes)
        self.masks, self.logits = os.path.join(pseudo_dir, masks_sub), os.path.join(pseudo_dir, logits_sub)
        missing = [f for f in self.files if not os.path.exists(os.path.join(self.masks, os.path.splitext(os.path.basename(f))[0] + '.png'))]
        if missing:
            raise FileNotFoundError(f'{len(missing)} 張影像缺偽遮罩，例如 {os.path.basename(missing[0])}')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        import torch.nn.functional as F
        path = self.files[i]
        scene, a, b = parse_name(path)
        stem = os.path.splitext(os.path.basename(path))[0]
        x = Image.open(path).convert('RGB')
        m = Image.open(os.path.join(self.masks, stem + '.png'))
        if m.mode != 'L':
            m = m.convert('L')
        if m.size != x.size:
            m = m.resize(x.size, Image.NEAREST)
        soft = torch.from_numpy(np.load(os.path.join(self.logits, stem + '.npy')).astype(np.float32))  # (5,h,w)
        H, W = self.hw
        if self.train:
            portrait = PORTRAIT_P > 0 and random.random() < PORTRAIT_P
            if portrait:  # 直幅增強（10-27 審查第 3 點，未開訓）：正立直幅裁切（軟標籤網格同比例裁切）
                w0, h0 = x.size; box = portrait_box(w0, h0)
                x, m = x.crop(box), m.crop(box)
                hs, ws = soft.shape[1:]
                t0, b0 = int(round(box[1] / h0 * hs)), int(round(box[3] / h0 * hs)); l0, r0 = int(round(box[0] / w0 * ws)), int(round(box[2] / w0 * ws))
                soft = soft[:, t0:max(b0, t0 + 1), l0:max(r0, l0 + 1)]
            k = 0 if portrait else rot_k()
            if k:
                x, m = x.rotate(90 * k, expand=True), m.rotate(90 * k, expand=True)
                soft = torch.rot90(soft, k, dims=(1, 2))  # PIL rotate 與 torch.rot90 皆為逆時針
            if random.random() < 0.5:
                x, m = x.transpose(Image.FLIP_LEFT_RIGHT), m.transpose(Image.FLIP_LEFT_RIGHT)
                soft = soft.flip(2)
        x = x.resize((W, H), Image.BILINEAR)
        m = m.resize((W, H), Image.NEAREST)
        soft = F.interpolate(soft[None], size=(H // 4, W // 4), mode='bilinear', align_corners=False)[0]
        soft = soft / soft.sum(0, keepdim=True).clamp_min(1e-6)
        xt = torch.from_numpy(np.asarray(x, dtype=np.float32) / 255.0).permute(2, 0, 1).contiguous()
        if self.train and random.random() < 0.3:
            xt = xt.clamp(0, 1) ** random.uniform(0.85, 1.18)
        item = {'x': xt, 'y': xt.clone(), 'm': torch.from_numpy(np.asarray(m, dtype=np.int64).copy()).clamp_(0, 4),
                'soft': soft, 'has_gt': 0, 'has_soft': 1, 'name': stem, 'scene': scene, 'a': a, 'b': b}
        if WITH_CLEAN:  # dev 影像沒有乾淨版本：不進 b 臂的特徵蒸餾損失
            item['xc'] = xt; item['has_clean'] = 0
        return item


def portrait_box(w0, h0):
    """直幅增強（10-27 審查第 3 點，未開訓）：正立直幅裁切框（寬高比 0.62–0.78，高度取原圖 75–100%），回傳 (left, top, right, bottom)。"""
    r = random.uniform(0.62, 0.78); ch = h0 * random.uniform(0.75, 1.0); cw = ch * r
    if cw > w0:
        cw = w0; ch = min(h0, cw / r)
    cw, ch = max(1, int(cw)), max(1, int(ch))
    left, top = random.randint(0, w0 - cw), random.randint(0, h0 - ch)
    return left, top, left + cw, top + ch


def rot_k():
    """90° 旋轉次數：預設 0–3 均勻；直幅增強模式（PORTRAIT_P>0）只取 0／1／3（0 佔一半，不上下顛倒）。"""
    return random.choice((0, 0, 1, 3)) if PORTRAIT_P > 0 else random.randint(0, 3)


def load_ab_table(path=None):
    """exp008 的逐 (a,b) 全域仿射擬合：退化 ≈ alpha·GT＋beta（逐通道），alpha_blockstd＝區塊間 alpha 的標準差。
    排除 frac_dark > 0.5 的近全黑格子（外部影像加上它們只剩雜訊）。"""
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'degradation_ab.csv')
    import csv
    rows = []
    for r in csv.DictReader(open(path)):
        if float(r['frac_dark'] or 0) > 0.5:
            continue
        f = lambda k: float(r[k]) if r[k] not in ('', None) else 0.0
        rows.append((int(r['a']), int(r['b']), np.array([f('alpha_r'), f('alpha_g'), f('alpha_b')], np.float32),
                     np.array([f('beta_r'), f('beta_g'), f('beta_b')], np.float32),
                     np.array([f('alpha_blockstd_r'), f('alpha_blockstd_g'), f('alpha_blockstd_b')], np.float32)))
    return rows


def redegrade(x, row, rng):
    """x：(3,H,W) 0–1 tensor。依 (a,b) 的仿射係數再退化，alpha 加 4×3 網格的平滑空間擾動（幅度＝區塊標準差）模擬逐像素距離差。"""
    import torch.nn.functional as F
    _, _, alpha, beta, bstd = row
    H, W = x.shape[-2:]
    grid = torch.from_numpy((alpha[:, None, None] + bstd[:, None, None] * rng.standard_normal((3, 3, 4))).astype(np.float32)).clamp_min(0)
    field = F.interpolate(grid[None], size=(H, W), mode='bicubic', align_corners=True)[0].clamp_min(0)
    return (field * x + torch.from_numpy(beta)[:, None, None]).clamp(0, 1)


class ExtSeg(Dataset):
    """act006 外部分割資料（ext_prepare.py 產出）：images/*.jpg ＋ masks/*.png（0–4，255＝ignore）。
    增強同 EURSPairs（裁切、90° 旋轉、翻轉）後，以 exp008 的 (a,b) 仿射隨機再退化（p_clean 機率保留原圖，外部影像本身已是水下）。
    預設只供分割：has_gt=0（不算復原損失）、has_soft=0（無 teacher 軟標籤）。
    rest_amax>0（exp037 b 臂候選）：再退化水型 a ≤ rest_amax 的樣本以再退化前的影像當復原 GT（has_gt=1）；
    exp008 顯示 a≤3 的退化近乎全域仿射，redegrade() 才像真的，a≥4 的空間變化背散射它模擬不了。"""

    def __init__(self, ext_root, hw=(768, 1024), train=True, with_soft=False, datasets=None, p_clean=0.2, rare_min_px=200, rest_amax=0, pseudo_dir='', drop_cls=()):
        import csv
        self.root, self.hw, self.train, self.with_soft, self.p_clean, self.rest_amax = ext_root, tuple(hw), train, with_soft, p_clean, rest_amax
        self.drop_cls = tuple(int(c) for c in drop_cls)   # exp086：外部遮罩中這些類別改 ignore（例：0 1＝丟掉外部 bg／coral 監督，只留 fish／elasmo／other 實例）
        rows = list(csv.DictReader(open(os.path.join(ext_root, 'ext_index.csv'))))
        if datasets:
            rows = [r for r in rows if r['ds'] in datasets]
        self.rows = rows
        self.rare = np.array([int(r['elasmo']) >= rare_min_px or int(r['other']) >= rare_min_px for r in rows])
        self.ab = load_ab_table()
        # exp053：teacher 偽標籤（exp051 的 ext_masks/<ds>/<stem>.png 原尺寸硬遮罩＋ext_logits/<ds>/<stem>.npy 1/4 解析 softmax，對應未裁切、拉伸到 (H,W) 的網格）。
        # 有偽標籤的樣本：m＝teacher 硬遮罩，但外部遮罩標為稀有類（elasmo／other）之處以外部遮罩為準（稀有類的外部實例遮罩比 teacher 可靠，exp026）；soft 給 KL 蒸餾（has_soft=1）。
        self.pseudo_dir = pseudo_dir
        self.has_pseudo = np.zeros(len(rows), dtype=bool)
        if pseudo_dir:
            for i, r in enumerate(rows):
                stem = os.path.splitext(os.path.basename(r['path_img']))[0]
                self.has_pseudo[i] = os.path.exists(os.path.join(pseudo_dir, 'ext_logits', r['ds'], stem + '.npy')) and os.path.exists(os.path.join(pseudo_dir, 'ext_masks', r['ds'], stem + '.png'))
            print(f'ExtSeg pseudo: {int(self.has_pseudo.sum())}/{len(rows)} rows have teacher soft labels from {pseudo_dir}', flush=True)

    def __len__(self):
        return len(self.rows)

    def sampler_weights(self, rare_boost=4.0):
        return torch.tensor(np.where(self.rare, rare_boost, 1.0), dtype=torch.double)

    def __getitem__(self, i):
        import torch.nn.functional as F
        r = self.rows[i]
        stem = os.path.splitext(os.path.basename(r['path_img']))[0]
        x = Image.open(os.path.join(self.root, r['path_img'])).convert('RGB')
        m = Image.open(os.path.join(self.root, r['path_mask']))
        if m.size != x.size:
            m = m.resize(x.size, Image.NEAREST)
        soft = None
        if self.pseudo_dir and self.has_pseudo[i]:  # exp053：teacher 硬遮罩（稀有類以外部遮罩覆蓋）＋1/4 解析 softmax
            mt = Image.open(os.path.join(self.pseudo_dir, 'ext_masks', r['ds'], stem + '.png'))
            if mt.mode != 'L':
                mt = mt.convert('L')
            if mt.size != x.size:
                mt = mt.resize(x.size, Image.NEAREST)
            me, mtn = np.asarray(m), np.asarray(mt).copy()
            rare = (me == 3) | (me == 4)
            mtn[rare] = me[rare]
            m = Image.fromarray(mtn.astype(np.uint8), mode='L')
            soft = torch.from_numpy(np.load(os.path.join(self.pseudo_dir, 'ext_logits', r['ds'], stem + '.npy')).astype(np.float32))  # (5,h,w)，未裁切拉伸網格
        if self.drop_cls:
            mn = np.asarray(m).copy(); mn[np.isin(mn, self.drop_cls)] = 255
            m = Image.fromarray(mn.astype(np.uint8), mode='L')
        if self.train:
            portrait = PORTRAIT_P > 0 and random.random() < PORTRAIT_P
            if portrait or random.random() < 0.7:
                w0, h0 = x.size
                if portrait:  # 直幅增強（10-27 審查第 3 點，未開訓）：正立直幅裁切
                    left, top, rr, bb = portrait_box(w0, h0); cw, ch = rr - left, bb - top
                else:
                    area = random.uniform(0.6, 1.0) * w0 * h0; ar = random.uniform(0.75, 1.333)
                    cw, ch = int(min(w0, (area * ar) ** 0.5)), int(min(h0, (area / ar) ** 0.5))
                    left, top = random.randint(0, w0 - cw), random.randint(0, h0 - ch)
                x, m = x.crop((left, top, left + cw, top + ch)), m.crop((left, top, left + cw, top + ch))
                if soft is not None:  # 同一個裁切框換算成軟標籤網格的比例座標（拉伸是逐軸線性縮放，比例不變）
                    hs, ws = soft.shape[1:]
                    t0, b0 = int(round(top / h0 * hs)), int(round((top + ch) / h0 * hs)); l0, r0 = int(round(left / w0 * ws)), int(round((left + cw) / w0 * ws))
                    soft = soft[:, t0:max(b0, t0 + 1), l0:max(r0, l0 + 1)]
            k = 0 if portrait else rot_k()
            if k:
                x, m = x.rotate(90 * k, expand=True), m.rotate(90 * k, expand=True)
                if soft is not None:
                    soft = torch.rot90(soft, k, dims=(1, 2))  # PIL rotate 與 torch.rot90 皆為逆時針
            if random.random() < 0.5:
                x, m = x.transpose(Image.FLIP_LEFT_RIGHT), m.transpose(Image.FLIP_LEFT_RIGHT)
                if soft is not None:
                    soft = soft.flip(2)
        H, W = self.hw
        x = x.resize((W, H), Image.BILINEAR); m = m.resize((W, H), Image.NEAREST)
        xt = torch.from_numpy(np.asarray(x, dtype=np.float32) / 255.0).permute(2, 0, 1).contiguous()
        x_clean = xt
        a = b = 0
        if self.train and random.random() >= self.p_clean:
            row = self.ab[random.randrange(len(self.ab))]; a, b = row[0], row[1]
            xt = redegrade(xt, row, np.random.default_rng(random.getrandbits(32)))
        has_gt = int(self.rest_amax > 0 and 1 <= a <= self.rest_amax)
        item = {'x': xt, 'name': stem, 'scene': 'ext_' + r['ds'], 'a': a, 'b': b,
                'y': x_clean if has_gt else xt.clone(), 'm': torch.from_numpy(np.asarray(m, dtype=np.int64).copy())}
        if self.with_soft:
            if soft is not None:
                soft = F.interpolate(soft[None], size=(H // 4, W // 4), mode='bilinear', align_corners=False)[0]
                soft = soft / soft.sum(0, keepdim=True).clamp_min(1e-6)
                item.update({'has_gt': has_gt, 'has_soft': 1, 'soft': soft.contiguous()})
            else:
                item.update({'has_gt': has_gt, 'has_soft': 0, 'soft': torch.zeros(5, H // 4, W // 4)})
        if WITH_CLEAN:  # exp068 b 臂：再退化前的外部原圖
            item['xc'] = x_clean; item['has_clean'] = 1
        return item


def mix_weights(w_base, w_ext, share):
    """把外部資料的抽樣權重縮放成總權重的 share 比例；回傳 (合併權重, 每 epoch 樣本數倍率 1/(1−share))。"""
    w_ext = w_ext / w_ext.sum() * w_base.sum() * share / (1 - share)
    return torch.cat([w_base, w_ext]), 1.0 / (1 - share)


def scene_balanced_sampler_weights(files, rare_boost=None):
    """讓每個場景被抽到的機率一致（每場景 200 張本來就等量，但加入外部／再退化資料後會失衡），
    並對稀有類別場景加權。rare_boost: {scene: 倍率}。"""
    scenes = [parse_name(f)[0] for f in files]
    counts = {}
    for s in scenes:
        counts[s] = counts.get(s, 0) + 1
    w = []
    for s in scenes:
        wt = 1.0 / counts[s]
        if rare_boost and s in rare_boost:
            wt *= rare_boost[s]
        w.append(wt)
    return torch.tensor(w, dtype=torch.double)
