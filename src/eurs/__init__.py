"""EURS（MaCVi @ WACV 2027）訓練／評估／匯出套件。

模組：
- folds.py     leave-scene-out 折切分（20 個訓練場景 → 4 折）
- data.py      cache768／cache1024 資料集、拉伸到靜態 (H,W)、增強
- model.py     物理引導雙頭網路（低解析估 T、B → 全解析逐元素反演；分割頭）
- losses.py    Charbonnier＋SSIM；CE＋Dice／Lovász
- metrics.py   本地 composite 複刻（PSNR／SSIM／sharpness／UIQM／UCIQE／CCF；pooled Qseg）
- train.py     訓練迴圈（AMP、EMA、cosine、leave-scene-out 驗證）
- export_onnx.py  dynamo 匯出 → onnxslim → 官方 FLOPs 檢查
- infer_t1.py  Track 1 推論（與 Track 2 相同的拉伸路徑）＋官方 resize 腳本
"""
CLASS_NAMES = ['background', 'coral', 'fish', 'elasmobranch', 'other']
NUM_CLASSES = 5
