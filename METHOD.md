# Method description

Team: APC_TDC   Track(s): 1 (this repository; Track 2 has its own repository)   Reported testing-phase submission: #19441 (`EURS_t1_sub046`)

## Approach (one page at most)
Two independent models produce the two outputs. Every output is computed from its own input image only.

- **Input.** The full-resolution image is scaled to the scoring size (longest side 1024 px, `round()`, PIL bilinear, downscale only), exactly as in `resize_for_submission.py`, and converted to RGB. All models work on this image, and all outputs have this size.
- **Restoration: NAFNet conditioned on the water type and level (29.2 M parameters).**
  - Architecture: a U-shaped NAFNet (width 32, 2/2/4/8 encoder blocks, 12 middle blocks, 2/2/2/2 decoder blocks).
  - Metadata: the Jerlov water type `a` (1–10) and the level `b` (5–100) are read from the end of the file name (`stem.rsplit("_", 2)`).
  - Physical prior: for every `(a, b)` a 3×4 linear inverse colour transform `M_ab` was fitted by ridge least squares on the 20 training scenes. The table is stored in the checkpoint.
  - The network sees four inputs: the image, a per-image standardised copy of it, `J0 = M_ab·[x; 1]`, and two constant planes `a/10` and `b/100`. It predicts a residual on top of `J0`.
  - Output: clamped to [0, 1], rounded to 8 bit, saved as JPEG (quality 98, 4:4:4). We uploaded JPEG to stay under the 1 GB upload limit.
- **Segmentation: per-pixel majority vote of three different teachers.**
  1. Frozen DINOv3 ViT-L/16 with rank-8 LoRA on q and v in every block. Features from blocks 6/12/18/24 feed a multi-scale pyramid and an FPN decoder. Trained and run at 1024×1360. 314.6 M parameters, 11.5 M of them trained.
  2. The same architecture at 768×1024. It is trained with *partial* labels: unlabelled pixels of the external instance datasets are ignored instead of pseudo-filled.
  3. Mask2Former with a Swin-L backbone (215 M parameters), initialised from ADE20K semantic segmentation, with a new 5-class head. It is trained with the native mask-classification loss and run at 768×1024. Per-pixel class scores follow the standard Mask2Former semantic inference.

  Each teacher works the same way:
  - stretch the image to the teacher's input size;
  - run the model on the image and on its horizontal flip, and average the logits;
  - softmax, then argmax;
  - resize the label map back to the scoring size with NEAREST.

  The final label is the majority of the three labels. When all three differ, teacher 1's label is used.
- **What is new.**
  - Restoration is conditioned on the file-name metadata, starting from a fitted per-degradation colour prior.
  - The segmentation ensemble is heterogeneous: two DINOv3 variants and one Mask2Former. A vote of six mostly-DINOv3 members did not improve on the single best teacher; the three-architecture vote did.

## Training data
- **EURS training split.** All 20 training scenes with all 200 degradations each (GT images and GT masks), plus the `(a, b)` values in the file names.
  - The recipes were developed with one fold of 5 held-out training scenes.
  - The final models were retrained on all 20 scenes. The last epoch (teachers) and the EMA weights after 20,000 iterations (restoration) are used. Nothing was selected on testing-phase images.
- **EURS development split.** Not used by any Track 1 model.
- **Testing-phase images.** Never used for training, fine-tuning, pseudo-labelling, model selection or adaptation.
  - The only use: one image per test scene (`_1_5`) was hashed with a 256-bit dHash. This checked that no external training image was a near-duplicate of a test scene.
  - Result: the nearest distance was 50/256 (threshold 24), so nothing was removed.
- **Restoration.** EURS training scenes only; no external data, no pretrained weights.
- **External data (segmentation teachers only).** Masks were mapped to the 5 EURS classes; classes whose EURS label was unclear were set to *ignore*.
  - For teachers 1 and 3, unlabelled pixels of external images were filled with confident predictions (≥ 0.7) of an earlier teacher. That earlier teacher (ConvNeXt-L, DINOv3-pretrained) was trained on 15 EURS training scenes only, with no external data. Teacher 2 leaves those pixels as *ignore*.
  - During training, external images were re-degraded with a per-`(a, b)` colour model (20 % kept clean).

  | Dataset | Source | Licence | Images used |
  |---|---|---|---|
  | USIS16K | official Google Drive release | per the authors' release | 16,050 |
  | UIIS10K | https://huggingface.co/datasets/LiamLian0727/UIIS10K | Apache-2.0 | 9,890 |
  | USIS10K | https://huggingface.co/datasets/LiamLian0727/USIS10K | Apache-2.0 | 2,962 |
  | Coralscapes | https://huggingface.co/datasets/EPFL-ECEO/coralscapes | Apache-2.0 (HF tag) | 2,074 |
  | SUIM | https://huggingface.co/datasets/SatwikKambham/suim | MIT (HF tag) | 957 |
- **Pretrained models.**
  - DINOv3 ViT-L/16 LVD-1689M (timm `vit_large_patch16_dinov3.lvd1689m`, DINOv3 License): backbone of teachers 1 and 2. Its weights are inside our two teacher checkpoints.
  - Mask2Former Swin-L ADE20K semantic (`facebook/mask2former-swin-large-ade-semantic`, licence "other" on Hugging Face; Mask2Former code MIT): initialisation of teacher 3.
  - ConvNeXt-L DINOv3 LVD-1689M (timm `convnext_large.dinov3_lvd1689m`, DINOv3 License): backbone of the earlier teacher that filled unlabelled pixels of the external images (data preparation only; it is not part of the submission).
- **Global choices made from testing-phase leaderboard scores.** Each was submitted on its own first. All are fixed settings, identical for every image:
  - which mask models enter the vote: 3 of the 6 we had;
  - the tie rule (teacher 1);
  - teacher 1's input size of 1024×1360.

## Anything non-standard at inference
- Resizing to the scoring size as above. The masks come back to that size with NEAREST.
- Horizontal-flip test-time augmentation for each teacher.
- Ensembling: majority vote of three teachers.
- Restoration uses the water type and level from the end of the file name (the leaderboard "Meta" box was ticked).
- bfloat16 autocast on CUDA for all models, as in our leaderboard run.

## Reproduction notes
- **Environment.** The root `Dockerfile` uses `pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime`, the same PyTorch build as our leaderboard run, with the packages pinned in `docker-requirements.txt`.
  - The base image's PyTorch 2.4.0 cannot run teacher 3: `transformers` 5.20.2, which trained and ran it, needs torch ≥ 2.6.
  - If the CUDA 12.8 image does not start with your driver, change only the `FROM` line to `pytorch/pytorch:2.11.0-cuda12.6-cudnn9-runtime` (same PyTorch release).
- **Weights.** One zip (6,992,577,519 bytes) behind the link in `weights/weights_link.yaml`, unpacked into `weights/`. It holds the four checkpoints below and a `LICENSES/` folder with the licence texts of the third-party weights (DINOv3 License for the DINOv3 backbone inside the two DINOv3 teachers; MIT for Mask2Former and Swin Transformer).

  | File | SHA-256 |
  |---|---|
  | `teacher_exp066b.pt` | `d34d8617f12c24301b20de1c8dfa07329532d6283f4aaa8a7786d36d387cea6e` |
  | `teacher_exp083.pt` | `741a7d40b7e8613a8a02b94769519214998dda14bf91eaf407bf5fee24f9ab9c` |
  | `teacher_exp078_m2f.pt` | `afcb9cf08562d283eb875f03058f9bdeccc01f5b859c97c628c279316cddc5bb` |
  | `restorer_exp034.pt` | `e1dcc0cc3d34a59713adc8a9e473bca042ca610dd21960616994566db4aa6c76` |

  These are the exact checkpoints of the leaderboard run (EMA weights are used).
- **No network.**
  - `run.py` sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.
  - Every model is built with `pretrained=False` from the arguments stored in its checkpoint. Mask2Former is built from the Hugging Face config stored in the checkpoint.
  - `src/eurs/teacher.py` and `src/eurs/restorer.py` also contain our training code. Their download helpers are only used in training and are never called by `run.py`.
- **Hardware of the leaderboard run.**
  - Masks were produced on NVIDIA A10 GPUs, in batches of 4.
  - Restoration ran on an A100.
  - `run.py` processes one image at a time; the organizers' `_seed()` also sets `cudnn.deterministic`. A few argmax labels can therefore flip at class boundaries.
  - Our own reproduction measurement is in the factsheet.
- **Run time and memory.** One 24 GB GPU is enough (all four models stay loaded). See the factsheet for the measured time on an A10 (same GA102 chip as the RTX 3090).
