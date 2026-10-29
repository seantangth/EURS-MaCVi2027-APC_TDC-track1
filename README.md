# EURS @ MaCVi WACV 2027 — team APC_TDC — Track 1 reproduction

This repository reproduces our reported Track 1 testing-phase submission **#19441** (`EURS_t1_sub046`, composite 0.691).
It follows the official template (https://github.com/CeviKle/EURS-MaCVi2027-repro-template). `METHOD.md` describes the
method, all training data, external data and pretrained models.

## Contents
| Path | What it is |
|---|---|
| `run.py` | `main(model_dir, input_dir, output_dir, device)`: full-resolution images in, `restored/<stem>.jpg` and `seg_masks/<stem>.png` out, at the scoring size |
| `src/eurs/` | model code: `restorer.py` (NAFNet with the water-type/level prior), `teacher.py` (DINOv3 + LoRA + FPN and Mask2Former teachers) |
| `weights/weights_link.yaml` | public link and SHA-256 of `eurs_t1_weights.zip` (four checkpoints and their third-party licences, 7.0 GB), hosted at https://huggingface.co/seantangth/EURS-MaCVi2027-APC_TDC-track1-weights and unpacked into `weights/` |
| `Dockerfile`, `docker-requirements.txt` | the environment of our leaderboard run: PyTorch 2.11.0 + CUDA 12.8, Python 3.12, pinned packages |
| `repro.yaml`, `METHOD.md` | hand-in metadata and method description |
| `repro_check.py`, `docker/` | the organizers' checker and base image, unchanged |

## Run
```bash
python repro_check.py . --download-weights          # downloads and unpacks the weights into weights/
docker build -t eurs-t1 .
docker run --rm --network none --gpus all --read-only --tmpfs /tmp:size=4g \
    -v "$PWD/weights":/pkg/weights:ro -v /path/to/degraded:/in:ro -v /path/to/out:/out \
    eurs-t1 python -c 'import run; run.main("/pkg/weights", "/in", "/out", "cuda")'
```
One 24 GB GPU is enough (peak about 5 GB). The four models run in bfloat16 autocast, as in our leaderboard run.
If the CUDA 12.8 image does not start with your driver, change only the `FROM` line of the `Dockerfile` to
`pytorch/pytorch:2.11.0-cuda12.6-cudnn9-runtime` (same PyTorch release).
