# EURS @ MaCVi WACV 2027: reproduction repository template

Start your final hand-in from this repository: click **Use this template** on GitHub (or download it as a ZIP). Do not copy its `.git` folder. The organizers **re-run your inference** on the testing-phase images
with the weights of the submission you report, score it with the leaderboard evaluator and compare. Only teams whose result
reproduces are ranked, in **both** tracks (F = 0.70 x Track 2 + 0.30 x Track 1), so hand in **one repository per
track you entered**: a team that entered both tracks creates two repositories from this template, one for each track (each with its own `repro.yaml`, weights and commit).

**Deadline: October 28, 2026, 23:59 AoE**, together with your factsheet (see the factsheet template).

## What to put in the repository

```
METHOD.md            one page: approach, training data, ALL external data / pretrained models, anything non-standard at inference
repro.yaml           team, track, submission_id (your best-scoring TESTING-PHASE submission in this track), contact
requirements.txt     ONLY extra pip packages the base image does not have, pinned (name==x.y.z); empty if you need none.
                     Never paste your whole `pip freeze`: it would replace the base image's PyTorch 2.4.0 and numpy
src/                 your model and utility code, importable, no absolute paths
run.py               Track 1:  main(model_dir, input_dir, output_dir, device)
export_onnx.py       Track 2:  main(model_dir, out_onnx)
onnx/                Track 2: onnx/submitted_model.onnx = a copy of the exact ONNX file you uploaded to the leaderboard (see below)
weights/             your checkpoint (committed), OR weights_link.yaml with a public download link, not both (see below)
repro_check.py       this template's checker: run it before you hand in
docker/Dockerfile    the organizers' base image (build it to test your code the way we run it)
Dockerfile           OPTIONAL, at the repository root: only if you need another PyTorch/CUDA/Python version (see "Need a different PyTorch version?")
```

**Which files for which track.** *Track 1:* `run.py` is required; delete `export_onnx.py` and the `onnx/` folder. *Track 2:* `export_onnx.py` is required and `onnx/submitted_model.onnx` is strongly recommended; `run.py` is
not needed. Delete `src/joint_model.py` (the baseline's network) and the files that do not apply, and replace the baseline code in `run.py` /
`export_onnx.py` with your own. If you committed your weights, delete `weights/weights_link.yaml`.

**Everything sits in the root of the repository** (not inside a sub-folder), at exactly these paths. A finished **Track 2** repository looks like this:

```
my-eurs-track2/
  repro.yaml                      team, track: eurs-t2, submission_id, contact
  METHOD.md                       approach, training data, external data
  requirements.txt                extra pinned packages (may be empty)
  export_onnx.py                  main(model_dir, out_onnx): regenerates the ONNX from the weights
  src/                            the model code export_onnx.py imports
  weights/best.pt                 your checkpoint, committed  (OR weights/weights_link.yaml, not both)
  onnx/submitted_model.onnx       a copy of the ONNX you uploaded to the leaderboard
  repro_check.py                  from this template, leave it as is
```

A finished **Track 1** repository has the same files, with `run.py` instead of `export_onnx.py` and no `onnx/` folder. A team that entered both
tracks makes two such repositories.

* `run.py` reads the degraded images in `input_dir` and writes `output_dir/restored/<stem>.png` (RGB) and
  `output_dir/seg_masks/<stem>.png` (single channel, labels 0 to 4), resized to **exactly** the scoring size: the input scaled down so that its longest side is 1024 px, aspect ratio kept (an input
  whose longest side is already 1024 px or less keeps its size). The evaluator does not resize for you and **rejects any other size**: use the same
  `resize_for_submission.py` step as for your leaderboard submissions. **File formats:** `restored/` files must really be PNG or JPEG, `seg_masks/` files must really be PNG (the evaluator identifies
  a file by its content, not by its name or extension; any other format is rejected). The input images in `input_dir` are the **full-resolution** files
  we released (some up to tens of megapixels), so decode them one at a time. **Every output must come from its own input image only.** `model_dir` is your `weights/` folder.
  `run.py` here is the organizer baseline: replace the model, keep the signature and the output format.
* **Reading the file name.** `input_dir` keeps the released names `<scene>_<type>_<level>.png`. Your code may read the water **type and level** from the trailing
  `_<type>_<level>` (your leaderboard pipeline could). In our check that each output depends only on its own input, we run Track 1 a second time on a copy
  in which the **scene** part of every name is replaced by a random string and the trailing `_<type>_<level>` is kept. So read type and level from the **end**
  of the name (for example `stem.rsplit("_", 2)`), and do not use the scene part or any other image.
* `export_onnx.py` (Track 2) regenerates the ONNX file you submitted from your weights: static input (1, 3, H, W) in [0, 1],
  outputs `restored_image` and `seg_logits`, at most 12 GFLOPs and 1.0 M parameters. We compare it with your leaderboard ONNX.
  **Put every step between your checkpoint and the final file into this one script** (export, graph simplification, any quantization or
  fusing), so that `main(model_dir, out_onnx)` alone regenerates the file you uploaded. Packages that only the export needs (for example
  `onnxsim`) go into `requirements.txt`, pinned; the export runs without network.
* **`onnx/submitted_model.onnx` (Track 2, strongly recommended):** a copy of the exact ONNX file you uploaded to the leaderboard for the submission you
  report. Give its SHA-256 in the factsheet. `python repro_check.py . --export-onnx` regenerates the file and tells you whether it equals your copy
  (byte-identical, or the same graph and weights with only the file header, such as the torch version, different: that is fine). We check your export against the
  leaderboard archive in any case; the copy lets you see a problem before you hand in.
* **What we do with a Track 2 hand-in, in two steps.** (1) We regenerate the ONNX in a sandbox and compare it with your leaderboard file: it
  must give the same outputs. (2) We then evaluate the regenerated file on the Jetson AGX Orin with the same pipeline as the leaderboard and compare
  throughput, energy and accuracy with your leaderboard result. This applies to every team; differences are reviewed by a person.
* **Training code is not needed.** A link to it in `METHOD.md` is welcome.
* **No network at run time**, so weights cannot be downloaded by your code. Make inference deterministic (fixed seeds,
  `model.eval()`, no test-time randomness). We run on one RTX 3090-class GPU.

## Weights: commit them, or give a public link (one way, not both)

**Committed:** small models go in `weights/`. Check that git really tracks them (`git ls-files weights`); delete `weights/weights_link.yaml`.

**Large weights:** up to **20 GB**, as one file or one archive, behind a public link. Hugging Face is the most reliable host for big files (Google Drive may refuse very large
downloads). Give the SHA-256, put the licence texts of third-party weights (for example the SAM licence) in the archive and list them in `METHOD.md`. At run time the
weights are mounted **read-only** at `weights/` (`model_dir`); nothing is copied into the image. The whole model must fit on one 24 GB GPU and in 32 GB of RAM.

**Link:** larger models go behind a public link described in `weights/weights_link.yaml` (`url:`, `sha256:` of the file, and optionally
`filename:`); give the same link and hash in your factsheet. We download it **once, before the run**, verify the hash and save it in `weights/`
under the name the link's server gives it, or under `filename:` if you set it (recommended, so your `run.py` knows the exact name). A `.zip` or
`.tar.gz` is unpacked into `weights/` keeping its inner folders. Accepted: direct links, GitHub releases, Hugging Face, Dropbox, Google Drive
**file** links shared with "anyone with the link"; one file or one archive. Not accepted: Drive folders, OneDrive, links that need a login, Git LFS.
Keep the link working at least until the final results are announced (October 30, 2026). Do not commit the downloaded file to a link-based repository.
To test a link-based repository yourself, fetch the file into `weights/` first with `python repro_check.py . --download-weights`, then run the checks.

The repository must be **public** (we clone it over https without credentials, so access-restricted repositories cannot be fetched) and pinned: give the **commit hash** in the factsheet and do not
change that commit afterwards.

## Reproduce your own pipeline exactly

We compare our re-run with the submission you uploaded to the leaderboard, so `run.py` must do **exactly what your own inference script did** for that
submission: the same resizing and interpolation, the same normalisation, the same conversion to 8 bit (rounding or truncation: copy your own, the
template's `run.py` uses the organizer baseline's), the same output sizes, the same device type if you can (the reference is whatever you used for the leaderboard;
say in `METHOD.md` if it was a CPU or an unusual GPU). Run your own script and `run.py` on the same images and check the outputs are identical.

## Need a different PyTorch version? Use your own Dockerfile

The base image has PyTorch 2.4.0, CUDA 12.1 and Python 3.11. If your code needs something else (for example a newer `transformers` that needs PyTorch 2.5), put a
`Dockerfile` at the **root** of your repository: we then build it **instead of** the base image plus `requirements.txt`. Do not pin `torch` in `requirements.txt`: it is a
download of about 3 GB that is not reliable on our build machine.

```dockerfile
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime      # already on our machine; the 2.4.0 image works too
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt  # pinned, WITHOUT torch
COPY . /pkg
WORKDIR /pkg
```
Rules: it is built from the repository root **with network access** (limit 30 minutes; another base image has to be downloaded within that time); it must copy the
repository to `/pkg` (`COPY . /pkg`); `weights/` is left out of the build context and mounted read-only at `/pkg/weights` when your code runs; the container runs
with **no network**, as an unprivileged user, with a read-only root and a 4 GB `/tmp` (set offline modes such as `HF_HUB_OFFLINE=1` and load models from `weights/`).
The machine has an RTX 3090 and a driver that runs CUDA 12.x images (we tested PyTorch 2.5.1 with CUDA 12.4). `python repro_check.py .` checks the Dockerfile; to test
the way we run it, build it yourself and run the checker inside with `--network none`.

## Data use

The released testing-phase images are for producing your submission only. Do not train, fine-tune, pseudo-label or otherwise adapt a model on them, and do not
use one test image to change the output for another. **Allowed:** choosing between your own submissions, or setting a few global values (a random seed,
which models go into an ensemble, weights, thresholds), from the **leaderboard scores** of your testing-phase submissions, as long as each choice is one
fixed value applied identically to every image and you state it in the factsheet (Section 2). Say in `METHOD.md` which data your weights were trained on,
and answer the matching questions in the factsheet.

## How a re-run is judged

We re-run on an NVIDIA GPU of the RTX 3090 class, with PyTorch 2.4.0, CUDA 12.1 and Python 3.11 (the base image), and compare image by image with your leaderboard outputs and
metric by metric with your leaderboard scores. Tiny differences that come from the GPU or from rounding are tolerated within a tolerance we set from
measured run-to-run noise; anything larger is reviewed by a person before any decision. The tolerance is not published as a number.

## Check your repository before you hand in

```bash
python repro_check.py .                                        # layout, repro.yaml, weights link, signatures
python repro_check.py . --run --input /path/to/degraded --n 5  # also runs main() on 5 images and checks the outputs
python repro_check.py . --export-onnx                          # Track 2: also exports and checks the ONNX file
python repro_check.py . --download-weights                     # link-based repo: download the weights into weights/ first
```
No installation needed. To run it the way the organizers do (base image, **no network**), build the base image once and use it:

```bash
docker build -t eurs-base docker/
docker run --rm --network none --gpus all -v "$PWD":/pkg:ro -v /path/to/images:/in:ro -v /tmp/out:/out \
    eurs-base python /pkg/repro_check.py /pkg --run --input /in --output /out
```
(Leave out `--gpus all` if you have no GPU: the check then runs on the CPU, which can differ from a GPU run by one grey level in a few pixels.)
For a link-based repository run `python repro_check.py . --download-weights` first, so that the weights are in `weights/` for the test. Your `requirements.txt`
is installed on top of the base image before the real run. The checker validates the *form* of your hand-in; whether the results reproduce is decided
by the organizers by running it.

## What happens next

From October 28 we fetch each repository at the commit you name, build the environment, run `run.py` with no network, score
the outputs and compare with your reported testing-phase result. If something fails for a packaging reason we write to the
address in `repro.yaml`. Anything that is not a clean match is reviewed by a person; no team is disqualified automatically.
Final rankings are announced by October 30, 2026 (AoE).

## License

This template is released under the MIT License (see `LICENSE`). When you click **Use this template**, the `LICENSE` file is copied into your repository:
replace it (or at least its copyright line) with the license of **your own** code. The baseline model in `src/` and `run.py` is the organizers' reference baseline.

