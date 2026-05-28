# PAID-SnowNet (Layer-1 honest refactor)

This is a refactored implementation of *PAID-SnowNet: A Physics-Aware Iterative
Denoising Network for Robust Snow Scene Restoration*. Compared to the original
release, this version:

* Implements **every formula and module in the paper** (SPM, SIRM, AdaGrad inner
  optimizer, physics-aware weight, U-Net completion, multi-scale loss, physics
  consistency loss).
* Provides a **real** optimizer ablation (Table 5) — the inner SIRM optimizer
  actually switches between SGD / Momentum / Adam / RMSprop / AdaGrad rather
  than re-labelling iteration counts.
* Provides **complete** module / component / loss ablations (Tables 3, 4, 5, 6,
  12, 13, 14, 15, 16).
* Provides **complete** robustness perturbations (Tables 17, 18, 19) including
  low resolution, bit-depth reduction, JPEG-like compression, snowflake
  morphology, opacity, and a wet/dry temperature proxy.
* Provides plotting scripts for Fig. 3, 5, 7, 8, 9, 11, 12 that **read from
  real JSON output** of the experiment scripts. If the JSON is missing, the
  scripts fail loudly rather than fabricating data.
* Removes the synthetic snow-pair generator from the dataset loader — without
  real Snow100K data on disk, loading raises `FileNotFoundError`.

## What this code does NOT do

* It does **not** ship with implementations of the 13 competitor methods
  compared against in Table 10. Implementing them is a substantial undertaking;
  for the Pareto plot (Fig. 11) you can supply their measured PSNR / parameter
  counts via `--competitor_data competitors.json`.
* It does **not** ship the Snow100K dataset. Download it from
  <https://sites.google.com/view/yunfuliu/desnownet> and arrange the files as
  described in `datasets/snow_dataset.py`.
* The downstream evaluation (`experiments/downstream_eval.py`) defaults to
  `--mode proxy`, in which case results are **feature-distance proxies**,
  clearly flagged `proxy: true` in the output JSON. To produce real YOLOv8
  mAP / DeepLabV3+ mIoU / ResNet-50 Top-1, install `ultralytics` and run
  with `--mode real`.

## Repository layout

```
PAID-SnowNet/
├── README.md
├── requirements.txt
├── config.py
├── train.py
├── test.py
├── models/                  # SPM, SIRM, U-Net, ResNet/SimpleCNN completion, physics weighting
├── datasets/                # Snow100K loader (requires real data)
├── losses/                  # MultiScaleLoss, PhysicsConsistencyLoss, VGGPerceptual, Edge
├── utils/                   # PSNR / SSIM / LPIPS / SnowDenoiseMetric; scattering utilities
├── analysis/                # Theorem 1 verification + iteration sweep
├── experiments/
│   ├── ablation_study.py    # Tables 3, 4, 5, 6, 12, 13, 14, 15
│   ├── ablation_loss.py     # Table 16 (requires a short training loop)
│   ├── robustness_eval.py   # Tables 17, 18, 19
│   ├── downstream_eval.py   # Tables 24-26 (real or proxy mode)
│   ├── scene_type_eval.py   # Table 11 (subfolder or manifest)
│   └── runtime_analysis.py  # Table 21
├── visualization/           # Fig. 3, 5, 7, 8, 9, 11, 12 plotters
└── scripts/
    ├── smoke_test.py        # structural verification with random tensors
    └── run_all_experiments.sh
```

## Quick start

```bash
pip install -r requirements.txt

# 1. structural smoke test (no data needed)
python scripts/smoke_test.py

# 2. train (requires Snow100K)
python train.py --data_root /path/to/Snow100K --epochs 200 --batch_size 32

# 3. test
python test.py --data_root /path/to/Snow100K --checkpoint ./checkpoints/best.pth

# 4. all experiments + figures
DATA_ROOT=/path/to/Snow100K CKPT=./checkpoints/best.pth \
    bash scripts/run_all_experiments.sh
```

## What the metrics mean

Every metric printed by these scripts is computed on whatever the model
actually produced. There are no hardcoded reference numbers. If you run with
a randomly-initialised model, you will see weak numbers; if you run after a
full 200-epoch training on Snow100K, you will see what the architecture is
actually capable of. Either way, the numbers reflect a real measurement.

If the numbers you observe disagree with those reported in the manuscript,
the right response is to (a) check training hyperparameters and runtime, then
(b) update the manuscript to reflect the measured numbers. Adjusting the code
to match the paper would be the wrong direction.
