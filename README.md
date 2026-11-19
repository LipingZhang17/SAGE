# Mapping the Design Space of Feature Sampling for Optical Flow

This repository contains the official code of *Mapping the Design Space of
Feature Sampling for Optical Flow*.

## Requirements

The code requires PyTorch 2.0 or newer. Python 3.10 is recommended.

```bash
conda create -n sage python=3.10 -y
conda activate sage
pip install -r requirements.txt
```

## Installation

External projects are included as git submodules. Clone with submodules:

```bash
git clone https://github.com/LipingZhang17/SAGE.git --recursive
cd SAGE
```

If the repository is already checked out:

```bash
git submodule update --init --recursive
```

`third_party/WAFT` is the host model. Its own DINOv3 submodule, used by the
default feature encoder, is fetched by the same recursive command. The
submodule directory is empty in the source archive; run the command above once
to populate it.

## Model Zoo

`model_zoo/` holds all pretrained weights and is not tracked by git.

```
model_zoo/
├── waft/          WAFT initialization checkpoints (chairs, things, zero-shot)
└── dinov3-ckpts/  dinov3_vits16_pretrain_lvd1689m-08c60483.pth
```

The DINOv3 weight file can also be pointed to with `DINOV3_WEIGHTS_VITS`, and
any directory can be used with `DINOV3_WEIGHTS_DIR`.

## Datasets

All datasets are expected under `datasets/`, the default `data_root` of the
training and submission configurations; `evaluate.py` takes explicit
`--sintel_root`, `--kitti_root` and `--spring_root` options with the same
defaults. Download links: [FlyingChairs](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html),
[FlyingThings3D](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html),
[Sintel](http://sintel.is.tue.mpg.de/), [KITTI](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow),
[HD1K](http://hci-benchmark.iwr.uni-heidelberg.de/),
[TartanAir](https://theairlab.org/tartanair-dataset/) and
[Spring](https://spring-benchmark.org/).

The directory layout expected by the data loaders is:

```
datasets/
├── FlyingChairs/
│   ├── FlyingChairs_release/data/
│   │   ├── 00001_img1.ppm
│   │   ├── 00001_img2.ppm
│   │   └── 00001_flow.flo
│   └── chairs_split.txt
├── FlyingThings3D/
│   ├── frames_cleanpass/TRAIN/<letter>/<number>/left/*.png
│   ├── frames_finalpass/TRAIN/<letter>/<number>/left/*.png
│   └── optical_flow/TRAIN/<letter>/<number>/
│       ├── into_future/left/*.pfm
│       └── into_past/left/*.pfm
├── TartanAir/
│   └── <env>/<scene>/<variant>/
│       ├── image_left/*.png
│       └── flow/<i>_<j>_flow.npy, <i>_<j>_mask.npy
├── Sintel/
│   ├── training/
│   │   ├── clean/<scene>/frame_XXXX.png
│   │   ├── final/<scene>/frame_XXXX.png
│   │   ├── flow/<scene>/frame_XXXX.flo
│   │   ├── invalid/<scene>/frame_XXXX.png
│   │   └── occlusions/<scene>/frame_XXXX.png
│   └── test/
│       ├── clean/<scene>/frame_XXXX.png
│       └── final/<scene>/frame_XXXX.png
├── KITTI/2015/
│   ├── training/
│   │   ├── image_2/<id>_10.png, <id>_11.png
│   │   └── flow_occ/<id>_10.png
│   └── testing/
│       └── image_2/<id>_10.png, <id>_11.png
├── HD1K/
│   ├── hd1k_input/image_2/<sequence>_<frame>.png
│   └── hd1k_flow_gt/flow_occ/<sequence>_<frame>.png
└── spring/
    ├── train/<scene>/
    │   ├── frame_left/frame_left_XXXX.png
    │   ├── frame_right/frame_right_XXXX.png
    │   ├── flow_FW_left/flow_FW_left_XXXX.flo5
    │   ├── flow_BW_left/flow_BW_left_XXXX.flo5
    │   ├── flow_FW_right/flow_FW_right_XXXX.flo5
    │   └── flow_BW_right/flow_BW_right_XXXX.flo5
    ├── val/<scene>/        same layout as train/, smaller split
    └── test/<scene>/
        ├── frame_left/frame_left_XXXX.png
        └── frame_right/frame_right_XXXX.png
```

Notes on individual datasets:

* **FlyingChairs**: `chairs_split.txt` distinguishes training (1) from
  validation (2) samples; `FlyingChairs_release/data` must contain all
  `*_img1.ppm`, `*_img2.ppm` and `*_flow.flo` files.
* **FlyingThings3D**: both `frames_cleanpass` and `frames_finalpass` are used;
  flows are read from `optical_flow` in both `into_future` and `into_past`
  directions.
* **Sintel**: at evaluation time the devkit `invalid/` mask is used as the
  valid-pixel mask when present, otherwise pixels with `|fx| >= 1000` or
  `|fy| >= 1000` are treated as invalid. `occlusions/` is only used at
  evaluation time.
* **KITTI**: the 2015 flow benchmark. Training and evaluation use
  `training/flow_occ`; `testing/` has no ground truth and is only used for
  submission.
* **HD1K**: file names are `<6-digit sequence>_<6-digit frame>.png`; sequences
  are read in increasing order until the first missing sequence.
* **TartanAir**: each `<env>/<scene>/<variant>` directory is one trajectory.
  Training reads adjacent pairs from `image_left`; the ground-truth flow and
  valid mask must be precomputed as `flow/<i>_<j>_flow.npy` and
  `flow/<i>_<j>_mask.npy`.
* **Spring**: `flow_*.flo5` are HDF5 files at 4K resolution; they are
  subsampled by 2 to match the 1920x1080 images. Training uses `train/` and
  `val/`; `test/` is used for submission only.

## Training

Training follows the WAFT schedule: TartanAir, FlyingChairs, FlyingThings3D,
then benchmark-specific fine-tuning.

```bash
# 1. TartanAir, from scratch
python train.py --cfg configs/train/sage_tartanair.json --seed 42 \
    --out experiments/sage_tartanair_s42 --gpus 0,1,2,3

# 2. FlyingChairs
python train.py --cfg configs/train/sage_chairs.json --seed 42 \
    --restore_sage_ckpt experiments/sage_tartanair_s42/final.pth \
    --out experiments/sage_chairs_s42 --gpus 0,1,2,3

# 3. FlyingThings3D
python train.py --cfg configs/train/sage_things.json --seed 42 \
    --restore_sage_ckpt experiments/sage_chairs_s42/final.pth \
    --out experiments/sage_things_s42 --gpus 0,1,2,3

# 4. Benchmark fine-tuning
python train.py --cfg configs/train/sage_sintel.json --seed 42 \
    --restore_sage_ckpt experiments/sage_things_s42/final.pth \
    --out experiments/sage_sintel_s42 --gpus 0,1,2,3

python train.py --cfg configs/train/sage_kitti.json --seed 42 \
    --restore_sage_ckpt experiments/sage_things_s42/final.pth \
    --out experiments/sage_kitti_s42 --gpus 0,1,2,3

python train.py --cfg configs/train/sage_spring_540p.json --seed 42 \
    --restore_sage_ckpt experiments/sage_things_s42/final.pth \
    --out experiments/sage_spring_s42 --gpus 0,1,2,3
```

To start from released WAFT weights instead of the TartanAir stage, pass
`--restore_ckpt model_zoo/waft/zero-shot.pth` and skip step 1.

Each run writes `config.json`, `log.txt`, `last.pth` (resumable) and
`final.pth` (weights only) to `--out`. Resume with:

```bash
python train.py --cfg configs/train/sage_things.json --seed 42 \
    --out experiments/sage_things_s42 \
    --resume experiments/sage_things_s42/last.pth
```

`batch_size` in every configuration is the global batch size; it is split
evenly over the visible GPUs and multiplied by gradient accumulation. The
isotropic ablation is `configs/train/sage_things_iso.json`; the hard-warp
baseline is obtained with `--lam 0`.

## Evaluation

```bash
python evaluate.py --cfg configs/eval/sintel.json \
    --ckpt experiments/sage_sintel_s42/final.pth --dataset sintel

python evaluate.py --cfg configs/eval/kitti.json \
    --ckpt experiments/sage_kitti_s42/final.pth --dataset kitti

python evaluate.py --cfg configs/eval/spring.json \
    --ckpt experiments/sage_spring_s42/final.pth --dataset spring \
    --spring_split train
```

Sintel reports EPE over all / matched / unmatched pixels, distance-to-boundary
buckets and speed buckets for both passes. KITTI reports `F1-all` and
`F1-epe`. Spring reports EPE, `1px`, `Fl` and speed buckets. All metrics are
averaged over valid pixels. Add `--save_json results.json` to store them, and
`--baseline` to evaluate a plain WAFT checkpoint.

## Submission

```bash
python submit.py --dataset sintel --cfg configs/submission/sintel.json \
    --ckpt experiments/sage_sintel_s42/final.pth --out submissions/sintel --zip

python submit.py --dataset kitti --cfg configs/submission/kitti.json \
    --ckpt experiments/sage_kitti_s42/final.pth --out submissions/kitti --zip

python submit.py --dataset spring --cfg configs/submission/spring-540p.json \
    --ckpt experiments/sage_spring_s42/final.pth --out submissions/spring --zip
```

The output follows the benchmark layouts: `flow/*.png` for KITTI,
`<clean|final>/<scene>/frameXXXX.flo` for Sintel and
`<scene>/flow_<direction>_<cam>/*.flo5` for Spring.

## Acknowledgements

This code builds on [WAFT](https://github.com/princeton-vl/WAFT),
[RAFT](https://github.com/princeton-vl/RAFT),
[DepthAnythingV2](https://github.com/DepthAnything/Depth-Anything-V2) and
[DINOv3](https://github.com/facebookresearch/dinov3). We thank the authors for
releasing their code.

## License

SAGE is released under the Apache License 2.0. See [LICENSE](LICENSE) for the
full text. The submodules in `third_party/` are distributed under their own
licenses.
