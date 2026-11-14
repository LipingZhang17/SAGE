"""Evaluate SAGE on MPI Sintel, KITTI-15 or Spring.

Examples:
    python evaluate.py --cfg configs/eval/sintel.json \
        --ckpt experiments/sage_sintel_s42/final.pth --dataset sintel
    python evaluate.py --cfg configs/eval/kitti.json --ckpt ... --dataset kitti
    python evaluate.py --cfg configs/eval/spring.json --ckpt ... --dataset spring \
        --spring_split train

Use ``--baseline`` to evaluate a plain WAFT checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sage import bootstrap  # noqa: E402,F401

# The WAFT imports below resolve through the sys.path entry added by bootstrap.
# isort: off
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from sage.checkpoint import load_sage_ckpt  # noqa: E402
from sage.config import add_eval_arguments, parse_args  # noqa: E402
from sage.data.sintel_valid import resolve_valid_mask  # noqa: E402
from sage.inference import infer_flow  # noqa: E402
from sage.metrics.kitti import KittiAccum  # noqa: E402
from sage.metrics.sintel import OVERALL_LABELS, SintelAccum, bucket_masks, epe_map  # noqa: E402
from sage.metrics.spring import SPEED_BUCKETS, SpringAccum  # noqa: E402
from sage.model import SAGEModel  # noqa: E402
from utils import frame_utils  # noqa: E402
from utils.utils import load_ckpt  # noqa: E402

# isort: on


def build_model(args):
    """Build the evaluated model, either plain WAFT or SAGE."""
    if args.baseline:
        from model import fetch_model

        model = fetch_model(args)
        load_ckpt(model, args.ckpt)
    else:
        kwargs = dict(
            kernel=args.softselect_kernel,
            anisotropic=bool(args.anisotropic),
            use_content=bool(args.use_content),
            learn_mu=bool(args.learn_mu),
            init_seed=args.init_seed,
        )
        model = SAGEModel(
            args,
            lam=float(args.lam),
            k=int(args.softselect_k),
            learnable=True,
            softselect_kwargs=kwargs,
        )
        load_sage_ckpt(model, args.ckpt)
    return model.cuda().eval()


def _load_pair(p1: str, p2: str) -> tuple[torch.Tensor, torch.Tensor]:
    a = np.array(Image.open(p1).convert("RGB"))
    b = np.array(Image.open(p2).convert("RGB"))
    return (
        torch.from_numpy(a).permute(2, 0, 1).float(),
        torch.from_numpy(b).permute(2, 0, 1).float(),
    )


def _read_flo(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        magic = float(np.fromfile(f, np.float32, count=1)[0])
        if magic != 202021.25:
            raise ValueError(f"invalid .flo file: {path}")
        w = int(np.fromfile(f, np.int32, count=1)[0])
        h = int(np.fromfile(f, np.int32, count=1)[0])
        data = np.fromfile(f, np.float32, count=2 * w * h)
    return data.reshape(h, w, 2).astype(np.float32)


def _iter(seq, desc: str):
    """Wrap an iterable in a progress bar; a no-op without tqdm."""
    try:
        from tqdm import tqdm

        return tqdm(seq, desc=desc)
    except ImportError:
        return seq


# --------------------------------------------------------------------------- Sintel
def _sintel_frames(root: str):
    flow_root = os.path.join(root, "training", "flow")
    scenes = sorted(d for d in os.listdir(flow_root) if os.path.isdir(os.path.join(flow_root, d)))
    frames = []
    for scene in scenes:
        for name in sorted(os.listdir(os.path.join(flow_root, scene))):
            if name.endswith(".flo"):
                frames.append((scene, name[:-4]))
    return scenes, frames


def _sintel_gt(root: str, scene: str, frame: str, valid_mode: str):
    base = os.path.join(root, "training")
    flow_path = os.path.join(base, "flow", scene, frame + ".flo")
    if not os.path.isfile(flow_path):
        return None
    flow = _read_flo(flow_path).transpose(2, 0, 1)
    valid = resolve_valid_mask(flow.transpose(1, 2, 0), root, scene, frame, "training", valid_mode)
    occ_path = os.path.join(base, "occlusions", scene, frame + ".png")
    occ = (
        (np.array(Image.open(occ_path)) > 0).astype(np.float32)
        if os.path.isfile(occ_path)
        else np.zeros_like(valid)
    )
    return flow, valid, occ


@torch.no_grad()
def evaluate_sintel(args, model, device, root: str, max_frames: int = 0):
    _, frames = _sintel_frames(root)
    if max_frames > 0:
        frames = frames[:max_frames]
    scenes = sorted({scene for scene, _ in frames})
    accum = SintelAccum()
    for scene, frame in _iter(frames, "sintel"):
        gt = _sintel_gt(root, scene, frame, args.valid_mode)
        if gt is None:
            continue
        flow_np, valid, occ = gt
        n = int(frame.split("_")[-1])
        for pass_name in ("clean", "final"):
            d = os.path.join(root, "training", pass_name, scene)
            p1 = os.path.join(d, frame + ".png")
            p2 = os.path.join(d, f"frame_{n + 1:04d}.png")
            if not (os.path.isfile(p1) and os.path.isfile(p2)):
                continue
            masks = bucket_masks(flow_np, valid, occ)
            img1, img2 = _load_pair(p1, p2)
            pred = infer_flow(model, img1[None].to(device), img2[None].to(device), args.scale)
            epe = epe_map(
                pred[0].permute(1, 2, 0).float().cpu().numpy(), flow_np.transpose(1, 2, 0)
            )
            accum.update(pass_name, scene, epe, masks)
    return accum, scenes


def _print_sintel(accum: SintelAccum, scenes):
    for pass_name in ("clean", "final"):
        print(f"\nSintel {pass_name}")
        print(f"{'metric':<22}{'EPE':>10}{'n_px':>14}")
        print("-" * 46)
        for key, label in OVERALL_LABELS:
            print(
                f"{label:<22}{accum.epe(pass_name, key):>10.3f}{accum.counts[pass_name][key]:>14,}"
            )
    print(f"\n{'scene':<24}{'clean':>10}{'final':>10}")
    print("-" * 44)
    for scene in sorted(set(scenes)):
        print(
            f"{scene:<24}{accum.scene_epe('clean', scene):>10.3f}"
            f"{accum.scene_epe('final', scene):>10.3f}"
        )


# --------------------------------------------------------------------------- KITTI
def _kitti_frames(root: str):
    base = os.path.join(root, "training")
    img_dir = None
    for cand in ("image_2", "colored_0"):
        p = os.path.join(base, cand)
        if os.path.isdir(p) and any(f.endswith("_10.png") for f in os.listdir(p)):
            img_dir = p
            break
    if img_dir is None:
        raise FileNotFoundError(f"no image_2/ or colored_0/ under {base}")
    ids = sorted(f[: -len("_10.png")] for f in os.listdir(img_dir) if f.endswith("_10.png"))
    frames = []
    for fid in ids:
        p1 = os.path.join(img_dir, fid + "_10.png")
        p2 = os.path.join(img_dir, fid + "_11.png")
        gt = os.path.join(base, "flow_occ", fid + "_10.png")
        if os.path.isfile(p2) and os.path.isfile(gt):
            frames.append((fid, p1, p2, gt))
    return frames


@torch.no_grad()
def evaluate_kitti(args, model, device, root: str, max_frames: int = 0):
    frames = _kitti_frames(root)
    if max_frames > 0:
        frames = frames[:max_frames]
    accum = KittiAccum()
    for fid, p1, p2, gt_path in _iter(frames, "kitti"):
        img1, img2 = _load_pair(p1, p2)
        pred = infer_flow(model, img1[None].to(device), img2[None].to(device), args.scale)
        flow, valid = frame_utils.readFlowKITTI(gt_path)
        flow = np.ascontiguousarray(np.asarray(flow, dtype=np.float32).transpose(2, 0, 1))
        valid = (np.asarray(valid, dtype=np.float32) >= 0.5) & np.isfinite(flow).all(axis=0)
        flow = np.where(np.isfinite(flow), flow, 0.0).astype(np.float32)
        pred_hw2 = pred[0].permute(1, 2, 0).float().cpu().numpy()
        accum.update(fid, pred_hw2, flow.transpose(1, 2, 0), valid.astype(np.float32))
    return accum


# --------------------------------------------------------------------------- Spring
def _spring_frames(root: str, split: str):
    seq_root = os.path.join(root, split)
    if not os.path.isdir(seq_root):
        raise FileNotFoundError(f"Spring split not found: {seq_root}")
    frames = []
    for scene in sorted(os.listdir(seq_root)):
        scene_dir = os.path.join(seq_root, scene)
        if not os.path.isdir(scene_dir):
            continue
        for cam in ("left", "right"):
            frame_dir = os.path.join(scene_dir, f"frame_{cam}")
            if not os.path.isdir(frame_dir):
                continue
            n = len([f for f in os.listdir(frame_dir) if f.endswith(".png")])
            entries = [(i, "FW") for i in range(1, n)]
            entries += [(i, "BW") for i in reversed(range(2, n + 1))]
            for i, direction in entries:
                img1 = os.path.join(frame_dir, f"frame_{cam}_{i:04d}.png")
                step = i + 1 if direction == "FW" else i - 1
                img2 = os.path.join(frame_dir, f"frame_{cam}_{step:04d}.png")
                gt = os.path.join(
                    scene_dir, f"flow_{direction}_{cam}", f"flow_{direction}_{cam}_{i:04d}.flo5"
                )
                if os.path.isfile(img1) and os.path.isfile(img2) and os.path.isfile(gt):
                    frames.append((f"{scene}/{cam}/{direction}_{i:04d}", img1, img2, gt))
    return frames


def _spring_gt(path: str):
    flow = np.asarray(frame_utils.readFlo5Flow(path), dtype=np.float32)[::2, ::2]
    valid = (np.abs(flow[..., 0]) < 1000) & (np.abs(flow[..., 1]) < 1000)
    return np.ascontiguousarray(flow.transpose(2, 0, 1)), valid.astype(np.float32)


@torch.no_grad()
def evaluate_spring(args, model, device, root: str, split: str, max_frames: int = 0):
    frames = _spring_frames(root, split)
    if max_frames > 0:
        frames = frames[:max_frames]
    accum = SpringAccum()
    for fid, p1, p2, gt_path in _iter(frames, f"spring/{split}"):
        img1, img2 = _load_pair(p1, p2)
        pred = infer_flow(model, img1[None].to(device), img2[None].to(device), args.scale)
        gt, valid = _spring_gt(gt_path)
        pred_hw2 = pred[0].permute(1, 2, 0).float().cpu().numpy()
        accum.update(fid, pred_hw2, gt.transpose(1, 2, 0), valid)
    return accum


# --------------------------------------------------------------------------- report
def _print_kitti(acc: KittiAccum, root: str) -> None:
    m = acc.metrics()
    print(f"\nKITTI train split ({root})")
    print(f"{'F1-all (%)':<16}{m['fl_all']:>10.3f}{acc.px:>14,}")
    print(f"{'F1-epe (px)':<16}{m['fl_epe']:>10.3f}{acc.n_frame:>14,}")


def _print_spring(acc: SpringAccum, split: str) -> None:
    m = acc.metrics()
    print(f"\nSpring {split}")
    print(f"{'metric':<16}{'value':>10}{'n_px':>16}")
    print("-" * 44)
    print(f"{'EPE (px)':<16}{m['epe']:>10.3f}{m['n_px']:>16,}")
    print(f"{'1px (%)':<16}{m['px1']:>10.3f}{m['n_px']:>16,}")
    print(f"{'Fl (%)':<16}{m['fl']:>10.3f}{m['n_px']:>16,}")
    for name, lo, hi in SPEED_BUCKETS:
        hi_txt = f"{hi:g}" if hi is not None else "inf"
        print(
            f"{name.replace('_', '-'):<16}{m[name]:>10.3f}{int(m[name + '_n']):>16,}"
            f"   |gt| in [{lo:g}, {hi_txt})"
        )


def main():
    parser = argparse.ArgumentParser(description="Evaluate SAGE")
    add_eval_arguments(parser)
    parser.add_argument("--sintel_root", default="datasets/Sintel")
    parser.add_argument("--kitti_root", default="datasets/KITTI/2015")
    parser.add_argument("--spring_root", default="datasets/spring")
    parser.add_argument("--spring_split", default="train", choices=["train", "val", "train_val"])
    parser.add_argument("--valid_mode", default="auto", choices=["auto", "invalid", "flow"])
    parser.add_argument(
        "--max_frames", default=0, type=int, help="limit the number of evaluated frames (debug)"
    )
    parser.add_argument(
        "--save_json", default=None, type=str, help="write the metrics to a JSON file"
    )
    args = parse_args(parser)

    if not torch.cuda.is_available():
        raise SystemExit("evaluation requires a GPU")
    model = build_model(args)
    device = torch.device("cuda")

    if args.dataset == "sintel":
        accum, scenes = evaluate_sintel(args, model, device, args.sintel_root, args.max_frames)
        _print_sintel(accum, scenes)
        results = {p: {k: accum.epe(p, k) for k, _ in OVERALL_LABELS} for p in ("clean", "final")}
    elif args.dataset == "kitti":
        acc = evaluate_kitti(args, model, device, args.kitti_root, args.max_frames)
        _print_kitti(acc, args.kitti_root)
        results = acc.metrics()
    else:
        acc = evaluate_spring(
            args, model, device, args.spring_root, args.spring_split, args.max_frames
        )
        _print_spring(acc, args.spring_split)
        results = acc.metrics()

    if args.save_json:
        payload = {"dataset": args.dataset, "checkpoint": args.ckpt, "metrics": results}
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[sage] metrics written to {args.save_json}")


if __name__ == "__main__":
    main()
