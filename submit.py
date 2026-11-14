"""Create submission packages for MPI Sintel, KITTI-15 and Spring.

Examples:
    python submit.py --dataset kitti --cfg configs/submission/kitti.json \
        --ckpt experiments/sage_kitti_s42/final.pth --out submissions/kitti
    python submit.py --dataset sintel --cfg configs/submission/sintel.json \
        --ckpt experiments/sage_sintel_s42/final.pth --out submissions/sintel
    python submit.py --dataset spring --cfg configs/submission/spring-540p.json \
        --ckpt experiments/sage_spring_s42/final.pth --out submissions/spring

The output directory follows each benchmark's expected layout:

* KITTI:  ``<out>/flow/000000_10.png``
* Sintel: ``<out>/<clean|final>/<scene>/frameXXXX.flo``
* Spring: ``<out>/<scene>/flow_<direction>_<cam>/flow_<direction>_<cam>_XXXX.flo5``
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sage import bootstrap  # noqa: E402,F401

# The WAFT imports below resolve through the sys.path entry added by bootstrap.
# isort: off
import torch  # noqa: E402
from sage.checkpoint import load_sage_ckpt  # noqa: E402
from sage.config import add_common_arguments, parse_args  # noqa: E402
from sage.inference import infer_flow  # noqa: E402
from sage.model import SAGEModel  # noqa: E402
from utils import frame_utils  # noqa: E402
from utils.utils import load_ckpt  # noqa: E402

# isort: on


def build_model(args):
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


def _progress(seq, desc):
    """Wrap an iterable in a progress bar; a no-op without tqdm."""
    try:
        from tqdm import tqdm

        return tqdm(seq, desc=desc)
    except ImportError:
        return seq


@torch.no_grad()
def create_kitti(args, model, out: str) -> int:
    from dataloader.flow.kitti import KITTI

    flow_dir = os.path.join(out, "flow")
    os.makedirs(flow_dir, exist_ok=True)
    dataset = KITTI(
        split="testing", aug_params=None, root=os.path.join(args.data_root, "KITTI", "2015")
    )
    for i in _progress(range(len(dataset)), "kitti"):
        image1, image2, (frame_id, _) = dataset[i]
        flow = infer_flow(model, image1[None].cuda(), image2[None].cuda(), args.scale)
        flow = flow[0].permute(1, 2, 0).cpu().numpy()
        frame_utils.writeFlowKITTI(os.path.join(flow_dir, frame_id), flow)
    return len(dataset)


@torch.no_grad()
def create_sintel(args, model, out: str) -> int:
    from dataloader.flow.sintel import MpiSintel

    count = 0
    for dstype in ("clean", "final"):
        dataset = MpiSintel(
            split="test",
            aug_params=None,
            dstype=dstype,
            root=os.path.join(args.data_root, "Sintel"),
        )
        for i in _progress(range(len(dataset)), f"sintel/{dstype}"):
            image1, image2, (scene, frame) = dataset[i]
            flow = infer_flow(model, image1[None].cuda(), image2[None].cuda(), args.scale)
            flow = flow[0].permute(1, 2, 0).cpu().numpy()
            scene_dir = os.path.join(out, dstype, scene)
            os.makedirs(scene_dir, exist_ok=True)
            frame_utils.writeFlow(os.path.join(scene_dir, f"frame{frame:04d}.flo"), flow)
            count += 1
    return count


@torch.no_grad()
def create_spring(args, model, out: str) -> int:
    from dataloader.flow.spring import Spring

    dataset = Spring(split="test", aug_params=None, root=os.path.join(args.data_root, "spring"))
    for i in _progress(range(len(dataset)), "spring"):
        image1, image2, (frame, scene, cam, direction) = dataset[i]
        flow = infer_flow(model, image1[None].cuda(), image2[None].cuda(), args.scale)
        flow = flow[0].permute(1, 2, 0).cpu().numpy()
        seq_dir = os.path.join(out, scene, f"flow_{direction}_{cam}")
        os.makedirs(seq_dir, exist_ok=True)
        frame_utils.writeFlo5File(
            flow, os.path.join(seq_dir, f"flow_{direction}_{cam}_{frame:04d}.flo5")
        )
    return len(dataset)


def _count(root: str, suffix: str) -> int:
    return sum(len([f for f in files if f.endswith(suffix)]) for _, _, files in os.walk(root))


def verify(dataset: str, out: str, expected: int) -> None:
    """Check that the package contains the expected number of files."""
    if dataset == "kitti":
        count = _count(os.path.join(out, "flow"), ".png")
    elif dataset == "sintel":
        count = sum(_count(os.path.join(out, d), ".flo") for d in ("clean", "final"))
    else:
        count = _count(out, ".flo5")
    print(f"[submit] {dataset}: {count} flow files (expected {expected})")
    if count != expected:
        raise RuntimeError("submission package is incomplete")


def main():
    parser = argparse.ArgumentParser(description="Create a submission package")
    add_common_arguments(parser)
    parser.add_argument("--ckpt", required=True, type=str)
    parser.add_argument("--dataset", required=True, type=str, choices=["sintel", "kitti", "spring"])
    parser.add_argument("--out", required=True, type=str, help="output directory")
    parser.add_argument(
        "--scale", default=None, type=float, help="input resize exponent (Spring 540p uses -1)"
    )
    parser.add_argument("--baseline", action="store_true", help="submit a plain WAFT checkpoint")
    parser.add_argument(
        "--zip", action="store_true", help="also pack the output directory into <out>.zip"
    )
    args = parse_args(parser)
    if args.scale is None:
        args.scale = 0.0
    os.makedirs(args.out, exist_ok=True)

    model = build_model(args)
    if args.dataset == "kitti":
        n = create_kitti(args, model, args.out)
    elif args.dataset == "sintel":
        n = create_sintel(args, model, args.out)
    else:
        n = create_spring(args, model, args.out)

    verify(args.dataset, args.out, n)
    print(f"[submit] done: {os.path.abspath(args.out)}")
    if args.zip:
        import shutil

        archive = shutil.make_archive(os.path.abspath(args.out), "zip", args.out)
        print(f"[submit] archive: {archive}")


if __name__ == "__main__":
    main()
