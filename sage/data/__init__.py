"""Training data.

The dataset classes come from WAFT; the composition and the ``DataLoader`` are
built here so that every path is resolved against ``data_root`` and the batch
and worker settings are the same for all datasets.
"""

from __future__ import annotations

import os

import torch
import torch.utils.data as data
from torch.utils.data.distributed import DistributedSampler


def _root(args, *parts: str) -> str:
    return os.path.join(str(args.data_root), *parts)


def _check(path: str, name: str) -> None:
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{name} not found: {path} (see datasets/ in README.md)")


def _aug(args, min_scale: float, max_scale: float, flip: bool = True) -> dict:
    return {
        "crop_size": args.image_size,
        "min_scale": args.scale + min_scale,
        "max_scale": args.scale + max_scale,
        "do_flip": flip,
    }


def build_dataset(args):
    """Build the training dataset named by ``args.dataset``."""
    from dataloader.flow.chairs import FlyingChairs
    from dataloader.flow.kitti import KITTI
    from dataloader.flow.sintel import MpiSintel
    from dataloader.flow.spring import Spring
    from dataloader.flow.things import FlyingThings3D
    from dataloader.stereo.tartanair import TartanAir

    key = str(args.dataset).strip().lower()

    if key == "chairs":
        return FlyingChairs(
            _aug(args, -0.1, 1.0), split="training", root=_root(args, "FlyingChairs")
        )

    if key == "things":
        aug = _aug(args, -0.4, 0.8)
        root = _root(args, "FlyingThings3D")
        _check(root, "FlyingThings3D")
        return FlyingThings3D(aug, root=root, dstype="frames_cleanpass") + FlyingThings3D(
            aug, root=root, dstype="frames_finalpass"
        )

    if key == "tartanair":
        # Adjacent pairs only: the upstream precomputed
        # ``flow/<i>_<j>_flow.npy`` and ``<i>_<j>_mask.npy`` are required.
        root = _root(args, "TartanAir")
        _check(root, "TartanAir")
        return TartanAir(_aug(args, -0.2, 0.2), root=root)

    if key == "sintel":
        aug = _aug(args, -0.2, 0.6)
        root = _root(args, "Sintel")
        _check(root, "Sintel")
        return MpiSintel(aug, split="training", root=root, dstype="clean") + MpiSintel(
            aug, split="training", root=root, dstype="final"
        )

    if key == "kitti":
        root = _root(args, "KITTI", "2015")
        _check(root, "KITTI 2015")
        return KITTI(_aug(args, -0.2, 0.4, flip=False), split="training", root=root)

    if key == "tskh":
        return _build_tskh(args)

    if key == "spring_train":
        aug = _aug(args, 0.0, 0.2)
        root = _root(args, "spring")
        _check(root, "Spring")
        return Spring(aug, split="train", root=root) + Spring(aug, split="val", root=root)

    raise ValueError(f"unknown dataset: {args.dataset!r}")


def _build_tskh(args):
    """Sintel + KITTI + HD1K + Things, weighted as in WAFT."""
    from dataloader.flow.hd1k import HD1K
    from dataloader.flow.kitti import KITTI
    from dataloader.flow.sintel import MpiSintel
    from dataloader.flow.things import FlyingThings3D

    things_root = _root(args, "FlyingThings3D")
    sintel_root = _root(args, "Sintel")
    kitti_root = _root(args, "KITTI", "2015")
    hd1k_root = _root(args, "HD1K")
    for path, name in (
        (things_root, "FlyingThings3D"),
        (sintel_root, "Sintel"),
        (kitti_root, "KITTI 2015"),
        (hd1k_root, "HD1K"),
    ):
        _check(path, name)

    things_aug = _aug(args, -0.4, 0.8)
    things = FlyingThings3D(
        things_aug, root=things_root, dstype="frames_cleanpass"
    ) + FlyingThings3D(things_aug, root=things_root, dstype="frames_finalpass")

    sintel_aug = _aug(args, -0.2, 0.6)
    sintel_clean = MpiSintel(sintel_aug, split="training", root=sintel_root, dstype="clean")
    sintel_final = MpiSintel(sintel_aug, split="training", root=sintel_root, dstype="final")

    kitti = KITTI(_aug(args, -0.3, 0.5), split="training", root=kitti_root)
    hd1k = HD1K(_aug(args, -0.5, 0.2), root=hd1k_root)

    return 20 * sintel_clean + 20 * sintel_final + 80 * kitti + 30 * hd1k + things


def fetch_dataloader(args, rank: int = 0, world_size: int = 1, use_ddp: bool = False):
    """Build the training loader for ``args.dataset``."""
    dataset = build_dataset(args)
    num_workers = int(args.num_workers or 0)
    n_gpu = max(torch.cuda.device_count(), 1)
    per_gpu = max(args.batch_size // n_gpu, 1)

    if use_ddp:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        return data.DataLoader(
            dataset,
            batch_size=per_gpu,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
    return data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
