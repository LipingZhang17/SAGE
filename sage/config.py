"""Configuration handling.

Settings live in a JSON file passed with ``--cfg``. Any command-line flag that
is not ``None`` overrides the corresponding JSON entry. Keys missing from both
sources fall back to :data:`SAGE_DEFAULTS`.
"""

from __future__ import annotations

import argparse
import json

SAGE_DEFAULTS = {
    # model
    "variant": "sage",
    "seed": 42,
    "amp": 1,
    "feature_encoder": "dinov3",
    "iterative_module": "vits",
    "algorithm": "waft-a2",
    "iters": 5,
    "var_min": 0,
    "var_max": 10,
    # soft read-out
    "lam": 1.0,
    "lam_warmup": 0,
    "softselect_k": 5,
    "softselect_kernel": "gaussian",
    "anisotropic": True,
    "use_content": False,
    "learn_mu": False,
    "init_seed": None,
    "aniso_reg": 0.0,
    # optimization
    "image_size": [432, 960],
    "scale": 0,
    "batch_size": 8,
    "grad_accum": 1,
    "num_steps": 50000,
    "lr": 4e-4,
    "wdecay": 1e-5,
    "epsilon": 1e-8,
    "clip": 1.0,
    "gamma": 0.85,
    "loss_type": "nll",
    # runtime
    "num_workers": 4,
    "print_freq": 100,
    "save_every": 1000,
    "save_mode": "last",
    "keep_ckpts": 5,
    "data_root": "datasets",
    "valid_mode": "auto",
    "restore_ckpt": None,
    "restore_sage_ckpt": None,
    "freeze_encoder": None,
}


def json_to_args(path: str) -> argparse.Namespace:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    args = argparse.Namespace()
    for k, v in data.items():
        setattr(args, k, v)
    return args


def build_args(cfg_path: str) -> argparse.Namespace:
    """Build a namespace from a JSON config plus defaults."""
    args = json_to_args(cfg_path)
    for k, v in SAGE_DEFAULTS.items():
        if not hasattr(args, k):
            setattr(args, k, v)
    return args


def parse_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    entry = parser.parse_args()
    args = build_args(entry.cfg)
    for k, v in vars(entry).items():
        if k != "cfg" and v is not None:
            setattr(args, k, v)
    if getattr(args, "backbone", None):
        args.feature_encoder = args.backbone
    if len(args.image_size) != 2:
        raise ValueError("image_size must be [height, width]")
    for key in ("batch_size", "grad_accum", "num_steps", "softselect_k"):
        if int(getattr(args, key, 0)) < 1:
            raise ValueError(f"{key} must be a positive integer")
    return args


def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--cfg", required=True, type=str, help="JSON config file")
    parser.add_argument(
        "--backbone",
        default=None,
        type=str,
        choices=["dinov3", "dav2", "twins"],
        help="override feature_encoder",
    )
    return parser


def add_eval_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    add_common_arguments(parser)
    parser.add_argument("--ckpt", required=True, type=str, help="checkpoint path")
    parser.add_argument("--dataset", required=True, type=str, choices=["sintel", "kitti", "spring"])
    parser.add_argument("--scale", default=None, type=float)
    parser.add_argument(
        "--baseline", action="store_true", help="evaluate a plain WAFT checkpoint (no SAGE head)"
    )
    return parser
