"""Train SAGE on optical-flow datasets.

Example:
    python train.py --cfg configs/train/sage_things.json --seed 42 \
        --restore_ckpt model_zoo/waft/zero-shot.pth --out experiments/sage_things_s42

Outputs are written to ``--out``: ``config.json``, ``log.txt``, ``last.pth``
(resumable) and ``final.pth`` (weights only).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from contextlib import suppress

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sage import bootstrap  # noqa: E402,F401  (adds WAFT to sys.path)

# The WAFT imports below resolve through the sys.path entry added by bootstrap.
# isort: off
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.optim as optim  # noqa: E402
from criterion.loss import sequence_loss  # noqa: E402
from sage.checkpoint import atomic_save, load_sage_ckpt, safe_torch_load  # noqa: E402
from sage.config import add_common_arguments, parse_args  # noqa: E402
from sage.data import fetch_dataloader  # noqa: E402
from sage.model import SAGEModel  # noqa: E402
from sage.runtime import (  # noqa: E402
    AverageMeter,
    fmt_time,
    install_stop_handlers,
    restore_rng_state,
    rng_state,
    seed_worker,
    set_seed,
    stop_requested,
)
from utils.ddp_utils import init_ddp, setup_ddp  # noqa: E402
from utils.utils import load_ckpt  # noqa: E402

# isort: on

_NEW_AMP = hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler")
if _NEW_AMP:
    from torch.amp import GradScaler, autocast
else:  # torch < 2.4
    from torch.cuda.amp import GradScaler, autocast


def amp_context(enabled: bool):
    """Autocast context for CUDA, with the signature of the installed torch."""
    return autocast("cuda", enabled=enabled) if _NEW_AMP else autocast(enabled=enabled)


def l1_sequence_loss(output, flow_gt, valid, gamma: float = 0.8) -> torch.Tensor:
    """L1 (end-point error) loss over the prediction sequence."""
    n = len(output["flow"])
    valid = valid >= 0.5
    loss = 0.0
    for i in range(n):
        weight = gamma ** (n - i - 1)
        err = (output["flow"][i] - flow_gt).abs().sum(dim=1)
        if valid.sum() == 0:
            loss = loss + 0.0 * err.sum()
        else:
            loss = loss + weight * (valid * err).sum() / valid.sum()
    return loss


def configure_trainable(model: nn.Module, from_scratch: bool, freeze_encoder: bool | None) -> None:
    """Freeze the feature encoder unless training from scratch.

    ``freeze_encoder=None`` freezes the encoder when a checkpoint is restored
    and trains all parameters otherwise.
    """
    if freeze_encoder is None:
        freeze_encoder = not from_scratch
    if not freeze_encoder:
        return
    for name, module in model.named_children():
        for p in module.parameters():
            p.requires_grad = name != "encoder"


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


def _all_ranks_true(flag: bool, use_ddp: bool, device) -> bool:
    """Logical OR of a flag across ranks (no-op without DDP)."""
    if not use_ddp:
        return flag
    t = torch.tensor([1.0 if flag else 0.0], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return bool(t.item())


def _spawn_loader(loader, seed: int):
    """Rebuild a loader with a spawn context, safe after CUDA initialization."""
    workers = getattr(loader, "num_workers", 0)
    if workers <= 0:
        return loader
    gen = torch.Generator()
    gen.manual_seed(seed)
    return torch.utils.data.DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        sampler=loader.sampler,
        num_workers=workers,
        drop_last=loader.drop_last,
        pin_memory=True,
        prefetch_factor=4,
        multiprocessing_context="spawn",
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=gen,
    )


def _prune(path: str, keep: int) -> None:
    """Keep the ``keep`` most recent ``step_*.pth`` files."""
    if keep <= 0:
        return
    files = []
    for name in os.listdir(path):
        m = re.fullmatch(r"step_(\d+)\.pth", name)
        if m:
            files.append((int(m.group(1)), name))
    files.sort()
    for _, name in files[:-keep]:
        with suppress(OSError):
            os.remove(os.path.join(path, name))


def _make_ckpt(model, optimizer, scheduler, step: int, epoch: int) -> dict:
    return {
        "model": _unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "epoch": epoch,
        "rng": rng_state(),
    }


def train(args, rank: int = 0, world_size: int = 1, use_ddp: bool = False) -> None:
    out_dir = os.path.abspath(args.out)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    log_file = None
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2, ensure_ascii=False)
        log_file = open(os.path.join(out_dir, "log.txt"), "a", encoding="utf-8")  # noqa: SIM115

    def log(msg: str) -> None:
        if rank == 0:
            print(msg, flush=True)
            log_file.write(msg + "\n")
            log_file.flush()

    softselect_kwargs = {
        "kernel": args.softselect_kernel,
        "anisotropic": bool(args.anisotropic),
        "use_content": bool(args.use_content),
        "learn_mu": bool(args.learn_mu),
        "init_seed": args.init_seed,
    }
    model = SAGEModel(
        args,
        lam=float(args.lam),
        k=int(args.softselect_k),
        learnable=True,
        softselect_kwargs=softselect_kwargs,
    ).to(device)

    from_scratch = not (args.restore_ckpt or args.restore_sage_ckpt or args.resume)
    if args.restore_ckpt:
        load_ckpt(model, args.restore_ckpt)
        log(f"[sage] restored WAFT checkpoint: {args.restore_ckpt}")
    if args.restore_sage_ckpt:
        load_sage_ckpt(model, args.restore_sage_ckpt)
        log(f"[sage] restored SAGE checkpoint: {args.restore_sage_ckpt}")

    # A resumed run must keep the parameter set of the saved optimizer.
    freeze_encoder = args.freeze_encoder
    if args.resume and freeze_encoder is None:
        freeze_encoder = False
    configure_trainable(model, from_scratch=from_scratch, freeze_encoder=freeze_encoder)

    resume = None
    if args.resume:
        resume = safe_torch_load(args.resume, map_location="cpu")
        model.load_state_dict(resume["model"], strict=False)

    if use_ddp:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        args.lr,
        args.num_steps + 100,
        pct_start=0.05,
        cycle_momentum=False,
        anneal_strategy="linear",
    )

    start_step = 0
    if resume is not None:
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        start_step = int(resume.get("step", 0))
        restore_rng_state(resume.get("rng"))
        log(f"[sage] resumed from {args.resume} at step {start_step}")

    loader = fetch_dataloader(args, rank=rank, world_size=world_size, use_ddp=use_ddp)
    if use_ddp:
        loader = _spawn_loader(loader, args.seed)
        if hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(0)

    if len(loader.dataset) == 0:
        raise RuntimeError(f"dataset {args.dataset!r} is empty")

    n_gpu = max(torch.cuda.device_count(), 1)
    per_gpu = args.batch_size // n_gpu
    if per_gpu < 1:
        raise ValueError(f"batch_size={args.batch_size} is smaller than the GPU count {n_gpu}")

    log(
        f"[sage] backbone={args.feature_encoder} variant={args.variant} "
        f"lam={args.lam} k={args.softselect_k} kernel={args.softselect_kernel} "
        f"anisotropic={args.anisotropic} iters={args.iters}"
    )
    log(
        f"[sage] dataset={args.dataset} pairs={len(loader.dataset)} "
        f"batch={per_gpu}/gpu x {n_gpu} gpu x accum {args.grad_accum} "
        f"= {args.batch_size * args.grad_accum}"
    )
    log(f"[sage] trainable parameters: {sum(p.numel() for p in params):,}")
    log(f"[sage] run directory: {out_dir}")

    amp_enabled = bool(args.amp) and torch.cuda.is_available()
    scaler = (
        GradScaler("cuda", enabled=amp_enabled) if _NEW_AMP else GradScaler(enabled=amp_enabled)
    )

    loss_meter, epe_meter = AverageMeter(), AverageMeter()
    total_steps = start_step
    epoch = 0
    interrupted = False
    should_stop = False
    bad_windows = 0
    window_bad = False
    t0 = time.time()

    while not should_stop and total_steps < args.num_steps:
        if use_ddp and hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)
        epoch += 1
        window_bad = False
        for i_batch, blob in enumerate(loader):
            if stop_requested():
                interrupted = True
                should_stop = True
                break
            if args.lam_warmup > 0:
                _unwrap(model).lam = float(args.lam) * min(1.0, total_steps / args.lam_warmup)

            image1, image2, flow, valid = [x.to(device, non_blocking=True) for x in blob]
            with amp_context(amp_enabled):
                output = model(image1, image2, flow_gt=None if args.loss_type == "l1" else flow)
                if args.loss_type == "l1":
                    loss = l1_sequence_loss(output, flow, valid, args.gamma)
                else:
                    loss = sequence_loss(output, flow, valid, args.gamma)

            if args.aniso_reg > 0:
                prec_learned, prec_struct = _unwrap(model).precision_pair(image1, image2)
                if prec_learned is not None:
                    diff = prec_learned - prec_struct.detach()
                    loss = loss + args.aniso_reg * (diff**2).mean()

            epe = None
            if rank == 0 and valid.sum() > 0:
                err = ((flow - output["flow"][-1]) ** 2).sum(dim=1).sqrt()
                epe = float(((err * valid).sum() / valid.sum()).detach())

            window_bad = window_bad or not bool(torch.isfinite(loss.detach()).all())
            is_last = (i_batch + 1) % args.grad_accum == 0
            scaled = loss / args.grad_accum
            if use_ddp and not is_last:
                with model.no_sync():
                    scaler.scale(scaled).backward()
            else:
                scaler.scale(scaled).backward()
            if not is_last:
                continue

            if _all_ranks_true(window_bad, use_ddp, device):
                bad_windows += 1
                window_bad = False
                log(f"[sage] step {total_steps + 1}: non-finite loss, update skipped")
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                total_steps += 1
                if total_steps >= args.num_steps:
                    should_stop = True
                    break
                continue
            window_bad = False

            scaler.unscale_(optimizer)
            gnorm = torch.nn.utils.clip_grad_norm_(params, args.clip)
            if not bool(torch.isfinite(gnorm).all()):
                bad_windows += 1
                log(f"[sage] step {total_steps + 1}: non-finite gradient, update skipped")
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                scheduler.step()
                total_steps += 1
                if total_steps >= args.num_steps:
                    should_stop = True
                    break
                continue

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            total_steps += 1

            if rank == 0:
                loss_meter.update(float(loss.item()))
                if epe is not None:
                    epe_meter.update(epe)

                if args.print_freq > 0 and total_steps % args.print_freq == 0:
                    elapsed = time.time() - t0
                    done = max(total_steps - start_step, 1)
                    eta = elapsed / done * max(args.num_steps - total_steps, 0)
                    log(
                        f"step={total_steps}/{args.num_steps} "
                        f"loss={loss_meter.avg:.4f} epe={epe_meter.avg:.4f} "
                        f"lr={scheduler.get_last_lr()[0]:.2e} "
                        f"lam={_unwrap(model).lam:.3f} "
                        f"elapsed={fmt_time(elapsed)} eta={fmt_time(eta)}"
                    )
                    loss_meter.reset()
                    epe_meter.reset()

                if args.save_every > 0 and total_steps % args.save_every == 0:
                    ckpt = _make_ckpt(model, optimizer, scheduler, total_steps, epoch)
                    atomic_save(ckpt, os.path.join(out_dir, "last.pth"))
                    if args.save_mode == "steps":
                        atomic_save(ckpt, os.path.join(out_dir, f"step_{total_steps}.pth"))
                        _prune(out_dir, int(args.keep_ckpts))

            if total_steps >= args.num_steps:
                should_stop = True
                break

        if should_stop:
            break
        if args.grad_accum > 1:
            optimizer.zero_grad(set_to_none=True)

    if use_ddp and dist.is_initialized():
        dist.barrier()

    if rank == 0:
        if interrupted:
            ckpt = _make_ckpt(model, optimizer, scheduler, total_steps, epoch)
            atomic_save(ckpt, os.path.join(out_dir, "last.pth"))
            log(f"[sage] interrupted at step {total_steps}; last.pth saved")
        else:
            atomic_save(_unwrap(model).state_dict(), os.path.join(out_dir, "final.pth"))
            log(f"[sage] finished; final.pth saved to {out_dir}")
        if bad_windows:
            log(f"[sage] skipped optimization steps: {bad_windows}")
        log_file.close()
    if use_ddp and dist.is_initialized():
        dist.destroy_process_group()


def main(rank: int, world_size: int, args, use_ddp: bool) -> None:
    set_seed(int(args.seed))
    install_stop_handlers()
    if use_ddp:
        setup_ddp(rank, world_size)
    train(args, rank=rank, world_size=world_size, use_ddp=use_ddp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAGE")
    add_common_arguments(parser)
    parser.add_argument(
        "--out", required=True, type=str, help="output directory, e.g. experiments/sage_things_s42"
    )
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument(
        "--dataset", default=None, type=str, help="override the dataset entry of the config"
    )
    parser.add_argument(
        "--data_root", default=None, type=str, help="root directory that contains the datasets"
    )
    parser.add_argument(
        "--restore_ckpt", default=None, type=str, help="WAFT checkpoint to initialize from"
    )
    parser.add_argument(
        "--restore_sage_ckpt",
        default=None,
        type=str,
        help="SAGE checkpoint of a previous training stage",
    )
    parser.add_argument("--resume", default=None, type=str, help="resume from a last.pth file")
    parser.add_argument("--variant", default=None, type=str, choices=["sage", "sage-iso", "waft"])
    parser.add_argument("--num_steps", default=None, type=int)
    parser.add_argument(
        "--batch_size", default=None, type=int, help="global batch size across all GPUs"
    )
    parser.add_argument("--grad_accum", default=None, type=int)
    parser.add_argument("--lr", default=None, type=float)
    parser.add_argument("--lam", default=None, type=float)
    parser.add_argument("--softselect_k", default=None, type=int)
    parser.add_argument(
        "--softselect_kernel", default=None, type=str, choices=["gaussian", "laplace", "mixture"]
    )
    parser.add_argument("--anisotropic", default=None, type=int, choices=[0, 1])
    parser.add_argument("--use_content", default=None, type=int, choices=[0, 1])
    parser.add_argument("--learn_mu", default=None, type=int, choices=[0, 1])
    parser.add_argument("--init_seed", default=None, type=int)
    parser.add_argument("--aniso_reg", default=None, type=float)
    parser.add_argument("--freeze_encoder", default=None, type=int, choices=[0, 1])
    parser.add_argument(
        "--amp",
        default=None,
        type=int,
        choices=[0, 1],
        help="enable mixed precision (default: 1)",
    )
    parser.add_argument("--save_every", default=None, type=int)
    parser.add_argument("--save_mode", default=None, type=str, choices=["last", "steps"])
    parser.add_argument("--keep_ckpts", default=None, type=int)
    parser.add_argument("--num_workers", default=None, type=int)
    parser.add_argument("--print_freq", default=None, type=int)
    parser.add_argument(
        "--gpus", default=None, type=str, help="comma separated GPU ids, e.g. 0,1,2,3"
    )
    args = parse_args(parser)
    for key in ("anisotropic", "use_content", "learn_mu", "freeze_encoder", "amp"):
        if getattr(args, key) is not None:
            setattr(args, key, bool(getattr(args, key)))
    args.seed = int(args.seed)
    if getattr(args, "gpus", None):
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    if not torch.cuda.is_available():
        raise SystemExit("training requires at least one CUDA GPU")
    _, world_size = init_ddp()
    if world_size > 1:
        ctx = mp.spawn(main, nprocs=world_size, args=(world_size, args, True), join=False)
        ctx.join()
    else:
        main(0, 1, args, False)
    print("Done.")
