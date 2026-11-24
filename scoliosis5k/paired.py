"""Connect the motion encoder to the paired classifier and training loop."""

from __future__ import annotations

import math
import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


BATCH_SIZE = 8
ADAPTATION_PARAMETERS = 73_728
HEAD_LR = 3e-4
ADAPTATION_LR = 1e-4
WEIGHT_DECAY = 0.01


class MotionPaired(nn.Module):
    def __init__(self, paired: nn.Module, adapted_rgb: nn.Module, *, seed: int) -> None:
        super().__init__()
        if getattr(adapted_rgb, "arm", None) not in ("D", "C"):
            raise ValueError("the encoder must expose the actual D or C motion arm")
        self.paired = paired
        self.adapted_rgb = adapted_rgb
        self.seed = seed

    def train(self, mode: bool = True) -> MotionPaired:
        super().train(mode)
        self.adapted_rgb.model.eval()
        return self

    def forward(self, batch: dict) -> dict:
        old_rgb, rgb_valid = batch["aux_images"], batch["aux_valid"]
        if old_rgb.shape[1:] != (24, 3, 224, 149) or old_rgb.dtype != torch.float32:
            raise ValueError("expected FP32 RGB pixels shaped [B,24,3,224,149] before letterbox padding")
        dense, valid = self.adapted_rgb.encode_dense(old_rgb, rgb_valid)
        return self.paired.forward_from_rgb_dense(batch["silhouettes"], batch["frame_valid"], dense, valid)


def build_motion_paired(adapted_rgb: nn.Module, *, seed: int) -> MotionPaired:
    """Seed the Paired classifier independently of the motion adapter."""
    from .distillation import CrossModalTeacherT1

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        paired = CrossModalTeacherT1(mode="paired")
    return MotionPaired(paired, adapted_rgb, seed=seed).train()


def build_public_motion_paired(*, arm: str, seed: int, checkpoint: str | Path) -> MotionPaired:
    """Compose pretrained DINOv2, a Q/V adapter and the Paired classifier."""
    from .motion import MotionConditionedLoRADenseAdapter
    from .dinov2 import build_dinov2_dense_adapter

    with torch.random.fork_rng(devices=[]):
        base = build_dinov2_dense_adapter(checkpoint)
    encoder = MotionConditionedLoRADenseAdapter(base.model, arm=arm, seed=seed + 100_000, checkpoint_blocks=True)
    return build_motion_paired(encoder, seed=seed)


def expected_updates(people: int, epochs: int) -> int:
    return math.ceil(people / BATCH_SIZE) * epochs


def _lr_multiplier(step: int, total_steps: int, warmup_fraction: float = 0.05) -> float:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    warmup = max(1, int(round(total_steps * warmup_fraction)))
    if step < warmup:
        return float(step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def make_optimizer(model: MotionPaired, total_updates: int) -> tuple:
    """Apply the original warmup/cosine multiplier to both parameter-group rates."""
    adapted = sum(parameter.numel() for parameter in model.adapted_rgb.parameters() if parameter.requires_grad)
    if adapted != ADAPTATION_PARAMETERS:
        raise ValueError(f"motion adaptation must expose exactly {ADAPTATION_PARAMETERS} trainable parameters")
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.dtype != torch.float32:
            raise ValueError("Paired and adapter parameters must remain FP32")
        lr = ADAPTATION_LR if name.startswith("adapted_rgb.") else HEAD_LR
        decay = 0.0 if parameter.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower() else WEIGHT_DECAY
        groups.setdefault((lr, decay), []).append(parameter)
    optimizer = torch.optim.AdamW([
        {"params": parameters, "lr": lr, "weight_decay": decay}
        for (lr, decay), parameters in groups.items()
    ])
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: _lr_multiplier(step, total_updates))
    return optimizer, schedule


def train_step(model: MotionPaired, batch: dict, target: torch.Tensor, optimizer, scheduler) -> dict:
    """Apply the paired classification and teacher losses to a batch of paired RGB and silhouette windows."""
    from .distillation import cosine_teacher_loss

    optimizer.zero_grad(set_to_none=True)
    out = model(batch)
    ce = F.cross_entropy(out["logits"], batch["labels"])
    auxiliary = cosine_teacher_loss(out["predicted_teacher"], target)
    objective = ce + auxiliary
    objective.backward()
    if any(parameter.requires_grad and parameter.grad is None for parameter in model.parameters()):
        raise RuntimeError("the Paired objective did not reach every trainable head or adapter tensor")
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    scheduler.step()
    return {"classification_CE": float(ce.detach()), "paired_cosine": float(auxiliary.detach()),
            "total_loss": float(objective.detach()), "preclip_grad_norm": float(norm),
            "correct": int((out["logits"].argmax(dim=1) == batch["labels"]).sum())}


def _delta_keys(model: MotionPaired) -> set[str]:
    return {name for name in model.state_dict() if name.startswith("paired.")} | {
        name for name, parameter in model.named_parameters()
        if name.startswith("adapted_rgb.") and parameter.requires_grad
    }


def save_delta(model: MotionPaired, path: Path, *, identity: dict) -> None:
    """Save the Paired classifier state and trainable motion-adapter parameters."""
    if identity.get("arm") != model.adapted_rgb.arm or identity.get("seed") != model.seed:
        raise ValueError("checkpoint identity differs from the actual model arm or seed")
    state = model.state_dict()
    delta = {name: state[name].detach().cpu().clone() for name in sorted(_delta_keys(model))}
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as out:
        torch.save({"identity": dict(identity), "delta": delta}, out)


def load_delta(model: MotionPaired, path: Path, *, identity: dict) -> MotionPaired:
    """Strictly reload into a fresh model with the same frozen base and arm identity."""
    if identity.get("arm") != model.adapted_rgb.arm or identity.get("seed") != model.seed:
        raise ValueError("checkpoint identity differs from the actual model arm or seed")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    delta = saved["delta"]
    if saved["identity"] != identity or set(delta) != _delta_keys(model):
        raise ValueError("checkpoint does not match the arm, source or exact delta keys")
    state = model.state_dict()
    for name, value in delta.items():
        original = state[name]
        if value.shape != original.shape or value.dtype != original.dtype or not torch.isfinite(value).all():
            raise ValueError(f"delta dtype, shape or values differ at {name}")
    state.update(delta)
    model.load_state_dict(state, strict=True)
    return model.eval()
