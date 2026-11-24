"""Window-conditioned rank-2 + rank-2 Q/V adapters for DINOv2-S/14."""

from __future__ import annotations

from contextvars import ContextVar
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .rgb_adapter import (
    PIXEL_ROUNDOFF_TOLERANCE,
    ResolutionDINOv2DenseAdapter,
)


_BLOCK_COUNT = 12
_TOKEN_DIM = 384
_PATCH_COUNT = 256
_PATCH_SIZE = 14
_VALID_FRAME_MICROBATCH = 32
_PAD_LEFT = 37
_PAD_RIGHT = 38
_PIXEL_SHAPE = (3, 224, 149)
_ACTIVE_GATE: ContextVar[Tensor | None] = ContextVar("motion_adapter_gate", default=None)


def _validate_sequence(images: Tensor, frame_valid: Tensor) -> None:
    if images.ndim != 5 or images.shape[2:] != _PIXEL_SHAPE:
        raise ValueError("images must have shape [B,T,3,224,149]")
    if images.shape[0] == 0 or images.shape[1] == 0:
        raise ValueError("images must contain at least one batch item and frame")
    if frame_valid.shape != images.shape[:2] or frame_valid.dtype is not torch.bool:
        raise ValueError("frame_valid must be bool [B,T]")
    if frame_valid.device != images.device:
        raise ValueError("frame_valid must share the image device")
    if not bool(frame_valid.any(dim=1).all().item()):
        raise ValueError("each sample must have one valid RGB frame")


def _check_valid_pixels(images: Tensor, frame_valid: Tensor) -> None:
    valid = images.reshape(-1, *_PIXEL_SHAPE)[frame_valid.reshape(-1)]
    if not bool(torch.isfinite(valid).all().item()):
        raise ValueError("valid RGB frames must be finite")
    if (
        bool((valid < -PIXEL_ROUNDOFF_TOLERANCE).any().item())
        or bool((valid > 1.0 + PIXEL_ROUNDOFF_TOLERANCE).any().item())
    ):
        raise ValueError("valid RGB pixels must lie in [0,1] apart from resize roundoff")


def _compute_motion_gate_validated(images: Tensor, frame_valid: Tensor) -> Tensor:
    batch, time = images.shape[:2]
    frame_mask = frame_valid[:, :, None, None, None]
    safe_images = images.to(dtype=torch.float32).masked_fill(~frame_mask, 0.0)
    valid_count = frame_valid.sum(dim=1).to(dtype=torch.float32).view(batch, 1, 1, 1, 1)
    first_valid = frame_valid.to(dtype=torch.int64).argmax(dim=1)
    reference = safe_images.gather(
        1,
        first_valid.view(batch, 1, 1, 1, 1).expand(-1, 1, 3, 224, 149),
    )
    centered = (safe_images - reference).masked_fill(~frame_mask, 0.0)
    mean_offset = centered.sum(dim=1, keepdim=True) / valid_count
    pixel_difference = (centered - mean_offset).abs().mean(dim=2)
    pixel_difference = F.pad(pixel_difference, (_PAD_LEFT, _PAD_RIGHT, 0, 0))

    patch_sums = F.avg_pool2d(
        pixel_difference.reshape(batch * time, 1, 224, 224),
        kernel_size=_PATCH_SIZE,
        stride=_PATCH_SIZE,
    ).reshape(batch, time, _PATCH_COUNT) * (_PATCH_SIZE * _PATCH_SIZE)
    real_pixel_mask = F.pad(
        torch.ones((1, 1, 224, 149), dtype=torch.float32, device=images.device),
        (_PAD_LEFT, _PAD_RIGHT, 0, 0),
    )
    real_pixel_counts = F.avg_pool2d(
        real_pixel_mask, kernel_size=_PATCH_SIZE, stride=_PATCH_SIZE
    ).reshape(_PATCH_COUNT) * (_PATCH_SIZE * _PATCH_SIZE)
    real_patch_mask = real_pixel_counts > 0
    patch_values = patch_sums / real_pixel_counts.clamp_min(1.0).view(1, 1, _PATCH_COUNT)

    sample_scale = (
        patch_values * frame_valid[:, :, None] * real_patch_mask.view(1, 1, _PATCH_COUNT)
    ).sum(dim=(1, 2)) / (
        frame_valid.sum(dim=1).to(dtype=torch.float32) * real_patch_mask.sum()
    )
    safe_scale = torch.where(sample_scale > 0.0, sample_scale, torch.ones_like(sample_scale))
    gate = patch_values / (patch_values + safe_scale[:, None, None])
    gate = torch.where(sample_scale[:, None, None] > 0.0, gate, 0.5)
    gate = gate.masked_fill(~frame_valid[:, :, None], 0.5)
    gate = gate.masked_fill(~real_patch_mask.view(1, 1, _PATCH_COUNT), 0.5)
    cls_gate = torch.full((batch, time, 1), 0.5, dtype=torch.float32, device=images.device)
    return torch.cat((cls_gate, gate), dim=-1)


def compute_motion_gate(images: Tensor, frame_valid: Tensor) -> Tensor:
    """Return CLS and patch gates shaped ``[B,T,257]`` for resized RGB windows."""
    _validate_sequence(images, frame_valid)
    _check_valid_pixels(images, frame_valid)
    return _compute_motion_gate_validated(images, frame_valid)


class MotionConditionedQVLoRALinear(nn.Linear):
    """Fused QKV projection with two independently gated rank-2 Q/V updates."""

    def __init__(self, base: nn.Linear) -> None:
        if base.in_features != _TOKEN_DIM or base.out_features != 3 * _TOKEN_DIM:
            raise ValueError("DINOv2 QKV layers must have shape 384 -> 1152")

        nn.Module.__init__(self)
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.weight = base.weight
        self.bias = base.bias

        factor_options = {"dtype": base.weight.dtype, "device": "cpu"}
        for prefix in ("q", "v"):
            initial = torch.empty(4, _TOKEN_DIM, **factor_options)
            nn.init.kaiming_uniform_(initial, a=math.sqrt(5))
            first, second = initial.chunk(2, dim=0)
            setattr(self, f"lora_{prefix}_A0", nn.Parameter(first.to(base.weight.device)))
            setattr(self, f"lora_{prefix}_A1", nn.Parameter(second.to(base.weight.device)))
            for branch in (0, 1):
                setattr(
                    self,
                    f"lora_{prefix}_B{branch}",
                    nn.Parameter(
                        torch.zeros(
                            _TOKEN_DIM,
                            2,
                            dtype=base.weight.dtype,
                            device=base.weight.device,
                        )
                    ),
                )

    def forward(self, inputs: Tensor, gate: Tensor | None = None) -> Tensor:
        if gate is None:
            gate = _ACTIVE_GATE.get()
        if gate is None:
            raise RuntimeError("motion gate must be supplied for each DINOv2 block call")
        if inputs.ndim != 3 or gate.shape != inputs.shape[:2]:
            raise ValueError("motion gate must match the QKV input batch and token axes")

        qkv = F.linear(inputs, self.weight, self.bias)
        q_base, k_base, v_base = qkv.chunk(3, dim=-1)
        q0 = F.linear(F.linear(inputs, self.lora_q_A0), self.lora_q_B0)
        q1 = F.linear(F.linear(inputs, self.lora_q_A1), self.lora_q_B1)
        v0 = F.linear(F.linear(inputs, self.lora_v_A0), self.lora_v_B0)
        v1 = F.linear(F.linear(inputs, self.lora_v_A1), self.lora_v_B1)
        motion = gate.to(dtype=inputs.dtype).unsqueeze(-1)
        q_update = 2.0 * ((1.0 - motion) * q0 + motion * q1)
        v_update = 2.0 * ((1.0 - motion) * v0 + motion * v1)
        return torch.cat((q_base + q_update, k_base, v_base + v_update), dim=-1)


class MotionConditionedLoRADenseAdapter(ResolutionDINOv2DenseAdapter):
    """Encode resized RGB windows using static C or motion-conditioned D Q/V updates."""

    def __init__(
        self,
        model: nn.Module,
        *,
        arm: str,
        seed: int = 0,
        checkpoint_blocks: bool = True,
    ) -> None:
        if arm not in ("C", "D"):
            raise ValueError("arm must be 'C' (static) or 'D' (motion-conditioned)")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if type(checkpoint_blocks) is not bool:
            raise TypeError("checkpoint_blocks must be bool")
        super().__init__(model)
        self.arm = arm
        self.checkpoint_blocks = checkpoint_blocks
        self._blocks = self._validated_blocks()

        # Adapter initialization does not consume the caller's head RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            for block in self._blocks:
                block.attn.qkv = MotionConditionedQVLoRALinear(block.attn.qkv)

    def _validated_blocks(self) -> tuple[nn.Module, ...]:
        if getattr(self.model, "chunked_blocks", None) is not False:
            raise ValueError("motion-conditioned Q/V LoRA requires unchunked DINOv2 blocks")
        blocks = getattr(self.model, "blocks", None)
        if not isinstance(blocks, (nn.ModuleList, nn.Sequential)):
            raise ValueError("motion-conditioned Q/V LoRA requires a direct blocks sequence")
        if len(blocks) != _BLOCK_COUNT:
            raise ValueError("motion-conditioned Q/V LoRA requires exactly 12 DINOv2 blocks")
        if not callable(getattr(self.model, "prepare_tokens_with_masks", None)):
            raise ValueError("DINOv2 must expose prepare_tokens_with_masks")
        if not isinstance(getattr(self.model, "norm", None), nn.Module):
            raise ValueError("DINOv2 must expose its final norm module")

        for index, block in enumerate(blocks):
            attention = getattr(block, "attn", None)
            qkv = getattr(attention, "qkv", None)
            if not isinstance(qkv, nn.Linear):
                raise ValueError(f"block {index} must expose fused attn.qkv as nn.Linear")
            if qkv.in_features != _TOKEN_DIM or qkv.out_features != 3 * _TOKEN_DIM:
                raise ValueError("DINOv2 QKV projections must have shape 384 -> 1152")
        return tuple(blocks)

    def _run_block(self, block: nn.Module, tokens: Tensor, gate: Tensor) -> Tensor:
        context_token = _ACTIVE_GATE.set(gate)
        try:
            return block(tokens)
        finally:
            _ACTIVE_GATE.reset(context_token)

    def _forward_patch_tokens(self, pixels: Tensor, gate: Tensor) -> Tensor:
        tokens = self.model.prepare_tokens_with_masks(pixels, None)
        if not isinstance(tokens, Tensor) or tokens.ndim != 3:
            raise TypeError("prepare_tokens_with_masks must return [N,L,384] tokens")
        if tokens.shape[0] != pixels.shape[0] or tokens.shape[-1] != _TOKEN_DIM:
            raise ValueError("DINOv2 prepared tokens must have shape [N,L,384]")
        if tokens.shape[1] != _PATCH_COUNT + 1:
            raise ValueError("DINOv2 must return one CLS token and 256 patch tokens")

        for block in self._blocks:
            if self.training and self.checkpoint_blocks:
                tokens = checkpoint(
                    lambda hidden, motion_gate, current=block: self._run_block(
                        current, hidden, motion_gate
                    ),
                    tokens,
                    gate,
                    use_reentrant=False,
                )
            else:
                tokens = self._run_block(block, tokens, gate)

        normalized = self.model.norm(tokens)
        if (
            not isinstance(normalized, Tensor)
            or normalized.ndim != 3
            or normalized.shape[:2] != tokens.shape[:2]
            or normalized.shape[-1] != _TOKEN_DIM
        ):
            raise ValueError("DINOv2 final norm must return [N,257,384] tokens")
        return normalized[:, 1:, :]

    def encode_dense(self, images: Tensor, frame_valid: Tensor) -> tuple[Tensor, Tensor]:
        _validate_sequence(images, frame_valid)
        batch, time = images.shape[:2]
        if self.arm == "D":
            gates = _compute_motion_gate_validated(images, frame_valid)
        else:
            gates = torch.full(
                (batch, time, _PATCH_COUNT + 1),
                0.5,
                dtype=torch.float32,
                device=images.device,
            )
        indices = torch.nonzero(frame_valid.reshape(-1), as_tuple=False).flatten()
        flat_images = images.reshape(batch * time, *_PIXEL_SHAPE)
        flat_gates = gates.reshape(batch * time, _PATCH_COUNT + 1)
        patch_chunks = []
        for frame_indices in indices.split(_VALID_FRAME_MICROBATCH):
            frames = flat_images.index_select(0, frame_indices)
            pixels = self._preprocess(frames)
            frame_gates = flat_gates.index_select(0, frame_indices)
            patch_chunks.append(self._forward_patch_tokens(pixels, frame_gates))
        patches = torch.cat(patch_chunks, dim=0)
        if patches.shape[1:] != (_PATCH_COUNT, _TOKEN_DIM):
            raise ValueError("DINOv2 patch output must have shape [N,256,384]")
        if patches.dtype is not torch.float32 or patches.device != images.device:
            raise ValueError("DINOv2 patch output must be float32 on the input device")
        if not bool(torch.isfinite(patches).all().item()):
            raise ValueError("DINOv2 patch output must be finite")

        dense = torch.zeros(
            batch * time,
            _PATCH_COUNT,
            _TOKEN_DIM,
            device=patches.device,
            dtype=patches.dtype,
        )
        dense.index_copy_(0, indices, patches)
        return dense.reshape(batch, time, _PATCH_COUNT, _TOKEN_DIM), frame_valid


__all__ = [
    "MotionConditionedLoRADenseAdapter",
    "MotionConditionedQVLoRALinear",
    "compute_motion_gate",
]
