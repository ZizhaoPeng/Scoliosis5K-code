"""DINOv2-S/14 dense per-frame encoder and checkpoint loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class DINOv2DenseAdapter(nn.Module):
    """Return 16x16 normalized patch tokens for each valid RGB frame."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.register_buffer(
            "mean",
            torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()
        return self

    def _preprocess(self, frames: Tensor) -> Tensor:
        if frames.ndim != 4 or frames.shape[1:] != (3, 96, 64):
            raise ValueError("valid RGB frames must have shape [N,3,96,64]")
        if not bool(torch.isfinite(frames).all().item()):
            raise ValueError("valid RGB frames must be finite")
        if bool((frames < -1).any().item()) or bool((frames > 1).any().item()):
            raise ValueError("valid RGB frames must lie in [-1,1]")
        pixels = (frames.to(dtype=torch.float32) + 1.0) / 2.0
        pixels = F.interpolate(
            pixels, size=(224, 149), mode="bilinear", align_corners=False, antialias=True
        )
        left = (224 - 149) // 2
        right = 224 - 149 - left
        pixels = F.pad(pixels, (left, right, 0, 0), value=0.0)
        return (pixels - self.mean) / self.std

    def encode_dense(
        self, images: Tensor, frame_valid: Tensor
    ) -> tuple[Tensor, Tensor]:
        if images.ndim != 5 or images.shape[2:] != (3, 96, 64):
            raise ValueError("images must have shape [B,T,3,96,64]")
        if frame_valid.shape != images.shape[:2] or frame_valid.dtype is not torch.bool:
            raise ValueError("frame_valid must be bool [B,T]")
        if frame_valid.device != images.device:
            raise ValueError("frame_valid must share the image device")
        if not bool(frame_valid.any(dim=1).all().item()):
            raise ValueError("each sample must have one valid RGB frame")
        batch, time = images.shape[:2]
        flat_valid = frame_valid.reshape(-1)
        indices = torch.nonzero(flat_valid, as_tuple=False).flatten()
        frames = images.reshape(batch * time, 3, 96, 64).index_select(0, indices)
        with torch.no_grad():
            output = self.model.forward_features(self._preprocess(frames))
        if not isinstance(output, Mapping):
            raise TypeError("DINOv2 forward_features must return a mapping")
        patches = output.get("x_norm_patchtokens")
        if not isinstance(patches, Tensor) or patches.shape[1:] != (256, 384):
            raise ValueError("DINOv2 patch output must have shape [N,256,384]")
        if patches.dtype is not torch.float32 or patches.device != images.device:
            raise ValueError("DINOv2 patch output must be float32 on the input device")
        if not bool(torch.isfinite(patches).all().item()):
            raise ValueError("DINOv2 patch output must be finite")
        dense = torch.zeros(
            batch * time, 256, 384, device=patches.device, dtype=patches.dtype
        )
        dense.index_copy_(0, indices, patches)
        return dense.reshape(batch, time, 256, 384), frame_valid


def _load_state(path: Path) -> Mapping[str, Tensor]:
    if not path.is_file():
        raise FileNotFoundError(path)
    state: Any = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping) or not state:
        raise ValueError("DINOv2 checkpoint must be a non-empty state dict")
    return state


def build_dinov2_dense_adapter(checkpoint: str | Path) -> DINOv2DenseAdapter:
    """Load ViT-S/14 from a local checkpoint with upstream DINOv2 importable."""
    from dinov2.hub.backbones import dinov2_vits14

    model = dinov2_vits14(pretrained=False)
    model.load_state_dict(_load_state(Path(checkpoint)), strict=True)
    return DINOv2DenseAdapter(model)


__all__ = ["DINOv2DenseAdapter", "build_dinov2_dense_adapter"]
