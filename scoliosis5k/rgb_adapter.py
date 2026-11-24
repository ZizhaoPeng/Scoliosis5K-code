"""DINOv2 encoder for pre-resized RGB pixels."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor
from torch.nn import functional as F

from .dinov2 import DINOv2DenseAdapter


PIXEL_SHAPE = (3, 224, 149)
PIXEL_ROUNDOFF_TOLERANCE = 8.0 * torch.finfo(torch.float32).eps


class ResolutionDINOv2DenseAdapter(DINOv2DenseAdapter):
    """Pad and normalize pixels, allowing float32 resize roundoff around [0,1]."""

    def _preprocess(self, frames: Tensor) -> Tensor:
        if frames.ndim != 4 or frames.shape[1:] != PIXEL_SHAPE:
            raise ValueError("valid RGB frames must have shape [N,3,224,149]")
        if not bool(torch.isfinite(frames).all().item()):
            raise ValueError("valid RGB frames must be finite")
        if (
            bool((frames < -PIXEL_ROUNDOFF_TOLERANCE).any().item())
            or bool((frames > 1.0 + PIXEL_ROUNDOFF_TOLERANCE).any().item())
        ):
            raise ValueError(
                "valid RGB pixels must lie in [0,1] apart from float32 resize roundoff"
            )
        pixels = F.pad(frames.to(dtype=torch.float32), (37, 38, 0, 0), value=0.0)
        return (pixels - self.mean) / self.std

    def encode_dense(
        self, images: Tensor, frame_valid: Tensor
    ) -> tuple[Tensor, Tensor]:
        if images.ndim != 5 or images.shape[2:] != PIXEL_SHAPE:
            raise ValueError("images must have shape [B,T,3,224,149]")
        if frame_valid.shape != images.shape[:2] or frame_valid.dtype is not torch.bool:
            raise ValueError("frame_valid must be bool [B,T]")
        if frame_valid.device != images.device:
            raise ValueError("frame_valid must share the image device")
        if not bool(frame_valid.any(dim=1).all().item()):
            raise ValueError("each sample must have one valid RGB frame")

        batch, time = images.shape[:2]
        indices = torch.nonzero(frame_valid.reshape(-1), as_tuple=False).flatten()
        frames = images.reshape(batch * time, *PIXEL_SHAPE).index_select(0, indices)
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
