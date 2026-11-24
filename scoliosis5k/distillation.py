"""T1 wrapper with an auxiliary silhouette-to-RGB teacher prediction head."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .video import MultimodalTransformerT1


MODES = ("baseline", "paired", "shuffled")
SILHOUETTE_DIM = 128
TEACHER_DIM = 384


class _ExternalRGBEncoder(nn.Module):
    def forward(self, *_args, **_kwargs) -> None:
        raise RuntimeError("provide dense RGB tokens through forward_from_rgb_dense")


def new_t1() -> MultimodalTransformerT1:
    return MultimodalTransformerT1(
        _ExternalRGBEncoder(),
        rgb_token_dim=TEACHER_DIM,
        common_dim=SILHOUETTE_DIM,
        patch_size=8,
        silhouette_heads=4,
        silhouette_layers=1,
        attention="divided",
        pooling="attentive",
        dropout=0.1,
        position_encoding="separate_axes",
        freeze_rgb_encoder=True,
    )


class CrossModalTeacherT1(nn.Module):
    """Keep T1 logits unchanged and predict a normalized RGB teacher vector.

    The mode labels the training arm only. Every instance has the same T1
    base and predictor; the caller chooses whether to include the auxiliary
    loss and which TRAIN-only target to supply.
    """

    def __init__(self, mode: str = "baseline") -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError("mode must be baseline, paired, or shuffled")
        self.mode = mode
        self.base = new_t1()
        self.predictor = nn.Sequential(
            nn.LayerNorm(SILHOUETTE_DIM),
            nn.Linear(SILHOUETTE_DIM, SILHOUETTE_DIM),
            nn.GELU(),
            nn.Linear(SILHOUETTE_DIM, TEACHER_DIM),
        )

    def forward_from_rgb_dense(
        self,
        silhouettes: Tensor,
        frame_valid: Tensor,
        rgb_dense: Tensor,
        rgb_dense_valid: Tensor,
    ) -> dict[str, Tensor]:
        outputs = self.base.forward_from_rgb_dense(
            silhouettes, frame_valid, rgb_dense, rgb_dense_valid
        )
        predicted_teacher = F.normalize(
            self.predictor(outputs["silhouette_pooled"]).float(),
            p=2,
            dim=-1,
            eps=1e-12,
        )
        return {**outputs, "predicted_teacher": predicted_teacher}

    def forward(
        self,
        silhouettes: Tensor,
        frame_valid: Tensor,
        rgb_dense: Tensor,
        rgb_dense_valid: Tensor,
    ) -> dict[str, Tensor]:
        return self.forward_from_rgb_dense(
            silhouettes, frame_valid, rgb_dense, rgb_dense_valid
        )


def cosine_teacher_loss(predicted_teacher: Tensor, target: Tensor) -> Tensor:
    """Mean cosine distance with an explicitly detached, validated target."""

    if not isinstance(predicted_teacher, Tensor) or not isinstance(target, Tensor):
        raise TypeError("predicted_teacher and target must be tensors")
    if predicted_teacher.ndim != 2 or target.ndim != 2:
        raise ValueError("teacher predictions and targets must have shape [B,D]")
    if predicted_teacher.shape != target.shape or predicted_teacher.shape[0] < 1:
        raise ValueError(
            "teacher predictions and targets must have the same non-empty shape"
        )
    prediction = predicted_teacher.float()
    target_value = target.detach().to(device=prediction.device, dtype=torch.float32)
    if not torch.isfinite(prediction).all():
        raise ValueError("teacher predictions contain non-finite values")
    if not torch.isfinite(target_value).all():
        raise ValueError("teacher targets contain non-finite values")
    target_norm = torch.linalg.vector_norm(target_value, dim=-1)
    if torch.any(target_norm <= 1e-12):
        raise ValueError("teacher targets must be non-degenerate")
    return (1.0 - F.cosine_similarity(prediction, target_value, dim=-1)).mean()


def masked_token_mean(dense: Tensor, frame_valid: Tensor) -> Tensor:
    """Mean valid-frame patch tokens per window in float32.

    ``dense`` has shape ``[B,T,P,D]``. Invalid frames are selected away with
    ``where`` before summation, so even NaNs in padded frames cannot leak into
    the result.
    """

    if not isinstance(dense, Tensor) or not isinstance(frame_valid, Tensor):
        raise TypeError("dense and frame_valid must be tensors")
    if dense.ndim != 4 or frame_valid.ndim != 2:
        raise ValueError("dense and frame_valid must have shapes [B,T,P,D] and [B,T]")
    batch, time, patches, dimension = dense.shape
    if batch < 1 or time < 1 or patches < 1 or dimension < 1:
        raise ValueError("dense token dimensions must be non-empty")
    if tuple(frame_valid.shape) != (batch, time):
        raise ValueError("frame_valid shape must match dense batch and time")
    if frame_valid.dtype != torch.bool:
        raise TypeError("frame_valid must be boolean")
    if frame_valid.device != dense.device:
        raise ValueError("dense and frame_valid must be on the same device")
    if not dense.is_floating_point():
        raise TypeError("dense tokens must have a floating-point dtype")

    valid_count = frame_valid.sum(dim=1)
    if torch.any(valid_count == 0):
        raise ValueError("every teacher window must have at least one valid frame")
    values = dense.to(dtype=torch.float32)
    if not torch.isfinite(values[frame_valid]).all():
        raise ValueError("valid dense tokens contain non-finite values")
    mask = frame_valid[:, :, None, None]
    selected = torch.where(mask, values, torch.zeros_like(values))
    denominator = (valid_count.to(dtype=torch.float32) * patches).unsqueeze(1)
    return selected.sum(dim=(1, 2)) / denominator


def normalize_teacher_vectors(vectors: Tensor, center: Tensor | None = None) -> Tensor:
    """Subtract an optional shared center and L2-normalize every row."""

    if not isinstance(vectors, Tensor):
        vectors = torch.as_tensor(vectors)
    if vectors.ndim != 2 or vectors.shape[0] < 1 or vectors.shape[1] < 1:
        raise ValueError("teacher vectors must have non-empty shape [N,D]")
    centered = vectors.to(dtype=torch.float32)
    if not torch.isfinite(centered).all():
        raise ValueError("teacher vectors contain non-finite values")
    if center is not None:
        if not isinstance(center, Tensor):
            center = torch.as_tensor(center)
        center = center.to(device=centered.device, dtype=torch.float32)
        if tuple(center.shape) != (centered.shape[1],):
            raise ValueError("teacher center must have shape [D]")
        if not torch.isfinite(center).all():
            raise ValueError("teacher center contains non-finite values")
        centered = centered - center
    norms = torch.linalg.vector_norm(centered, dim=-1, keepdim=True)
    if not torch.isfinite(norms).all() or torch.any(norms <= 1e-12):
        raise ValueError("teacher vectors contain degenerate centered targets")
    return centered / norms
