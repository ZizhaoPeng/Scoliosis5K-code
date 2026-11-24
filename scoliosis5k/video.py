"""T0 baseline and T1 multimodal Transformer classifiers.

The caller supplies an RGB encoder that returns dense tokens and a validity
mask. The RGB and silhouette branches each keep their own time axis until
mean or attentive pooling.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple, Union

import torch
from torch import nn

from .interfaces import apply_fusion_transform, apply_token_transform


CLASSES = 3
DEFAULT_COMMON_DIM = 128
DEFAULT_PATCH_SIZE = 8
DEFAULT_SILHOUETTE_HEADS = 4
DEFAULT_SILHOUETTE_LAYERS = 1


def _checked_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("{} must be a positive integer".format(name))
    return value


def _checked_patch_size(value: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    if isinstance(value, bool):
        raise ValueError("patch_size must be a positive integer or pair")
    if isinstance(value, int):
        value = (value, value)
    elif isinstance(value, tuple) and len(value) == 2:
        value = (value[0], value[1])
    else:
        raise ValueError("patch_size must be a positive integer or pair")
    return (
        _checked_positive_int(value[0], "patch height"),
        _checked_positive_int(value[1], "patch width"),
    )


def _validate_float_tensor(
    value: Any, name: str, ndim: int, device: Optional[torch.device] = None
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError("{} must be a torch.Tensor".format(name))
    if value.ndim != ndim:
        raise ValueError(
            "{} must have {} dimensions, got {}".format(name, ndim, value.ndim)
        )
    if value.dtype is not torch.float32:
        raise TypeError("{} must have dtype torch.float32".format(name))
    if device is not None and value.device != device:
        raise ValueError("{} must be on the silhouette device".format(name))
    return value


def _validate_mask(
    value: Any,
    name: str,
    shape: Tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError("{} must be a torch.Tensor".format(name))
    if tuple(value.shape) != shape:
        raise ValueError(
            "{} must have shape {}, got {}".format(name, shape, tuple(value.shape))
        )
    if value.dtype is not torch.bool:
        raise TypeError("{} must have dtype torch.bool".format(name))
    if value.device != device:
        raise ValueError("{} must be on the silhouette device".format(name))
    if not bool(value.any(dim=1).all().item()):
        raise ValueError("each sample must have at least one valid {} frame".format(name))
    return value


def _unwrap_dense_output(output: Any) -> torch.Tensor:
    """Extract a tensor from common encoder result containers.

    The adapter accepts a tensor directly or a mapping with exactly one
    explicit dense-token key.  It does not guess from arbitrary tensors in a
    container, because predictor or pooled outputs are not dense tokens.
    """

    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, Mapping):
        keys = [key for key in ("tokens", "dense_tokens") if key in output]
        tensor_values = [candidate for candidate in output.values() if isinstance(candidate, torch.Tensor)]
        if len(keys) != 1 or len(tensor_values) != 1:
            raise TypeError(
                "RGB encoder mapping must contain exactly one dense token tensor"
            )
        candidate = output[keys[0]]
        if not isinstance(candidate, torch.Tensor):
            raise TypeError("RGB encoder dense token key must contain a tensor")
        return candidate
    raise TypeError(
        "RGB encoder must return a tensor or an explicit tokens/dense_tokens mapping"
    )


def _as_per_frame_dense_tokens(
    output: torch.Tensor, batch: int, time: int
) -> torch.Tensor:
    """Normalize an encoder result to ``[B,T,N,D]`` without pairing branches."""

    if output.ndim == 5:
        # Dense spatial tokens written as [B,T,H_tokens,W_tokens,D].
        if output.shape[0] != batch or output.shape[1] != time:
            raise ValueError(
                "RGB dense tokens must preserve input [B,T] dimensions"
            )
        if output.shape[2] < 1 or output.shape[3] < 1 or output.shape[4] < 1:
            raise ValueError("RGB dense token grid must be non-empty")
        output = output.flatten(2, 3)
    elif output.ndim == 4:
        if output.shape[0] != batch or output.shape[1] != time:
            raise ValueError(
                "RGB dense tokens must have shape [B,T,N,D]"
            )
    elif output.ndim == 3:
        if output.shape[0] == batch * time:
            if output.shape[1] < 1 or output.shape[2] < 1:
                raise ValueError("RGB dense token sequence must be non-empty")
            output = output.reshape(batch, time, output.shape[1], output.shape[2])
        elif output.shape[0] == batch and time == 1:
            output = output.unsqueeze(1)
        else:
            raise ValueError(
                "RGB encoder output must be per-frame [B,T,N,D] or [B*T,N,D]"
            )
    elif output.ndim == 2:
        raise ValueError("RGB encoder pooled 2D output is not dense per-frame tokens")
    else:
        raise ValueError("RGB encoder output must contain dense per-frame tokens")

    if output.shape[2] < 1 or output.shape[3] < 1:
        raise ValueError("RGB dense token sequence must be non-empty")
    return output


def masked_mean_pool(tokens: torch.Tensor, frame_valid: torch.Tensor) -> torch.Tensor:
    """Mean-pool dense tokens using only the branch-local frame mask."""

    if tokens.ndim != 4:
        raise ValueError("tokens must have shape [B,T,N,D]")
    batch, time, token_count, _ = tokens.shape
    if batch < 1 or time < 1 or token_count < 1:
        raise ValueError("tokens must have non-empty batch/time/token dimensions")
    if frame_valid.shape != (batch, time) or frame_valid.dtype is not torch.bool:
        raise ValueError("frame_valid must be a bool tensor with shape [B,T]")
    if frame_valid.device != tokens.device:
        raise ValueError("frame_valid must be on the token device")
    if not bool(frame_valid.any(dim=1).all().item()):
        raise ValueError("each sample must have at least one valid frame")
    expanded = frame_valid[:, :, None].expand(batch, time, token_count)
    flat_mask = expanded.reshape(batch, time * token_count)
    flat_tokens = tokens.reshape(batch, time * token_count, tokens.shape[-1])
    # Remove padding before checking finiteness.  This deliberately handles
    # encoders that use NaN for padded RGB rows without allowing a NaN to enter
    # the pooled value through a zero weight.
    flat_tokens = torch.where(
        flat_mask[:, :, None], flat_tokens, torch.zeros_like(flat_tokens)
    )
    if not bool(torch.isfinite(flat_tokens).all().item()):
        raise ValueError("valid tokens must be finite")
    weights = flat_mask[:, :, None].to(dtype=flat_tokens.dtype)
    counts = flat_mask.sum(dim=1).to(dtype=flat_tokens.dtype).clamp_min(1.0)
    return (flat_tokens * weights).sum(dim=1) / counts[:, None]


def _sinusoidal_positions(
    length: int, dimension: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Create deterministic positions without a fixed maximum clip length."""

    positions = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    scale = torch.arange(0, dimension, 2, device=device, dtype=dtype)
    scale = torch.pow(10000.0, -scale / float(max(dimension, 1)))
    values = positions * scale.unsqueeze(0)
    encoding = torch.zeros(length, dimension, device=device, dtype=dtype)
    encoding[:, 0::2] = torch.sin(values)
    if dimension > 1:
        encoding[:, 1::2] = torch.cos(values[:, : encoding[:, 1::2].shape[1]])
    return encoding


def _separate_axis_positions(
    rows: int,
    columns: int,
    time: int,
    dimension: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if dimension < 3:
        raise ValueError("separate_axes position encoding needs model_dim >= 3")
    axis_dims = [dimension // 3] * 3
    for index in range(dimension % 3):
        axis_dims[index] += 1
    row_dim, column_dim, time_dim = axis_dims

    def place(length: int, axis_dim: int, start: int) -> torch.Tensor:
        result = torch.zeros(length, dimension, device=device, dtype=dtype)
        result[:, start : start + axis_dim] = _sinusoidal_positions(
            length, axis_dim, device, dtype
        )
        return result

    row = place(rows, row_dim, 0)[:, None, :]
    column = place(columns, column_dim, row_dim)[None, :, :]
    spatial = (row + column).reshape(rows * columns, dimension)
    temporal = place(time, time_dim, row_dim + column_dim)
    return spatial, temporal


def _checked_dropout(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError("{} must be a number in [0,1]".format(name))
    value = float(value)
    if value < 0.0 or value > 1.0:
        raise ValueError("{} must be a number in [0,1]".format(name))
    return value


class _PreNormAttentionBlock(nn.Module):
    """One pre-LN self-attention plus feed-forward residual block."""

    def __init__(
        self, model_dim: int, heads: int, feedforward_dim: int, dropout: float
    ) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(model_dim)
        self.attention = nn.MultiheadAttention(
            model_dim,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_ff = nn.LayerNorm(model_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(model_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, model_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        normalized = self.norm_attn(tokens)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        tokens = tokens + attended
        return tokens + self.feedforward(self.norm_ff(tokens))


class _DividedLayer(nn.Module):
    """Independent spatial and temporal pre-LN blocks for one layer."""

    def __init__(
        self, model_dim: int, heads: int, feedforward_dim: int, dropout: float
    ) -> None:
        super().__init__()
        self.spatial = _PreNormAttentionBlock(
            model_dim, heads, feedforward_dim, dropout
        )
        self.temporal = _PreNormAttentionBlock(
            model_dim, heads, feedforward_dim, dropout
        )


class _DividedTransformer(nn.Module):
    """TimeSformer-style factorized attention over valid silhouette frames."""

    def __init__(
        self,
        model_dim: int,
        heads: int,
        layers: int,
        feedforward_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _DividedLayer(model_dim, heads, feedforward_dim, dropout)
                for _ in range(layers)
            ]
        )

    def forward(
        self, tokens: torch.Tensor, frame_valid: torch.Tensor
    ) -> torch.Tensor:
        # Removing invalid rows before each attention call avoids the
        # all-key-padding NaN case.  Every sample has at least one valid time
        # by the input contract, so the temporal calls are also non-empty.
        batch, time, token_count, _ = tokens.shape
        for layer in self.layers:
            spatial_out = tokens.clone()
            spatial_inputs = []
            spatial_lengths = []
            for sample in range(batch):
                valid_time = torch.nonzero(frame_valid[sample], as_tuple=False).flatten()
                spatial_inputs.append(tokens[sample, valid_time])
                spatial_lengths.append((sample, valid_time.numel(), valid_time))
            spatial = layer.spatial(torch.cat(spatial_inputs, dim=0))
            offset = 0
            for sample, length, valid_time in spatial_lengths:
                spatial_out[sample, valid_time] = spatial[offset : offset + length]
                offset += length

            # All samples have at least one valid frame.  A key-padding mask
            # therefore has no all-masked row while processing every patch's
            # temporal sequence in one batched attention call.
            temporal = spatial_out.permute(0, 2, 1, 3).reshape(
                batch * token_count, time, tokens.shape[-1]
            )
            temporal_padding = ~frame_valid[:, None, :].expand(
                batch, token_count, time
            ).reshape(batch * token_count, time)
            temporal = layer.temporal(
                temporal, key_padding_mask=temporal_padding
            )
            tokens = temporal.reshape(batch, token_count, time, tokens.shape[-1]).permute(
                0, 2, 1, 3
            )
        return tokens


class RGBDenseTokenAdapter(nn.Module):
    """Adapt an injected encoder's dense per-frame tokens to ``output_dim``.

    Encoders that expose ``encode_dense(images, valid)`` return their tokens
    and validity mask. Encoders that expose only ``forward(images)`` are
    called with invalid rows zeroed and inherit the input mask.
    """

    def __init__(
        self,
        encoder: nn.Module,
        output_dim: int,
        rgb_token_dim: Optional[int] = None,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(encoder, nn.Module):
            raise TypeError("encoder must be an nn.Module supplied by the caller")
        if type(freeze_encoder) is not bool:
            raise TypeError("freeze_encoder must be bool")
        self.encoder = encoder
        self.freeze_encoder = freeze_encoder
        self.output_dim = _checked_positive_int(output_dim, "output_dim")
        if rgb_token_dim is not None:
            rgb_token_dim = _checked_positive_int(rgb_token_dim, "rgb_token_dim")
            self.rgb_token_dim = rgb_token_dim
            self.projection = nn.Linear(rgb_token_dim, output_dim)
        else:
            self.rgb_token_dim = None
            self.projection = nn.LazyLinear(output_dim)
        if self.freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(False)
            self.encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def _run_encoder(self, call):
        if self.freeze_encoder:
            with torch.no_grad():
                return call()
        return call()

    def _project(
        self, dense: torch.Tensor, frame_valid: torch.Tensor
    ) -> torch.Tensor:
        if dense.dtype is not torch.float32:
            raise TypeError("RGB encoder dense tokens must have dtype torch.float32")
        if dense.device != self.projection.weight.device:
            raise ValueError("RGB encoder dense tokens changed device")
        if self.rgb_token_dim is not None and dense.shape[-1] != self.rgb_token_dim:
            raise ValueError(
                "RGB encoder token dimension {} differs from configured {}".format(
                    dense.shape[-1], self.rgb_token_dim
                )
            )
        valid_tokens = torch.where(
            frame_valid[:, :, None, None], dense, torch.zeros_like(dense)
        )
        if not bool(torch.isfinite(valid_tokens).all().item()):
            raise ValueError("valid RGB dense tokens must be finite")
        return self.projection(valid_tokens)

    def project_dense(
        self, dense: torch.Tensor, frame_valid: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project cached raw encoder tokens while keeping projection trainable."""

        dense = _validate_float_tensor(dense, "rgb_dense", 4)
        if dense.shape[0] < 1 or dense.shape[1] < 1 or dense.shape[2] < 1:
            raise ValueError("rgb_dense must have non-empty batch/time/token dimensions")
        frame_valid = _validate_mask(
            frame_valid,
            "rgb_frame_valid",
            (dense.shape[0], dense.shape[1]),
            dense.device,
        )
        return self._project(dense, frame_valid), frame_valid

    def forward(
        self, aux_images: torch.Tensor, aux_valid: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if aux_images.ndim != 5 or aux_images.shape[2] != 3:
            raise ValueError("aux_images must have shape [B,T,3,H,W]")
        batch, time = aux_images.shape[:2]
        aux_valid = _validate_mask(
            aux_valid,
            "aux_valid",
            (batch, time),
            aux_images.device,
        )
        encode_dense = getattr(self.encoder, "encode_dense", None)
        if callable(encode_dense):
            result = self._run_encoder(lambda: encode_dense(aux_images, aux_valid))
            if not isinstance(result, (tuple, list)) or len(result) != 2:
                raise TypeError(
                    "encoder.encode_dense must return (tokens, frame_valid)"
                )
            dense, frame_valid = result
            if not isinstance(dense, torch.Tensor) or dense.ndim != 4:
                raise ValueError(
                    "encoder.encode_dense tokens must have shape [B,T,N,D]"
                )
            if dense.shape[0] != batch or dense.shape[1] < 1:
                raise ValueError(
                    "encoder.encode_dense tokens must preserve batch and have non-empty time"
                )
            if dense.shape[2] < 1 or dense.shape[3] < 1:
                raise ValueError("encoder.encode_dense tokens must be non-empty")
            frame_valid = _validate_mask(
                frame_valid,
                "rgb_frame_valid",
                (batch, dense.shape[1]),
                aux_images.device,
            )
            return self._project(dense, frame_valid), frame_valid

        # Zero invalid rows before calling forward(images). Encoders using
        # encode_dense handle masking because they may change the time axis.
        valid_image_mask = aux_valid[:, :, None, None, None]
        sanitized = torch.where(
            valid_image_mask, aux_images, torch.zeros_like(aux_images)
        )
        dense = _as_per_frame_dense_tokens(
            self._run_encoder(lambda: _unwrap_dense_output(self.encoder(sanitized))),
            batch,
            time,
        )
        return self._project(dense, aux_valid), aux_valid


class SilhouettePatchTransformer(nn.Module):
    """Linear per-frame patch embedding followed by a small Transformer."""

    def __init__(
        self,
        model_dim: int,
        patch_size: Union[int, Tuple[int, int]],
        heads: int,
        layers: int,
        feedforward_dim: Optional[int] = None,
        attention: str = "joint",
        dropout: float = 0.0,
        position_encoding: str = "legacy",
    ) -> None:
        super().__init__()
        model_dim = _checked_positive_int(model_dim, "model_dim")
        heads = _checked_positive_int(heads, "silhouette_heads")
        layers = _checked_positive_int(layers, "silhouette_layers")
        if model_dim % heads != 0:
            raise ValueError("model_dim must be divisible by silhouette_heads")
        if attention not in ("joint", "divided"):
            raise ValueError("silhouette attention must be 'joint' or 'divided'")
        if position_encoding not in ("legacy", "separate_axes"):
            raise ValueError(
                "position_encoding must be 'legacy' or 'separate_axes'"
            )
        dropout = _checked_dropout(dropout, "silhouette_dropout")
        self.model_dim = model_dim
        self.patch_size = _checked_patch_size(patch_size)
        self.attention = attention
        self.dropout = dropout
        self.position_encoding = position_encoding
        patch_area = self.patch_size[0] * self.patch_size[1]
        self.patch_embedding = nn.Linear(patch_area, model_dim)
        if feedforward_dim is None:
            feedforward_dim = 4 * model_dim
        feedforward_dim = _checked_positive_int(
            feedforward_dim, "silhouette_feedforward_dim"
        )
        if attention == "joint":
            layer = nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=heads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=layers)
        else:
            self.transformer = _DividedTransformer(
                model_dim,
                heads,
                layers,
                feedforward_dim,
                dropout,
            )

    def forward(
        self, silhouettes: torch.Tensor, frame_valid: torch.Tensor
    ) -> torch.Tensor:
        if silhouettes.ndim != 4:
            raise ValueError("silhouettes must have shape [B,T,H,W]")
        batch, time, height, width = silhouettes.shape
        frame_valid = _validate_mask(
            frame_valid, "frame_valid", (batch, time), silhouettes.device
        )
        patch_height, patch_width = self.patch_size
        if height % patch_height or width % patch_width:
            raise ValueError(
                "silhouette height and width must be divisible by patch_size"
            )

        valid_mask = frame_valid[:, :, None, None]
        sanitized = torch.where(valid_mask, silhouettes, torch.zeros_like(silhouettes))
        patches = sanitized.unfold(2, patch_height, patch_height).unfold(
            3, patch_width, patch_width
        )
        grid_height, grid_width = patches.shape[2:4]
        token_count = grid_height * grid_width
        patches = patches.contiguous().reshape(
            batch, time, token_count, patch_height * patch_width
        )
        tokens = self.patch_embedding(patches)
        # Spatial and within-branch temporal order are represented locally.
        # The RGB branch has its own time axis and never receives these
        # positions, so this does not create a cross-modal timestamp claim.
        if self.position_encoding == "legacy":
            spatial_positions = _sinusoidal_positions(
                token_count, self.model_dim, tokens.device, tokens.dtype
            )
            temporal_positions = _sinusoidal_positions(
                time, self.model_dim, tokens.device, tokens.dtype
            )
        else:
            spatial_positions, temporal_positions = _separate_axis_positions(
                grid_height,
                grid_width,
                time,
                self.model_dim,
                tokens.device,
                tokens.dtype,
            )
        tokens = tokens + spatial_positions[None, None, :, :]
        tokens = tokens + temporal_positions[None, :, None, :]
        if self.attention == "divided":
            return self.transformer(tokens, frame_valid)
        tokens = tokens.reshape(batch, time * token_count, self.model_dim)
        padding = ~frame_valid[:, :, None].expand(batch, time, token_count)
        tokens = self.transformer(
            tokens, src_key_padding_mask=padding.reshape(batch, time * token_count)
        )
        return tokens.reshape(batch, time, token_count, self.model_dim)


class MultimodalTransformerT0(nn.Module):
    """T0 pooled RGB/silhouette three-class baseline.

    The constructor accepts optional interface modules, but each slot is
    disabled by default.  The three flags can be switched independently.  A
    slot cannot change its branch shape or validity mask, which keeps future
    ablations separate from the baseline's data contract.
    """

    def __init__(
        self,
        rgb_encoder: nn.Module,
        rgb_token_dim: Optional[int] = None,
        common_dim: int = DEFAULT_COMMON_DIM,
        patch_size: Union[int, Tuple[int, int]] = DEFAULT_PATCH_SIZE,
        silhouette_heads: int = DEFAULT_SILHOUETTE_HEADS,
        silhouette_layers: int = DEFAULT_SILHOUETTE_LAYERS,
        silhouette_feedforward_dim: Optional[int] = None,
        rgb_token_transform: Optional[nn.Module] = None,
        silhouette_token_transform: Optional[nn.Module] = None,
        fusion_transform: Optional[nn.Module] = None,
        enable_rgb_token_transform: bool = False,
        enable_silhouette_token_transform: bool = False,
        enable_fusion_transform: bool = False,
        silhouette_attention: str = "joint",
        silhouette_dropout: float = 0.0,
        freeze_rgb_encoder: bool = False,
        silhouette_position_encoding: str = "legacy",
    ) -> None:
        super().__init__()
        common_dim = _checked_positive_int(common_dim, "common_dim")
        for name, enabled in (
            ("enable_rgb_token_transform", enable_rgb_token_transform),
            ("enable_silhouette_token_transform", enable_silhouette_token_transform),
            ("enable_fusion_transform", enable_fusion_transform),
        ):
            if type(enabled) is not bool:
                raise TypeError("{} must be bool".format(name))
        for name, transform in (
            ("rgb_token_transform", rgb_token_transform),
            ("silhouette_token_transform", silhouette_token_transform),
            ("fusion_transform", fusion_transform),
        ):
            if transform is not None and not isinstance(transform, nn.Module):
                raise TypeError("{} must be an nn.Module when supplied".format(name))
        if enable_rgb_token_transform and rgb_token_transform is None:
            raise ValueError("RGB token transform is enabled but not supplied")
        if enable_silhouette_token_transform and silhouette_token_transform is None:
            raise ValueError("silhouette token transform is enabled but not supplied")
        if enable_fusion_transform and fusion_transform is None:
            raise ValueError("fusion transform is enabled but not supplied")
        if type(freeze_rgb_encoder) is not bool:
            raise TypeError("freeze_rgb_encoder must be bool")

        self.common_dim = common_dim
        self.rgb_adapter = RGBDenseTokenAdapter(
            rgb_encoder,
            output_dim=common_dim,
            rgb_token_dim=rgb_token_dim,
            freeze_encoder=freeze_rgb_encoder,
        )
        self.silhouette_branch = SilhouettePatchTransformer(
            model_dim=common_dim,
            patch_size=patch_size,
            heads=silhouette_heads,
            layers=silhouette_layers,
            feedforward_dim=silhouette_feedforward_dim,
            attention=silhouette_attention,
            dropout=silhouette_dropout,
            position_encoding=silhouette_position_encoding,
        )
        # Each branch is projected independently to the common dimension.
        # Both inputs already have common_dim, but explicit projections keep
        # the T0 fusion contract visible and independently replaceable.
        self.silhouette_projection = nn.Linear(common_dim, common_dim)
        self.rgb_projection = nn.Linear(common_dim, common_dim)
        self.fusion_norm = nn.LayerNorm(2 * common_dim)
        self.classifier = nn.Linear(2 * common_dim, CLASSES)

        self.rgb_token_transform = rgb_token_transform
        self.silhouette_token_transform = silhouette_token_transform
        self.fusion_transform = fusion_transform
        self.enable_rgb_token_transform = enable_rgb_token_transform
        self.enable_silhouette_token_transform = enable_silhouette_token_transform
        self.enable_fusion_transform = enable_fusion_transform

    @staticmethod
    def _inputs_from_arguments(
        batch_or_silhouettes: Any,
        frame_valid: Optional[torch.Tensor],
        aux_images: Optional[torch.Tensor],
        aux_valid: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(batch_or_silhouettes, Mapping):
            if any(value is not None for value in (frame_valid, aux_images, aux_valid)):
                raise TypeError("mapping input cannot be combined with positional tensors")
            batch = batch_or_silhouettes
            required = ("silhouettes", "frame_valid", "aux_images", "aux_valid")
            missing = [key for key in required if key not in batch]
            if missing:
                raise KeyError("batch is missing required keys: {}".format(missing))
            return (
                batch["silhouettes"],
                batch["frame_valid"],
                batch["aux_images"],
                batch["aux_valid"],
            )
        if frame_valid is None or aux_images is None or aux_valid is None:
            raise TypeError(
                "positional forward requires silhouettes, frame_valid, aux_images, aux_valid"
            )
        return batch_or_silhouettes, frame_valid, aux_images, aux_valid

    @staticmethod
    def _validate_inputs(
        silhouettes: Any,
        frame_valid: Any,
        aux_images: Any,
        aux_valid: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        silhouettes = _validate_float_tensor(silhouettes, "silhouettes", 4)
        batch, silhouette_time = silhouettes.shape[:2]
        if (
            batch < 1
            or silhouette_time < 1
            or silhouettes.shape[2] < 1
            or silhouettes.shape[3] < 1
        ):
            raise ValueError("silhouettes must have non-empty batch/time/spatial dimensions")
        frame_valid = _validate_mask(
            frame_valid,
            "frame_valid",
            (batch, silhouette_time),
            silhouettes.device,
        )
        aux_images = _validate_float_tensor(
            aux_images, "aux_images", 5, silhouettes.device
        )
        if aux_images.shape[0] != batch or aux_images.shape[1] < 1 or aux_images.shape[2] != 3:
            raise ValueError("aux_images must have shape [B,T_rgb,3,H,W]")
        if aux_images.shape[3] < 1 or aux_images.shape[4] < 1:
            raise ValueError("aux_images must have non-empty spatial dimensions")
        aux_valid = _validate_mask(
            aux_valid,
            "aux_valid",
            (batch, aux_images.shape[1]),
            silhouettes.device,
        )
        return silhouettes, frame_valid, aux_images, aux_valid

    def _encode_branches(
        self,
        silhouettes: torch.Tensor,
        frame_valid: torch.Tensor,
        aux_images: torch.Tensor,
        aux_valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rgb_tokens, rgb_frame_valid = self.rgb_adapter(aux_images, aux_valid)
        return self._encode_branches_from_projected_rgb(
            silhouettes, frame_valid, rgb_tokens, rgb_frame_valid
        )

    def _encode_branches_from_dense(
        self,
        silhouettes: torch.Tensor,
        frame_valid: torch.Tensor,
        rgb_dense: torch.Tensor,
        rgb_frame_valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rgb_tokens, rgb_frame_valid = self.rgb_adapter.project_dense(
            rgb_dense, rgb_frame_valid
        )
        return self._encode_branches_from_projected_rgb(
            silhouettes, frame_valid, rgb_tokens, rgb_frame_valid
        )

    def _encode_branches_from_projected_rgb(
        self,
        silhouettes: torch.Tensor,
        frame_valid: torch.Tensor,
        rgb_tokens: torch.Tensor,
        rgb_frame_valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        silhouette_tokens = self.silhouette_branch(silhouettes, frame_valid)

        # These slots are branch-local.  They cannot observe the other branch
        # and cannot alter their own masks.
        silhouette_tokens, frame_valid = apply_token_transform(
            self.silhouette_token_transform,
            silhouette_tokens,
            frame_valid,
            self.enable_silhouette_token_transform,
            "silhouette_token_transform",
        )
        rgb_tokens, rgb_frame_valid = apply_token_transform(
            self.rgb_token_transform,
            rgb_tokens,
            rgb_frame_valid,
            self.enable_rgb_token_transform,
            "rgb_token_transform",
        )
        return silhouette_tokens, frame_valid, rgb_tokens, rgb_frame_valid

    def _fuse_pooled(
        self,
        silhouette_pooled: torch.Tensor,
        rgb_pooled: torch.Tensor,
        frame_valid: torch.Tensor,
        rgb_frame_valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        silhouette_projected = self.silhouette_projection(silhouette_pooled)
        rgb_projected = self.rgb_projection(rgb_pooled)
        fused = torch.cat((silhouette_projected, rgb_projected), dim=-1)
        branch_valid = torch.stack(
            (frame_valid.any(dim=1), rgb_frame_valid.any(dim=1)), dim=1
        )
        fused, branch_valid = apply_fusion_transform(
            self.fusion_transform,
            fused,
            branch_valid,
            self.enable_fusion_transform,
            "fusion_transform",
        )
        return self.fusion_norm(fused), branch_valid

    def forward(
        self,
        batch_or_silhouettes: Any,
        frame_valid: Optional[torch.Tensor] = None,
        aux_images: Optional[torch.Tensor] = None,
        aux_valid: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        silhouettes, frame_valid, aux_images, aux_valid = self._inputs_from_arguments(
            batch_or_silhouettes, frame_valid, aux_images, aux_valid
        )
        silhouettes, frame_valid, aux_images, aux_valid = self._validate_inputs(
            silhouettes, frame_valid, aux_images, aux_valid
        )
        silhouette_tokens, frame_valid, rgb_tokens, rgb_frame_valid = (
            self._encode_branches(silhouettes, frame_valid, aux_images, aux_valid)
        )

        # Sequence-level pooling is the only point at which branch values meet;
        # silhouette T and RGB T_rgb may differ and are never paired.
        silhouette_pooled = masked_mean_pool(silhouette_tokens, frame_valid)
        rgb_pooled = masked_mean_pool(rgb_tokens, rgb_frame_valid)
        normalized_fused, branch_valid = self._fuse_pooled(
            silhouette_pooled, rgb_pooled, frame_valid, rgb_frame_valid
        )
        logits = self.classifier(normalized_fused)
        return {
            "logits": logits,
            "silhouette_tokens": silhouette_tokens,
            "rgb_tokens": rgb_tokens,
            "silhouette_pooled": silhouette_pooled,
            "rgb_pooled": rgb_pooled,
            "fused": normalized_fused,
            "branch_valid": branch_valid,
            "rgb_frame_valid": rgb_frame_valid,
        }


class _AttentiveMaskedPool(nn.Module):
    """One learned-query cross-attention pooling layer without score output."""

    def __init__(self, model_dim: int, heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.empty(1, 1, model_dim))
        nn.init.normal_(self.query, mean=0.0, std=0.02)
        self.cross_attention = nn.MultiheadAttention(
            model_dim, heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(model_dim)

    def forward(
        self, tokens: torch.Tensor, frame_valid: torch.Tensor
    ) -> torch.Tensor:
        if tokens.ndim != 4:
            raise ValueError("tokens must have shape [B,T,N,D]")
        batch, time, token_count, dimension = tokens.shape
        if frame_valid.shape != (batch, time) or frame_valid.dtype is not torch.bool:
            raise ValueError("frame_valid must be a bool tensor with shape [B,T]")
        if frame_valid.device != tokens.device:
            raise ValueError("frame_valid must be on the token device")
        if not bool(frame_valid.any(dim=1).all().item()):
            raise ValueError("each sample must have at least one valid frame")

        flat_mask = frame_valid[:, :, None].expand(batch, time, token_count)
        flat_mask = flat_mask.reshape(batch, time * token_count)
        flat_tokens = tokens.reshape(batch, time * token_count, dimension)
        flat_tokens = torch.where(
            flat_mask[:, :, None], flat_tokens, torch.zeros_like(flat_tokens)
        )
        if not bool(torch.isfinite(flat_tokens).all().item()):
            raise ValueError("valid tokens must be finite")
        query = self.query.to(dtype=flat_tokens.dtype).expand(batch, -1, -1)
        pooled, _ = self.cross_attention(
            query,
            flat_tokens,
            flat_tokens,
            key_padding_mask=~flat_mask,
            need_weights=False,
        )
        return self.norm(pooled[:, 0])


class MultimodalTransformerT1(MultimodalTransformerT0):
    """Factorized silhouette encoder with masked attentive pooling."""

    def __init__(
        self,
        rgb_encoder: nn.Module,
        rgb_token_dim: Optional[int] = None,
        common_dim: int = DEFAULT_COMMON_DIM,
        patch_size: Union[int, Tuple[int, int]] = DEFAULT_PATCH_SIZE,
        silhouette_heads: int = DEFAULT_SILHOUETTE_HEADS,
        silhouette_layers: int = DEFAULT_SILHOUETTE_LAYERS,
        silhouette_feedforward_dim: Optional[int] = None,
        pooling: str = "attentive",
        attention: str = "divided",
        dropout: float = 0.1,
        position_encoding: str = "separate_axes",
        rgb_token_transform: Optional[nn.Module] = None,
        silhouette_token_transform: Optional[nn.Module] = None,
        fusion_transform: Optional[nn.Module] = None,
        enable_rgb_token_transform: bool = False,
        enable_silhouette_token_transform: bool = False,
        enable_fusion_transform: bool = False,
        freeze_rgb_encoder: bool = True,
    ) -> None:
        if pooling not in ("mean", "attentive"):
            raise ValueError("pooling must be 'mean' or 'attentive'")
        dropout = _checked_dropout(dropout, "dropout")
        super().__init__(
            rgb_encoder,
            rgb_token_dim=rgb_token_dim,
            common_dim=common_dim,
            patch_size=patch_size,
            silhouette_heads=silhouette_heads,
            silhouette_layers=silhouette_layers,
            silhouette_feedforward_dim=silhouette_feedforward_dim,
            rgb_token_transform=rgb_token_transform,
            silhouette_token_transform=silhouette_token_transform,
            fusion_transform=fusion_transform,
            enable_rgb_token_transform=enable_rgb_token_transform,
            enable_silhouette_token_transform=enable_silhouette_token_transform,
            enable_fusion_transform=enable_fusion_transform,
            silhouette_attention=attention,
            silhouette_dropout=dropout,
            freeze_rgb_encoder=freeze_rgb_encoder,
            silhouette_position_encoding=position_encoding,
        )
        self.pooling = pooling
        self.attention = attention
        self.dropout = dropout
        if pooling == "attentive":
            self.silhouette_attentive_pool = _AttentiveMaskedPool(
                common_dim, silhouette_heads, dropout
            )
            self.rgb_attentive_pool = _AttentiveMaskedPool(
                common_dim, silhouette_heads, dropout
            )
        else:
            self.silhouette_attentive_pool = None
            self.rgb_attentive_pool = None

    def _forward_encoded(
        self,
        silhouette_tokens: torch.Tensor,
        frame_valid: torch.Tensor,
        rgb_tokens: torch.Tensor,
        rgb_frame_valid: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if self.pooling == "mean":
            silhouette_pooled = masked_mean_pool(silhouette_tokens, frame_valid)
            rgb_pooled = masked_mean_pool(rgb_tokens, rgb_frame_valid)
        else:
            silhouette_pooled = self.silhouette_attentive_pool(
                silhouette_tokens, frame_valid
            )
            rgb_pooled = self.rgb_attentive_pool(rgb_tokens, rgb_frame_valid)
        normalized_fused, branch_valid = self._fuse_pooled(
            silhouette_pooled, rgb_pooled, frame_valid, rgb_frame_valid
        )
        logits = self.classifier(normalized_fused)
        return {
            "logits": logits,
            "silhouette_tokens": silhouette_tokens,
            "rgb_tokens": rgb_tokens,
            "silhouette_pooled": silhouette_pooled,
            "rgb_pooled": rgb_pooled,
            "fused": normalized_fused,
            "branch_valid": branch_valid,
            "rgb_frame_valid": rgb_frame_valid,
        }

    def forward(
        self,
        batch_or_silhouettes: Any,
        frame_valid: Optional[torch.Tensor] = None,
        aux_images: Optional[torch.Tensor] = None,
        aux_valid: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        silhouettes, frame_valid, aux_images, aux_valid = self._inputs_from_arguments(
            batch_or_silhouettes, frame_valid, aux_images, aux_valid
        )
        silhouettes, frame_valid, aux_images, aux_valid = self._validate_inputs(
            silhouettes, frame_valid, aux_images, aux_valid
        )
        encoded = self._encode_branches(
            silhouettes, frame_valid, aux_images, aux_valid
        )
        return self._forward_encoded(*encoded)

    def forward_from_rgb_dense(
        self,
        silhouettes: torch.Tensor,
        frame_valid: torch.Tensor,
        rgb_dense: torch.Tensor,
        rgb_frame_valid: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward from frozen raw RGB tokens cached before the trainable projection."""

        silhouettes = _validate_float_tensor(silhouettes, "silhouettes", 4)
        batch, silhouette_time = silhouettes.shape[:2]
        if batch < 1 or silhouette_time < 1:
            raise ValueError("silhouettes must have non-empty batch and time")
        frame_valid = _validate_mask(
            frame_valid,
            "frame_valid",
            (batch, silhouette_time),
            silhouettes.device,
        )
        rgb_dense = _validate_float_tensor(
            rgb_dense, "rgb_dense", 4, silhouettes.device
        )
        if rgb_dense.shape[0] != batch or rgb_dense.shape[1] < 1:
            raise ValueError("rgb_dense must preserve batch and have non-empty time")
        rgb_frame_valid = _validate_mask(
            rgb_frame_valid,
            "rgb_frame_valid",
            (batch, rgb_dense.shape[1]),
            silhouettes.device,
        )
        encoded = self._encode_branches_from_dense(
            silhouettes, frame_valid, rgb_dense, rgb_frame_valid
        )
        return self._forward_encoded(*encoded)


T0MultimodalTransformer = MultimodalTransformerT0


__all__ = [
    "CLASSES",
    "DEFAULT_COMMON_DIM",
    "DEFAULT_PATCH_SIZE",
    "DEFAULT_SILHOUETTE_HEADS",
    "DEFAULT_SILHOUETTE_LAYERS",
    "MultimodalTransformerT0",
    "MultimodalTransformerT1",
    "RGBDenseTokenAdapter",
    "SilhouettePatchTransformer",
    "T0MultimodalTransformer",
    "masked_mean_pool",
]
