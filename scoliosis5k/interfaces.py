"""Neutral, shape-checked extension points for the T0 model.

The extension points are deliberately small.  A token transform sees one
branch's dense tokens and that branch's own frame mask.  A fusion transform
sees the concatenated, projected branch summaries and a two-column mask that
records whether each branch has any valid frame.  Implementations may return
only the transformed tensor or ``(tensor, mask)``; in either form the mask is
required to remain bitwise unchanged.  This prevents a future slot from
silently changing data validity or introducing a cross-modal alignment rule.
"""

from __future__ import annotations

from typing import Any, Tuple

import torch
from torch import nn


class TokenTransform(nn.Module):
    """Base class for a branch-local dense-token transform.

    ``tokens`` has shape ``[batch, time, token, dim]`` and ``frame_valid``
    has shape ``[batch, time]``.  The default implementation is exactly
    neutral and returns the original objects.
    """

    def forward(
        self, tokens: torch.Tensor, frame_valid: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return tokens, frame_valid


class FusionTransform(nn.Module):
    """Base class for a sequence-level fusion transform.

    ``fused`` has shape ``[batch, 2 * common_dim]`` and ``branch_valid`` has
    shape ``[batch, 2]``.  The two mask columns correspond to silhouette and
    RGB respectively.  The default implementation is exactly neutral.
    """

    def forward(
        self, fused: torch.Tensor, branch_valid: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return fused, branch_valid


class IdentityTokenTransform(TokenTransform):
    """Explicit neutral token slot useful for control tests."""

    pass


class IdentityFusionTransform(FusionTransform):
    """Explicit neutral fusion slot useful for control tests."""

    pass


def _validate_mask(
    mask: torch.Tensor,
    expected_shape: Tuple[int, ...],
    device: torch.device,
    name: str,
) -> None:
    if not isinstance(mask, torch.Tensor):
        raise TypeError("{} must be a torch.Tensor".format(name))
    if tuple(mask.shape) != expected_shape:
        raise ValueError(
            "{} must have shape {}, got {}".format(
                name, expected_shape, tuple(mask.shape)
            )
        )
    if mask.dtype is not torch.bool:
        raise TypeError("{} must have dtype torch.bool".format(name))
    if mask.device != device:
        raise ValueError("{} must be on the token device".format(name))


def _unpack_transform_output(output: Any, name: str):
    if isinstance(output, tuple):
        if len(output) != 2:
            raise TypeError(
                "{} must return a tensor or a (tensor, mask) pair".format(name)
            )
        return output[0], output[1]
    return output, None


def _validate_tensor_output(
    output: Any,
    expected_shape: Tuple[int, ...],
    reference: torch.Tensor,
    name: str,
) -> torch.Tensor:
    if not isinstance(output, torch.Tensor):
        raise TypeError("{} must return a torch.Tensor".format(name))
    if tuple(output.shape) != expected_shape:
        raise ValueError(
            "{} changed token shape from {} to {}".format(
                name, expected_shape, tuple(output.shape)
            )
        )
    if output.dtype != reference.dtype:
        raise TypeError("{} changed tensor dtype".format(name))
    if output.device != reference.device:
        raise ValueError("{} changed tensor device".format(name))
    return output


def apply_token_transform(
    transform: Any,
    tokens: torch.Tensor,
    frame_valid: torch.Tensor,
    enabled: bool,
    name: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply one branch-local slot while enforcing its neutral contract."""

    if type(enabled) is not bool:
        raise TypeError("{} enabled flag must be bool".format(name))
    if tokens.ndim != 4:
        raise ValueError("{} input tokens must have shape [B,T,N,D]".format(name))
    batch, time = tokens.shape[:2]
    _validate_mask(frame_valid, (batch, time), tokens.device, name + " mask")
    if not enabled:
        return tokens, frame_valid
    if transform is None or not callable(transform):
        raise ValueError("{} is enabled but no callable was supplied".format(name))

    original_mask = frame_valid.clone()
    slot_mask = original_mask.clone()
    output = transform(tokens, slot_mask)
    if not torch.equal(slot_mask, original_mask):
        raise ValueError("{} is not allowed to change the frame mask".format(name))
    transformed, output_mask = _unpack_transform_output(output, name)
    transformed = _validate_tensor_output(
        transformed, tuple(tokens.shape), tokens, name
    )
    if output_mask is not None:
        _validate_mask(
            output_mask, (batch, time), tokens.device, name + " output mask"
        )
        if not torch.equal(output_mask, original_mask):
            raise ValueError("{} is not allowed to change the frame mask".format(name))
    return transformed, original_mask


def apply_fusion_transform(
    transform: Any,
    fused: torch.Tensor,
    branch_valid: torch.Tensor,
    enabled: bool,
    name: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply the sequence-level slot while enforcing its neutral contract."""

    if type(enabled) is not bool:
        raise TypeError("{} enabled flag must be bool".format(name))
    if fused.ndim != 2:
        raise ValueError("{} input must have shape [B,2D]".format(name))
    batch = fused.shape[0]
    _validate_mask(branch_valid, (batch, 2), fused.device, name + " mask")
    if not enabled:
        return fused, branch_valid
    if transform is None or not callable(transform):
        raise ValueError("{} is enabled but no callable was supplied".format(name))

    original_mask = branch_valid.clone()
    slot_mask = original_mask.clone()
    output = transform(fused, slot_mask)
    if not torch.equal(slot_mask, original_mask):
        raise ValueError("{} is not allowed to change the branch mask".format(name))
    transformed, output_mask = _unpack_transform_output(output, name)
    transformed = _validate_tensor_output(
        transformed, tuple(fused.shape), fused, name
    )
    if output_mask is not None:
        _validate_mask(
            output_mask, (batch, 2), fused.device, name + " output mask"
        )
        if not torch.equal(output_mask, original_mask):
            raise ValueError("{} is not allowed to change the branch mask".format(name))
    return transformed, original_mask


__all__ = [
    "FusionTransform",
    "IdentityFusionTransform",
    "IdentityTokenTransform",
    "TokenTransform",
    "apply_fusion_transform",
    "apply_token_transform",
]
