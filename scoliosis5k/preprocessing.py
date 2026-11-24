"""Pixel preparation for paired 24-frame windows."""

from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F


def prepare_window(
    rgb: Sequence[np.ndarray | None],
    silhouettes: Sequence[np.ndarray | None],
) -> tuple[dict[str, Tensor], np.ndarray]:
    """Prepare one indexed window and select its native regional RGB frame.

    RGB frames are uint8 HWC and silhouettes are uint8 64×64 arrays.
    Binary masks use 0 and 255. Mark invalid slots with None. Slot indices must
    reflect the original paired recording order.
    """
    if len(rgb) != 24 or len(silhouettes) != 24:
        raise ValueError("both streams must contain exactly 24 indexed slots")

    images = torch.zeros(1, 24, 3, 224, 149, dtype=torch.float32)
    masks = torch.zeros(1, 24, 64, 64, dtype=torch.float32)
    rgb_valid = torch.zeros(1, 24, dtype=torch.bool)
    sil_valid = torch.zeros_like(rgb_valid)
    for slot, frame in enumerate(rgb):
        if frame is None:
            continue
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or 0 in frame.shape:
            raise ValueError("RGB frames must be non-empty uint8 HWC arrays")
        native = torch.from_numpy(np.array(frame, copy=True)).permute(2, 0, 1).float().div_(255.0)
        low = F.interpolate(
            native[None], size=(96, 64), mode="bilinear", align_corners=False
        )
        low_normalized = low.clamp(0.0, 1.0) * 2.0 - 1.0
        images[0, slot] = F.interpolate(
            (low_normalized + 1.0) / 2.0,
            size=(224, 149),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )[0]
        rgb_valid[0, slot] = True
    for slot, silhouette in enumerate(silhouettes):
        if silhouette is None:
            continue
        if silhouette.shape != (64, 64) or silhouette.dtype != np.uint8:
            raise ValueError("silhouettes must be uint8 64x64 arrays")
        masks[0, slot] = torch.from_numpy(np.array(silhouette, copy=True)).float().div_(255.0)
        sil_valid[0, slot] = True
    if not rgb_valid.any() or not sil_valid.any():
        raise ValueError("each stream must contain at least one valid frame")

    _, regional_frame = min(
        ((i, frame) for i, frame in enumerate(rgb) if frame is not None),
        key=lambda item: (abs(item[0] - 11), item[0]),
    )
    batch = {
        "aux_images": images,
        "aux_valid": rgb_valid,
        "silhouettes": masks,
        "frame_valid": sil_valid,
    }
    return batch, regional_frame
