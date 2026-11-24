"""Load PE-Spatial weights and extract normalized regional RGB descriptors."""

from pathlib import Path

import cv2
import numpy as np
import torch


def reduce_patch_regions(patches: torch.Tensor) -> torch.Tensor:
    """Average the central patch columns into eight unit-normalized regions."""
    if tuple(patches.shape[1:]) != (1024, 768):
        raise ValueError("expected a 32x32 PE patch grid without the class token")
    central = patches.reshape(-1, 32, 32, 768)[:, :, 8:24, :]
    regions = central.reshape(-1, 4, 8, 2, 8, 768).mean(dim=(2, 4)).reshape(-1, 8, 768)
    return torch.nn.functional.normalize(regions, p=2, dim=-1, eps=1e-12)


def load_pe_spatial(checkpoint: Path) -> torch.nn.Module:
    """Load an explicitly supplied PE-Spatial checkpoint with strict matching.

    The upstream ``core.vision_encoder.pe`` module must already be importable.
    This function does not download model code or weights.
    """
    from core.vision_encoder.pe import VisionTransformer

    checkpoint = Path(checkpoint)
    encoder = VisionTransformer.from_config("PE-Spatial-B16-512", pretrained=False)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    elif "weights" in state:
        state = state["weights"]
    state = {key.replace("module.", ""): value for key, value in state.items()}
    if any(key.startswith("visual.") for key in state):
        state = {
            key.replace("visual.", ""): value
            for key, value in state.items()
            if "visual" in key
        }
    encoder.load_state_dict(state, strict=True)
    return encoder.eval()


def extract_regions(encoder: torch.nn.Module, native_rgb: np.ndarray) -> np.ndarray:
    """Return eight normalized PE descriptors as an FP16 ``[8, 768]`` array."""
    if (
        not isinstance(native_rgb, np.ndarray)
        or native_rgb.dtype != np.uint8
        or native_rgb.ndim != 3
        or native_rgb.shape[2] != 3
        or native_rgb.shape[0] < 1
        or native_rgb.shape[1] < 1
    ):
        raise ValueError("native RGB frame must be non-empty uint8 HWC RGB")

    resized = cv2.resize(native_rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
    pixels = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32)
    pixels.div_(127.5).sub_(1.0)
    with torch.inference_mode():
        patches = encoder.forward_features(pixels, layer_idx=-1, strip_cls_token=True)
        regions = reduce_patch_regions(patches)[0].cpu().numpy()
    if not np.isfinite(regions).all():
        raise FloatingPointError("PE-Spatial returned non-finite regional descriptors")
    return regions.astype(np.float16)
