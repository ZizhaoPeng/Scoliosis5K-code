"""Run the paired head and regional classifier on synthetic inputs."""

import numpy as np
import torch

from .distillation import CrossModalTeacherT1
from .motion import compute_motion_gate
from .preprocessing import prepare_window
from .regional import fit_regional, fuse_regional, predict_regional


def main() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(73111)
    rng = np.random.default_rng(73111)
    rgb = [rng.integers(0, 256, (96, 64, 3), dtype=np.uint8) for _ in range(24)]
    silhouettes = [rng.integers(0, 2, (64, 64), dtype=np.uint8) * 255 for _ in rgb]
    batch, _ = prepare_window(rgb, silhouettes)
    paired = CrossModalTeacherT1(mode="paired").eval()

    with torch.inference_mode():
        gate = compute_motion_gate(batch["aux_images"], batch["aux_valid"])
        synthetic_tokens = torch.randn(1, 24, 256, 384)
        video_logits = paired(
            batch["silhouettes"], batch["frame_valid"], synthetic_tokens, batch["aux_valid"]
        )["logits"].numpy()[:, None, :]

    descriptors = rng.standard_normal((15, 2, 8, 768)).astype(np.float32)
    valid = np.ones((15, 2), dtype=np.bool_)
    labels = np.repeat(np.arange(3), 5)
    people = np.array([f"synthetic-{i:02d}" for i in range(15)])
    regional = fit_regional(descriptors, valid, labels, people)
    query = rng.standard_normal((1, 1, 8, 768)).astype(np.float32)
    query_valid = np.ones((1, 1), dtype=np.bool_)
    scores = predict_regional(regional, query, query_valid)
    fused = fuse_regional(regional, scores, query_valid, video_logits)["P"]

    print(f"Motion gates {tuple(gate.shape)}")
    print(f"Paired video logits {video_logits[0, 0].round(4).tolist()}")
    print(f"Fused synthetic logits {fused[0, 0].round(4).tolist()}")


if __name__ == "__main__":
    main()
