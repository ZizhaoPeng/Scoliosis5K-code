import numpy as np
import torch

from scoliosis5k.preprocessing import prepare_window


def test_prepare_window_preserves_slots_and_selects_earlier_tie() -> None:
    earlier = np.full((64, 48, 3), 100, dtype=np.uint8)
    later = np.full_like(earlier, 200)
    silhouette = np.full((64, 64), 255, dtype=np.uint8)
    rgb = [None] * 24
    silhouettes = [None] * 24
    rgb[10], rgb[12] = earlier, later
    silhouettes[10] = silhouette

    batch, regional_frame = prepare_window(rgb, silhouettes)

    assert regional_frame is earlier
    assert batch["aux_valid"].nonzero()[:, 1].tolist() == [10, 12]
    assert batch["frame_valid"].nonzero()[:, 1].tolist() == [10]
    torch.testing.assert_close(batch["aux_images"][0, 10], torch.full((3, 224, 149), 100 / 255))
    assert batch["silhouettes"][0, 10].eq(1).all()
    assert batch["silhouettes"][0, 12].eq(0).all()
