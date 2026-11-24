from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from scoliosis5k import motion as model


class _FakeAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qkv = nn.Linear(384, 3 * 384)


class _FakeBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(384)
        self.attn = _FakeAttention()

    def forward(self, tokens: Tensor) -> Tensor:
        active_gate = model._ACTIVE_GATE.get()
        assert active_gate is not None
        indices = torch.tensor((0, 4), device=tokens.device)
        selected = tokens.index_select(1, indices)
        gate = active_gate.index_select(1, indices)
        q, _, v = self.attn.qkv(self.norm1(selected), gate).chunk(3, dim=-1)
        updated = selected + torch.tanh(0.01 * (q + v))
        output = tokens.clone()
        output[:, indices] = updated
        return output


class _FakeDINOv2(nn.Module):
    def __init__(self, seed: int = 9) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.projection = nn.Linear(3, 384)
            self.cls = nn.Parameter(torch.zeros(1, 1, 384))
            self.blocks = nn.ModuleList([_FakeBlock() for _ in range(12)])
            self.norm = nn.LayerNorm(384)
        self.chunked_blocks = False
        self.forward_batch_sizes: list[int] = []

    def prepare_tokens_with_masks(
        self, pixels: Tensor, masks: Tensor | None = None
    ) -> Tensor:
        if masks is not None:
            raise ValueError("the synthetic encoder does not use masks")
        self.forward_batch_sizes.append(pixels.shape[0])
        features = self.projection(pixels.mean(dim=(-2, -1)))
        patches = features.unsqueeze(1).expand(-1, 256, -1)
        cls = self.cls.expand(pixels.shape[0], -1, -1)
        return torch.cat((cls, patches), dim=1)


def _images_with_motion(mode: str) -> Tensor:
    images = torch.zeros((1, 2, 3, 224, 149), dtype=torch.float32)
    if mode == "uniform":
        images[:, 1] = 0.5
    elif mode == "localized":
        images[0, 1, :, :14, 5:19] = 1.0
    else:
        raise ValueError(mode)
    return images


def test_motion_gate_uses_real_patches_and_ignores_invalid_frames() -> None:
    images = _images_with_motion("localized")
    images = torch.cat((images, torch.full_like(images[:, :1], float("nan"))), dim=1)
    frame_valid = torch.tensor([[True, True, False]])

    gates = model.compute_motion_gate(images, frame_valid)

    assert gates.shape == (1, 3, 257)
    assert torch.all(gates[0, :, 0] == 0.5)
    assert torch.all(gates[0, 2] == 0.5)
    # Padded columns 0, 1, 14, and 15 contain no source pixels.
    for column in (0, 1, 14, 15):
        assert torch.all(gates[0, :2, 1 + column] == 0.5)
    assert torch.all(gates[0, :2, 1 + 3] > 0.5)
    assert torch.all(gates[0, :2, 1 + 8] == 0.0)


def test_repeated_random_texture_has_exact_half_gate_with_mixed_valid_frames() -> None:
    generator = torch.Generator().manual_seed(20261118)
    texture = torch.rand((2, 1, 3, 224, 149), generator=generator)
    images = texture.expand(2, 24, 3, 224, 149).clone()
    frame_valid = torch.ones((2, 24), dtype=torch.bool)
    frame_valid[1, ::2] = False
    images[~frame_valid] = float("nan")

    gates = model.compute_motion_gate(images, frame_valid)

    torch.testing.assert_close(gates, torch.full_like(gates, 0.5), rtol=0.0, atol=0.0)


def test_static_window_has_half_gate_and_rank4_equivalent_update() -> None:
    constant = torch.full((2, 3, 3, 224, 149), 0.375)
    valid = torch.ones((2, 3), dtype=torch.bool)
    gates = model.compute_motion_gate(constant, valid)
    torch.testing.assert_close(gates, torch.full_like(gates, 0.5), rtol=0.0, atol=0.0)

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(51)
        layer = model.MotionConditionedQVLoRALinear(nn.Linear(384, 1152))
    with torch.no_grad():
        for name, parameter in layer.named_parameters():
            if "_B" in name:
                parameter.normal_(mean=0.0, std=0.02)
    inputs = torch.randn((2, 3, 384), generator=torch.Generator().manual_seed(72))
    half_gate = torch.full((2, 3), 0.5)
    actual = layer(inputs, half_gate)

    q_base, k_base, v_base = F.linear(inputs, layer.weight, layer.bias).chunk(3, dim=-1)
    q_a = torch.cat((layer.lora_q_A0, layer.lora_q_A1), dim=0)
    q_b = torch.cat((layer.lora_q_B0, layer.lora_q_B1), dim=1)
    v_a = torch.cat((layer.lora_v_A0, layer.lora_v_A1), dim=0)
    v_b = torch.cat((layer.lora_v_B0, layer.lora_v_B1), dim=1)
    expected = torch.cat(
        (
            q_base + F.linear(F.linear(inputs, q_a), q_b),
            k_base,
            v_base + F.linear(F.linear(inputs, v_a), v_b),
        ),
        dim=-1,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


def _build_adapter(*, checkpoint_blocks: bool, seed: int = 33):
    return model.MotionConditionedLoRADenseAdapter(
        _FakeDINOv2(),
        arm="D",
        seed=seed,
        checkpoint_blocks=checkpoint_blocks,
    )


def _encode_full_window(adapter, images: Tensor, frame_valid: Tensor) -> Tensor:
    batch, time = images.shape[:2]
    gates = model.compute_motion_gate(images, frame_valid)
    indices = torch.nonzero(frame_valid.reshape(-1), as_tuple=False).flatten()
    frames = images.reshape(batch * time, 3, 224, 149).index_select(0, indices)
    patches = adapter._forward_patch_tokens(
        adapter._preprocess(frames),
        gates.reshape(batch * time, 257).index_select(0, indices),
    )
    dense = torch.zeros((batch * time, 256, 384), dtype=patches.dtype)
    dense.index_copy_(0, indices, patches)
    return dense.reshape(batch, time, 256, 384)


def test_encode_dense_handles_invalid_nan_frames_and_returns_zero_padding() -> None:
    rng_before = torch.random.get_rng_state().clone()
    adapter = _build_adapter(checkpoint_blocks=False)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    images = torch.cat(
        (_images_with_motion("localized"), torch.full((1, 1, 3, 224, 149), float("nan"))),
        dim=1,
    )
    valid = torch.tensor([[True, True, False]])

    dense, returned_valid = adapter.encode_dense(images, valid)

    assert adapter.arm == "D"
    assert adapter.model is not None
    assert torch.equal(returned_valid, valid)
    assert dense.shape == (1, 3, 256, 384)
    assert torch.isfinite(dense[valid]).all()
    assert torch.count_nonzero(dense[~valid]).item() == 0
    trainable_lora = [
        parameter for name, parameter in adapter.named_parameters()
        if ".lora_" in name and parameter.requires_grad
    ]
    assert sum(parameter.numel() for parameter in trainable_lora) == 73_728


def test_static_arm_matches_dynamic_arm_when_the_window_has_no_motion() -> None:
    static = model.MotionConditionedLoRADenseAdapter(
        _FakeDINOv2(), arm="C", seed=18, checkpoint_blocks=False
    )
    dynamic = model.MotionConditionedLoRADenseAdapter(
        _FakeDINOv2(), arm="D", seed=18, checkpoint_blocks=False
    )
    dynamic.load_state_dict(static.state_dict(), strict=True)
    with torch.no_grad():
        for name, parameter in static.named_parameters():
            if "_B" in name:
                parameter.normal_(mean=0.0, std=0.01)
    dynamic.load_state_dict(static.state_dict(), strict=True)
    constant = torch.full((1, 2, 3, 224, 149), 0.25)
    valid = torch.ones((1, 2), dtype=torch.bool)

    static_features, _ = static.encode_dense(constant, valid)
    dynamic_features, _ = dynamic.encode_dense(constant, valid)

    assert static.arm == "C"
    assert dynamic.arm == "D"
    torch.testing.assert_close(static_features, dynamic_features, rtol=0.0, atol=0.0)


def test_two_microbatch_checkpoint_recompute_keeps_each_motion_gate() -> None:
    checkpointed = _build_adapter(checkpoint_blocks=True)
    direct = _build_adapter(checkpoint_blocks=False)
    direct.load_state_dict(checkpointed.state_dict(), strict=True)
    with torch.no_grad():
        for name, parameter in checkpointed.named_parameters():
            if "_B" in name:
                parameter.normal_(mean=0.0, std=0.01)
    direct.load_state_dict(checkpointed.state_dict(), strict=True)

    batch_a = _images_with_motion("uniform")
    batch_b = _images_with_motion("localized")
    valid = torch.ones((1, 2), dtype=torch.bool)
    gate_a = model.compute_motion_gate(batch_a, valid)
    gate_b = model.compute_motion_gate(batch_b, valid)
    assert not torch.equal(gate_a[:, :, 4], gate_b[:, :, 4])

    loss_weights = torch.linspace(-1.0, 1.0, 384)
    for adapter in (checkpointed, direct):
        adapter.train()
        first, _ = adapter.encode_dense(batch_a, valid)
        second, _ = adapter.encode_dense(batch_b, valid)
        loss = (first[:, :, 3] * loss_weights).sum()
        loss = loss + (second[:, :, 3] * -loss_weights).sum()
        loss.backward()

    checkpointed_parameters = dict(checkpointed.named_parameters())
    direct_parameters = dict(direct.named_parameters())
    assert checkpointed_parameters.keys() == direct_parameters.keys()
    lora_branch_gradients = 0
    for name, parameter in checkpointed_parameters.items():
        other = direct_parameters[name]
        assert (parameter.grad is None) is (other.grad is None)
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, other.grad, rtol=1e-5, atol=1e-7)
            if ".lora_" in name and name.endswith(("_B0", "_B1")):
                assert torch.count_nonzero(parameter.grad).item() > 0
                lora_branch_gradients += 1
    assert lora_branch_gradients == 48


def test_encode_dense_splits_33_valid_frames_as_32_plus_tail() -> None:
    chunked = _build_adapter(checkpoint_blocks=True, seed=37)
    full = _build_adapter(checkpoint_blocks=True, seed=37)
    full.load_state_dict(chunked.state_dict(), strict=True)
    with torch.no_grad():
        for name, parameter in chunked.named_parameters():
            if "_B" in name:
                parameter.normal_(mean=0.0, std=0.01)
    full.load_state_dict(chunked.state_dict(), strict=True)

    images = torch.empty((2, 24, 3, 224, 149), dtype=torch.float32)
    for participant in range(2):
        for frame in range(24):
            images[participant, frame] = (participant * 24 + frame) / 47.0
    valid = torch.zeros((2, 24), dtype=torch.bool)
    valid[0] = True
    valid[1, :9] = True
    images[~valid] = float("nan")

    actual, _ = chunked.encode_dense(images, valid)
    expected = _encode_full_window(full, images, valid)

    assert chunked.model.forward_batch_sizes == [32, 1]
    assert full.model.forward_batch_sizes == [33]
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    loss_weights = torch.linspace(-1.0, 1.0, 384)
    (actual[:, :, 3] * loss_weights).sum().backward()
    (expected[:, :, 3] * loss_weights).sum().backward()
    chunked_parameters = dict(chunked.named_parameters())
    full_parameters = dict(full.named_parameters())
    assert chunked_parameters.keys() == full_parameters.keys()
    lora_gradients = 0
    for name, parameter in chunked_parameters.items():
        other = full_parameters[name]
        if ".lora_" not in name:
            assert parameter.grad is None
            continue
        assert parameter.grad is not None and other.grad is not None
        torch.testing.assert_close(parameter.grad, other.grad, rtol=1e-4, atol=1e-6)
        lora_gradients += int(torch.count_nonzero(parameter.grad).item() > 0)
    assert lora_gradients > 0
