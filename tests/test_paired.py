"""CPU fixtures using the original Paired model and source interfaces."""

from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from scoliosis5k.paired import (
    ADAPTATION_PARAMETERS, ADAPTATION_LR, HEAD_LR, build_motion_paired, expected_updates,
    load_delta, make_optimizer, save_delta, train_step,
)


class _SyntheticMotionEncoder(nn.Module):
    def __init__(self, arm="D"):
        super().__init__()
        self.arm = arm
        self.model = nn.Linear(1, 1).requires_grad_(False)
        self.factors = nn.Parameter(torch.linspace(-0.2, 0.2, ADAPTATION_PARAMETERS).reshape(-1, 384))

    def encode_dense(self, frames, valid):
        self.last_valid = valid
        vector = self.factors.mean(dim=0)[None, None] + frames[..., 0, 0].mean(-1, keepdim=True)
        return vector[:, :, None].expand(len(valid), 24, 256, 384), valid


class ConnectorTests(unittest.TestCase):
    def test_original_head_gradients_two_group_rates_and_strict_delta(self):
        from scoliosis5k.distillation import CrossModalTeacherT1

        torch.set_num_threads(1)
        seed = 73112
        torch.manual_seed(seed)
        original = CrossModalTeacherT1(mode="paired")
        core = _SyntheticMotionEncoder()
        legacy = _SyntheticMotionEncoder()
        del legacy.arm
        with self.assertRaisesRegex(ValueError, "actual D or C"):
            build_motion_paired(legacy, seed=seed)
        rng = torch.random.get_rng_state().clone()
        model = build_motion_paired(core, seed=seed)
        torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
        for key, tensor in original.state_dict().items():
            torch.testing.assert_close(tensor, model.paired.state_dict()[key], rtol=0, atol=0)
        self.assertTrue(model.paired.training and core.training)
        self.assertFalse(core.model.training)

        total = expected_updates(631, 30)
        self.assertEqual(total, 2370)
        self.assertEqual([expected_updates(n, 60) for n in (789, 791, 792)], [5940] * 3)
        optimizer, scheduler = make_optimizer(model, total)
        groups = {(group["initial_lr"], group["weight_decay"]) for group in optimizer.param_groups}
        self.assertEqual(groups, {(HEAD_LR, 0.01), (HEAD_LR, 0.0), (ADAPTATION_LR, 0.01)})
        torch.testing.assert_close(torch.tensor(scheduler.get_last_lr()),
                                   torch.tensor([group["initial_lr"] / round(total * 0.05)
                                                 for group in optimizer.param_groups]))
        batch = {"aux_images": torch.rand(1, 24, 3, 224, 149),
                 "aux_valid": torch.arange(24)[None] < 20,
                 "silhouettes": torch.rand(1, 24, 64, 64),
                 "frame_valid": torch.arange(24)[None] < 20,
                 "labels": torch.tensor([2])}
        target = torch.randn(1, 384, requires_grad=True)
        result = train_step(model, batch, target, optimizer, scheduler)
        self.assertAlmostEqual(result["total_loss"], result["classification_CE"] + result["paired_cosine"], places=6)
        self.assertIs(core.last_valid, batch["aux_valid"])
        self.assertGreater(float(core.factors.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.paired.predictor[-1].weight.grad.abs().sum()), 0.0)
        self.assertIsNone(core.model.weight.grad)
        self.assertIsNone(target.grad)

        identity = {"arm": "D", "seed": seed, "source": "synthetic-original", "global_step": 1}
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "motion.pt"
            save_delta(model, checkpoint, identity=identity)
            self.assertEqual(checkpoint.stat().st_mode & 0o777, 0o600)
            saved = torch.load(checkpoint, weights_only=True)
            self.assertNotIn("adapted_rgb.model.weight", saved["delta"])
            reloaded = build_motion_paired(_SyntheticMotionEncoder(), seed=seed)
            with self.assertRaisesRegex(ValueError, "actual model arm"):
                load_delta(reloaded, checkpoint, identity={**identity, "arm": "C"})
            wrong_model = build_motion_paired(_SyntheticMotionEncoder("C"), seed=seed)
            with self.assertRaisesRegex(ValueError, "actual model arm"):
                load_delta(wrong_model, checkpoint, identity=identity)
            load_delta(reloaded, checkpoint, identity=identity)
        with torch.inference_mode():
            torch.testing.assert_close(model.eval()(batch)["logits"], reloaded(batch)["logits"], rtol=0, atol=0)
