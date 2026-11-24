"""Synthetic checks for the fixed PE-Spatial regional kernel pipeline."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from scoliosis5k.regional import (
    _bandwidth,
    _fit_panel,
    _fit_temperature,
    _kernels,
    _window_scores,
    fit_regional,
    fuse_regional,
    load_regional,
    predict_regional,
    save_regional,
)
from scoliosis5k.pe import extract_regions


def _unit_regions(seed, windows):
    random = np.random.default_rng(seed)
    values = random.normal(size=(windows, 8, 768)).astype(np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


def _synthetic_training_panel():
    random = np.random.default_rng(814)
    people = np.asarray([f"fixture-person-{index:03d}" for index in range(16)])
    labels = np.arange(16, dtype=np.int64) % 3
    valid = np.ones((16, 2), dtype=np.bool_)
    valid[-1] = False
    regions = np.zeros((16, 2, 8, 768), dtype=np.float32)
    for person, label in enumerate(labels[:-1]):
        values = random.normal(scale=0.08, size=(2, 8, 768)).astype(np.float32)
        values[..., label] += 1.0
        values /= np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)
        regions[person] = values
    return regions.astype(np.float16), valid, labels, people


class _SyntheticPEEncoder:
    def forward_features(self, pixels, *, layer_idx, strip_cls_token):
        self.pixels = pixels
        self.layer_idx = layer_idx
        self.strip_cls_token = strip_cls_token
        return torch.ones((1, 1024, 768), dtype=torch.float32)


class PEExtractionTests(unittest.TestCase):
    def test_native_rgb_produces_normalized_fp16_regions(self):
        encoder = _SyntheticPEEncoder()
        frame = np.empty((12, 20, 3), dtype=np.uint8)
        frame[...] = (0, 127, 255)

        regions = extract_regions(encoder, frame)

        self.assertEqual(regions.shape, (8, 768))
        self.assertEqual(regions.dtype, np.float16)
        self.assertTrue(np.isfinite(regions).all())
        np.testing.assert_allclose(np.linalg.norm(regions.astype(np.float32), axis=1), 1.0,
                                   rtol=0, atol=5e-4)
        self.assertEqual(tuple(encoder.pixels.shape), (1, 3, 512, 512))
        self.assertEqual((encoder.layer_idx, encoder.strip_cls_token), (-1, True))
        np.testing.assert_allclose(
            encoder.pixels[0, :, 0, 0].numpy(),
            np.asarray((-1.0, 127 / 127.5 - 1.0, 1.0), dtype=np.float32),
            rtol=0,
            atol=1e-7,
        )


class RegionalKernelTests(unittest.TestCase):
    def test_window_predictions_average_to_direct_centered_person_ridge(self):
        windows, query = _unit_regions(11, 5), _unit_regions(12, 2)
        counts = np.array([2, 1, 2])
        labels = np.array([0, 1, 2])
        groups = (slice(0, 2), slice(2, 3), slice(3, 5))
        ridge = 0.1
        panel = _fit_panel(windows, counts, labels, (ridge,))
        train_kernels = _kernels(windows, windows, panel.bandwidth)
        query_kernels = _kernels(query, windows, panel.bandwidth)
        center = np.eye(3) - np.full((3, 3), 1 / 3)
        for train, held, actual in zip(train_kernels, query_kernels, _window_scores(panel, query)):
            gram = np.array([[train[i, j].mean() for j in groups] for i in groups])
            cross = np.array([held[:, j].mean() for j in groups])
            centered = cross - cross.mean() - gram.mean(axis=0) + gram.mean()
            direct = centered @ np.linalg.solve(center @ gram @ center + ridge * np.eye(3), center) + 1 / 3
            np.testing.assert_allclose(actual[:, 0, :].mean(axis=0), direct, atol=2e-7, rtol=0)

    def test_kernel_matches_literal_position_and_all_pairs_means(self):
        left = _unit_regions(1, 3)
        right = _unit_regions(2, 4)
        bandwidth = 0.73
        aligned, all_pairs = _kernels(left, right, bandwidth)
        self.assertTrue(np.isfinite(aligned).all())
        self.assertTrue(np.isfinite(all_pairs).all())

        expected_aligned = np.empty((3, 4), dtype=np.float64)
        expected_all_pairs = np.empty((3, 4), dtype=np.float64)
        for i in range(3):
            for j in range(4):
                similarities = np.empty((8, 8), dtype=np.float64)
                for row in range(8):
                    for column in range(8):
                        distance = np.sum((left[i, row] - right[j, column]) ** 2)
                        similarities[row, column] = np.exp(-distance / bandwidth)
                expected_aligned[i, j] = np.diag(similarities).mean()
                expected_all_pairs[i, j] = similarities.mean()

        np.testing.assert_allclose(aligned, expected_aligned, rtol=2e-6, atol=2e-8, equal_nan=False)
        np.testing.assert_allclose(all_pairs, expected_all_pairs, rtol=2e-6, atol=2e-8, equal_nan=False)
        np.testing.assert_allclose(all_pairs, _kernels(left, right[:, ::-1], bandwidth)[1],
                                   rtol=1e-6, atol=2e-8, equal_nan=False)
        self.assertGreater(np.max(np.abs(aligned - _kernels(left, right[:, ::-1], bandwidth)[0])),
                           1e-5)

    def test_both_kernel_gram_matrices_are_symmetric_positive_semidefinite(self):
        features = _unit_regions(5, 7)
        position, content = _kernels(features, features, 0.91)
        for gram in (position, content):
            self.assertTrue(np.isfinite(gram).all())
            np.testing.assert_allclose(gram, gram.T, rtol=1e-6, atol=2e-8, equal_nan=False)
            self.assertGreaterEqual(float(np.linalg.eigvalsh(gram).min()), -1e-7)

    def test_bandwidth_is_stable_and_has_the_prespecified_floor(self):
        repeated = np.zeros((6, 8, 768), dtype=np.float32)
        repeated[..., 0] = 1.0
        self.assertEqual(_bandwidth(repeated), 1e-4)
        varying = _unit_regions(6, 6)
        self.assertEqual(_bandwidth(varying), _bandwidth(varying.copy()))

    def test_temperature_fit_is_bounded_and_reports_its_cross_entropy(self):
        labels = np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64)
        scores = np.full((6, 3), -3.0, dtype=np.float64)
        scores[np.arange(6), labels] = 3.0
        temperature, ce = _fit_temperature(scores, labels)
        self.assertEqual(temperature, 16.0)
        self.assertTrue(0.25 <= temperature <= 16.0)
        self.assertTrue(np.isfinite(ce))
        shifted = scores - scores.max(axis=1, keepdims=True)
        expected_ce = np.mean(
            np.log(np.exp(temperature * shifted).sum(axis=1))
            - temperature * shifted[np.arange(len(labels)), labels]
        )
        self.assertAlmostEqual(ce, float(expected_ce), places=12)


class RegionalFitAndPredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.regions, cls.valid, cls.labels, cls.people = _synthetic_training_panel()
        cls.model = fit_regional(cls.regions, cls.valid, cls.labels, cls.people)

    def test_private_npz_reloads_exact_raw_and_fused_logits(self):
        queries = np.repeat(self.regions[:2, :1], 20, axis=1)
        valid = np.ones((2, 20), dtype=bool)
        valid[0, 3] = False
        prior = np.linspace(-1, 1, 120, dtype=np.float32).reshape(2, 20, 3)
        original = predict_regional(self.model, queries, valid)
        original_fused = fuse_regional(self.model, original, valid, prior)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "private_model.npz"
            save_regional(self.model, path)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                save_regional(self.model, path)
            reloaded = load_regional(path)
            self.assertEqual(reloaded.summary, self.model.summary)
            predictions = predict_regional(reloaded, queries, valid)
            fused = fuse_regional(reloaded, predictions, valid, prior)
            for arm in ("P", "A"):
                np.testing.assert_array_equal(predictions[arm], original[arm])
                np.testing.assert_array_equal(fused[arm], original_fused[arm])

    def test_fit_excludes_people_without_windows_and_reports_both_arms(self):
        summary = self.model.summary
        self.assertEqual(summary["fit_people"], 15)
        self.assertEqual(summary["excluded_people"], 1)
        self.assertEqual(summary["fit_windows"], 30)
        self.assertGreaterEqual(summary["bandwidth"], 1e-4)
        self.assertEqual(set(summary["arms"]), {"P", "A"})
        for arm in ("P", "A"):
            result = summary["arms"][arm]
            self.assertIn(result["lambda"], (0.01, 0.1, 1.0, 10.0))
            self.assertTrue(0.25 <= result["inverse_temperature"] <= 16.0)
            self.assertTrue(np.isfinite(result["OOF_CE"]))

    def test_fitting_is_invariant_to_input_row_order(self):
        order = np.arange(len(self.people))[::-1]
        reordered = fit_regional(self.regions[order], self.valid[order], self.labels[order],
                                 self.people[order])
        self.assertEqual(reordered.summary["fit_people"], self.model.summary["fit_people"])
        self.assertEqual(reordered.summary["excluded_people"], self.model.summary["excluded_people"])
        self.assertAlmostEqual(reordered.summary["bandwidth"], self.model.summary["bandwidth"], places=12)
        for arm in ("P", "A"):
            for key in ("lambda", "inverse_temperature", "OOF_CE"):
                self.assertAlmostEqual(reordered.summary["arms"][arm][key],
                                       self.model.summary["arms"][arm][key], places=10)

    def test_prediction_preserves_slots_and_fusion_uses_baseline_for_missing_images(self):
        one = self.regions[0, 0].astype(np.float32)
        query = np.zeros((2, 2, 8, 768), dtype=np.float32)
        query[0, 0] = one
        query[1, 0] = one
        query[1, 1] = self.regions[1, 1].astype(np.float32)
        valid = np.asarray([[True, False], [True, True]], dtype=np.bool_)
        branch = predict_regional(self.model, query, valid)
        self.assertEqual(set(branch), {"P", "A"})
        for arm in ("P", "A"):
            self.assertEqual(branch[arm].shape, (2, 2, 3))
            self.assertEqual(branch[arm].dtype, np.float32)
            self.assertTrue(np.isfinite(branch[arm]).all())
            np.testing.assert_array_equal(branch[arm][0, 1], np.zeros(3, dtype=np.float32))
            np.testing.assert_allclose(branch[arm][0, 0], branch[arm][1, 0], rtol=0, atol=0)

        baseline = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
        fused = fuse_regional(self.model, branch, valid, baseline)
        for arm in ("P", "A"):
            np.testing.assert_array_equal(fused[arm][~valid], baseline[~valid])
            scale = np.float32(0.35 * self.model.summary["arms"][arm]["inverse_temperature"])
            expected = np.float32(0.65) * baseline[valid] + scale * branch[arm][valid]
            np.testing.assert_array_equal(fused[arm][valid], expected)


if __name__ == "__main__":
    unittest.main()
