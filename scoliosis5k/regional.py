"""Fixed person-level regional kernel classifier for PE-Spatial."""

from dataclasses import dataclass
import json
import os
from pathlib import Path

import numpy as np


ARMS = ("P", "A")
LAMBDAS = (0.01, 0.1, 1.0, 10.0)
TEMPERATURE_BOUNDS = (0.25, 16.0)
SEED = 20261118
WINDOW_BLOCK = 32


@dataclass(frozen=True)
class _RidgePanel:
    windows: np.ndarray
    counts: np.ndarray
    bandwidth: float
    prior: np.ndarray
    column_means: tuple[np.ndarray, np.ndarray]
    weights: tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class RegionalClassifier:
    _panel: _RidgePanel
    _selected_columns: tuple[int, int]
    summary: dict


def save_regional(model: RegionalClassifier, path: Path) -> None:
    """Save the fitted PE readout and its TRAIN support windows as a private NPZ."""
    panel = model._panel
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        np.savez(
            stream, windows=panel.windows, counts=panel.counts, bandwidth=np.float64(panel.bandwidth),
            prior=panel.prior, P_mean=panel.column_means[0], A_mean=panel.column_means[1],
            P_weights=panel.weights[0], A_weights=panel.weights[1],
            selected_columns=np.asarray(model._selected_columns, dtype=np.int64),
            summary=json.dumps(model.summary, allow_nan=False),
        )


def load_regional(path: Path) -> RegionalClassifier:
    """Reload a saved readout without executing code or changing descriptor precision."""
    with np.load(path, allow_pickle=False) as archive:
        windows, counts, prior = archive["windows"], archive["counts"], archive["prior"]
        bandwidth = archive["bandwidth"]
        means = tuple(archive[f"{arm}_mean"] for arm in ARMS)
        weights = tuple(archive[f"{arm}_weights"] for arm in ARMS)
        columns = archive["selected_columns"]
        summary = json.loads(archive["summary"].item())

    people = counts.size
    selected = [summary["arms"][arm] for arm in ARMS]
    lambdas = tuple(sorted({arm["lambda"] for arm in selected}))
    expected_columns = tuple(lambdas.index(arm["lambda"]) for arm in selected)
    if (
        counts.ndim != 1 or counts.dtype != np.int64 or people == 0 or np.any((counts < 1) | (counts > 2))
        or windows.dtype != np.float32 or windows.shape != (int(counts.sum()), 8, 768)
        or bandwidth.shape != () or bandwidth.dtype != np.float64 or float(bandwidth) < 1e-4
        or prior.shape != (3,) or any(array.shape != (people,) for array in means)
        or any(array.shape != (people, len(lambdas) * 3) for array in weights)
        or any(array.dtype != np.float64 for array in (prior, *means, *weights))
        or any(not np.isfinite(array).all() for array in (windows, bandwidth, prior, *means, *weights))
        or columns.dtype != np.int64 or not np.array_equal(columns, expected_columns)
        or summary["fit_people"] != people or summary["fit_windows"] != len(windows)
        or summary["bandwidth"] != float(bandwidth) or any(value not in LAMBDAS for value in lambdas)
        or any(not 0.25 <= arm["inverse_temperature"] <= 16.0 for arm in selected)
    ):
        raise ValueError("saved PE readout differs from the fixed regional model structure")
    panel = _RidgePanel(windows, counts, float(bandwidth), prior, means, weights)
    return RegionalClassifier(panel, expected_columns, summary)


def _validated_windows(regions: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if (
        regions.ndim != 4 or regions.shape[2:] != (8, 768) or regions.shape[1] == 0
        or regions.dtype not in (np.float16, np.float32)
        or valid.shape != regions.shape[:2] or valid.dtype != np.bool_
    ):
        raise ValueError("expected float16/float32 [people,windows,8,768] and a bool window mask")
    values = regions[valid].astype(np.float32, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("valid regional descriptors must be finite")
    values /= np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), np.float32(1e-12))
    return values


def _bandwidth(first_windows: np.ndarray) -> float:
    order = np.random.default_rng(SEED).permutation(len(first_windows))[:128]
    selected = first_windows[order]
    upper = np.triu_indices(len(selected), k=1)
    distances = []
    for region in range(8):
        vectors = selected[:, region, :]
        squared = np.einsum("ij,ij->i", vectors, vectors)
        matrix = squared[:, None] + squared[None, :] - 2 * (vectors @ vectors.T)
        distances.append(np.maximum(matrix[upper], 0))
    return max(float(np.median(np.concatenate(distances))), 1e-4)


def _kernels(left: np.ndarray, right: np.ndarray, bandwidth: float) -> tuple[np.ndarray, np.ndarray]:
    x, y = left.reshape(-1, 768), right.reshape(-1, 768)
    x_squared = np.einsum("ij,ij->i", x, x)
    y_squared = np.einsum("ij,ij->i", y, y)
    values = x @ y.T
    values *= -2
    values += x_squared[:, None]
    values += y_squared[None, :]
    np.maximum(values, 0, out=values)
    values /= -bandwidth
    np.exp(values, out=values)
    pairs = values.reshape(len(left), 8, len(right), 8)
    aligned = pairs.diagonal(axis1=1, axis2=3).mean(axis=-1, dtype=np.float64)
    unaligned = pairs.mean(axis=(1, 3), dtype=np.float64)
    return aligned, unaligned


def _fit_panel(windows: np.ndarray, counts: np.ndarray, labels: np.ndarray,
               lambdas: tuple[float, ...]) -> _RidgePanel:
    size = len(counts)
    edges = np.concatenate(([0], counts.cumsum()))
    starts = edges[:-1]
    bandwidth = _bandwidth(windows[starts])
    target = np.eye(3)[labels]
    prior = target.mean(axis=0)
    centered_target = target - prior
    grams = [np.empty((size, size), dtype=np.float64) for _ in ARMS]
    for start in range(0, size, WINDOW_BLOCK // 2):
        stop = min(start + WINDOW_BLOCK // 2, size)
        block = windows[edges[start]:edges[stop]]
        for gram, kernel in zip(grams, _kernels(block, windows, bandwidth)):
            by_person = np.add.reduceat(kernel, starts, axis=1) / counts
            local = starts[start:stop] - edges[start]
            gram[start:stop] = np.add.reduceat(by_person, local, axis=0) / counts[start:stop, None]

    means, weights = [], []
    for gram in grams:
        gram = (gram + gram.T) / 2
        mean = gram.mean(axis=0)
        centered = gram - mean[None, :] - mean[:, None] + mean.mean()
        eigenvalues, basis = np.linalg.eigh(centered)
        shifted = eigenvalues[:, None] + np.asarray(lambdas)[None, :]
        projected = (basis.T @ centered_target)[:, None, :] / shifted[:, :, None]
        means.append(mean)
        weights.append(basis @ projected.reshape(size, -1))
    return _RidgePanel(windows, counts, bandwidth, prior, tuple(means), tuple(weights))


def _window_scores(panel: _RidgePanel, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    starts = np.concatenate(([0], panel.counts.cumsum()[:-1]))
    scores = tuple(np.empty((len(query), weights.shape[1] // 3, 3)) for weights in panel.weights)
    for start in range(0, len(query), WINDOW_BLOCK):
        end = min(start + WINDOW_BLOCK, len(query))
        kernels = _kernels(query[start:end], panel.windows, panel.bandwidth)
        for block, mean, weights, kernel in zip(scores, panel.column_means, panel.weights, kernels):
            per_person = np.add.reduceat(kernel, starts, axis=1) / panel.counts
            centered = per_person - per_person.mean(axis=1, keepdims=True) - mean + mean.mean()
            block[start:end] = (centered @ weights).reshape(end - start, -1, 3) + panel.prior
    return scores


def _fit_temperature(person_scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    scores = person_scores - person_scores.max(axis=1, keepdims=True)
    target = scores[np.arange(len(labels)), labels]

    def derivative(t: float) -> float:
        weights = np.exp(t * scores)
        expectation = (weights * scores).sum(axis=1) / weights.sum(axis=1)
        return float(np.mean(expectation - target))

    lo, hi = TEMPERATURE_BOUNDS
    # The derivative of CE is monotone, including when the minimum is an endpoint.
    if derivative(lo) >= 0:
        t = lo
    elif derivative(hi) <= 0:
        t = hi
    else:
        for _ in range(52):
            t = (lo + hi) / 2
            if derivative(t) < 0:
                lo = t
            else:
                hi = t
        t = (lo + hi) / 2
    ce = np.mean(np.log(np.exp(t * scores).sum(axis=1)) - t * target)
    return t, float(ce)


def fit_regional(train_regions: np.ndarray, train_valid: np.ndarray, labels: np.ndarray,
                 people: np.ndarray) -> RegionalClassifier:
    """Choose each kernel's ridge and inverse temperature using TRAIN people only."""
    from sklearn.model_selection import StratifiedKFold

    labels, people = np.asarray(labels), np.asarray(people)
    valid_windows = _validated_windows(train_regions, train_valid)
    size, width = train_valid.shape
    if (
        width != 2 or people.shape != (size,) or people.dtype.kind != "U"
        or len(np.unique(people)) != size or np.any(people == "")
        or labels.shape != (size,) or labels.dtype.kind not in "iu"
        or np.any((labels < 0) | (labels > 2))
    ):
        raise ValueError("fit requires two ordered window slots and one nominal label per unique person")
    regions = np.zeros(train_regions.shape, dtype=np.float32)
    regions[train_valid] = valid_windows
    order = np.argsort(people, kind="stable")
    order = order[train_valid[order].any(axis=1)]
    regions, valid, truth = regions[order], train_valid[order], labels[order].astype(np.int64)
    if np.any(np.bincount(truth, minlength=3) < 5):
        raise ValueError("stratified five-fold selection requires five valid people in each class")

    counts = valid.sum(axis=1)
    oof = {arm: np.empty((len(order), len(LAMBDAS), 3)) for arm in ARMS}
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for train, held in splitter.split(order, truth):
        panel = _fit_panel(regions[train][valid[train]], counts[train], truth[train], LAMBDAS)
        held_scores = _window_scores(panel, regions[held][valid[held]])
        starts = np.concatenate(([0], counts[held].cumsum()[:-1]))
        for arm, scores in zip(ARMS, held_scores):
            oof[arm][held] = np.add.reduceat(scores, starts, axis=0) / counts[held, None, None]

    selected = {}
    for arm in ARMS:
        candidates = [_fit_temperature(oof[arm][:, k], truth) for k in range(len(LAMBDAS))]
        index = min(range(len(LAMBDAS)), key=lambda k: (candidates[k][1], -LAMBDAS[k]))
        t, ce = candidates[index]
        selected[arm] = {"lambda": LAMBDAS[index], "inverse_temperature": t, "OOF_CE": ce}
    fitted = tuple(sorted({arm["lambda"] for arm in selected.values()}))
    panel = _fit_panel(regions[valid], counts, truth, fitted)
    summary = {
        "fit_people": len(order), "excluded_people": size - len(order),
        "fit_windows": int(counts.sum()), "bandwidth": panel.bandwidth, "arms": selected,
    }
    columns = tuple(fitted.index(selected[arm]["lambda"]) for arm in ARMS)
    return RegionalClassifier(panel, columns, summary)


def predict_regional(model: RegionalClassifier, regions: np.ndarray,
                     valid: np.ndarray) -> dict[str, np.ndarray]:
    """Score original window slots; zero is the placeholder for a missing image."""
    windows = _validated_windows(regions, valid)
    scores = _window_scores(model._panel, windows)
    result = {}
    for arm, block, index in zip(ARMS, scores, model._selected_columns):
        values = np.zeros((*valid.shape, 3), dtype=np.float32)
        values[valid] = block[:, index]
        result[arm] = values
    return result


def fuse_regional(model: RegionalClassifier, branch: dict[str, np.ndarray],
                  valid: np.ndarray, baseline: np.ndarray) -> dict[str, np.ndarray]:
    """Combine a frozen regional prediction with one Paired baseline seed."""
    if (
        valid.ndim != 2 or valid.dtype != np.bool_ or baseline.shape != (*valid.shape, 3)
        or baseline.dtype != np.float32 or not np.isfinite(baseline).all() or set(branch) != set(ARMS)
    ):
        raise ValueError("expected both regional arms, an aligned mask and float32 Paired scores")
    fused = {}
    for arm in ARMS:
        raw = branch[arm]
        if raw.shape != baseline.shape or raw.dtype != np.float32 or not np.isfinite(raw).all():
            raise ValueError("regional scores must match the finite float32 Paired score array")
        result = baseline.copy()
        scale = np.float32(0.35 * model.summary["arms"][arm]["inverse_temperature"])
        result[valid] = np.float32(0.65) * baseline[valid] + scale * raw[valid]
        fused[arm] = result
    return fused
