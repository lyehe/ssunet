"""Development diagnostics with an explicit training/validation data boundary."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from .datasets.neutrino import shard_sha256
from .neutrino_detection import PhotonWindows, calibrated_tail_probabilities
from .neutrino_detection_probe import balanced_log_loss, feature_design, predict_head


class DevelopmentStore:
    """Permit only declared development observations and frozen model metadata."""

    def __init__(self, root):
        """Anchor all accesses and retain an auditable list of relative input paths."""
        self.root = Path(root).resolve()
        self.accessed = set()

    def path(self, relative):
        """Reject test/calibration data, traversal and undeclared top-level files."""
        relative = Path(relative)
        top = {
            "config.json",
            "feature_contract.json",
            "head_selection.json",
            "head_lock.json",
            "frozen_reference.npz",
            "head_raw_prior.npz",
            "head_self_split.npz",
            "head_untrained.npz",
        }
        parts = relative.parts
        allowed = (
            len(parts) == 1
            and (parts[0] in top or re.fullmatch(r"(?:frozen|random)_seed_\d+\.pt", parts[0]))
        ) or (
            len(parts) == 2
            and parts[0] in ("train", "validation")
            and parts[1]
            in ("features.npz", "observations.npz", "injection_labels.npz", "fixed_scores.npz")
        )
        if not allowed:
            message = f"Development-only access rejected: {relative}"
            raise PermissionError(message)
        target = (self.root / relative).resolve()
        if not target.is_relative_to(self.root):
            message = f"Development-only access rejected: {relative}"
            raise PermissionError(message)
        self.accessed.add(relative.as_posix())
        return target

    def arrays(self, relative):
        """Read numeric arrays once without pickle, including scalar head parameters."""
        with np.load(self.path(relative), allow_pickle=False) as handle:
            values = {}
            for key in handle.files:
                value = handle[key]
                values[key] = value.item() if value.ndim == 0 else value
            return values

    def json(self, relative):
        """Read permitted metadata, without following any paths it contains."""
        return json.loads(self.path(relative).read_text(encoding="utf-8"))

    def digest(self, relative):
        """Fingerprint only an input allowed by the same access boundary."""
        return shard_sha256(self.path(relative))


def validation_tail_sensitivity(scores, truth, randomizers, *, seed, partitions, rates, strength):
    """Cross-calibrate within validation; repeated partitions are not new evidence."""
    signal = truth["expected_signal"] > 0
    groups = truth["source_group"]
    null_indices = np.flatnonzero(~signal)
    unique = np.unique(groups[signal])
    if len(null_indices) < 4 or len(unique) < 2:
        message = "Validation tail diagnosis requires background and independent source groups"
        raise ValueError(message)
    rng = np.random.default_rng(seed)
    source_order = rng.permutation(unique)
    source_fold = {int(group): index % 2 for index, group in enumerate(source_order)}
    signal_folds = np.asarray([source_fold.get(int(group), -1) for group in groups])
    records = []
    for repeat in range(partitions):
        null_fold = np.full(len(groups), -1)
        shuffled = rng.permutation(null_indices)
        null_fold[shuffled] = np.arange(len(shuffled)) % 2
        p = np.empty(len(scores))
        for fold in (0, 1):
            calibration = (~signal) & (null_fold != fold)
            assessment = ((~signal) & (null_fold == fold)) | (signal & (signal_folds == fold))
            p[assessment] = calibrated_tail_probabilities(
                scores[calibration], scores[assessment], randomizers[assessment]
            )
        for rate in rates:
            decision = p <= rate
            levels = {
                str(float(level)): float(decision[truth["expected_signal"] == level].mean())
                for level in np.unique(truth["expected_signal"][signal])
            }
            records.append(
                {
                    "partition": repeat,
                    "target_false_positive_rate": rate,
                    "false_positive_rate": float(decision[~signal].mean()),
                    "signal_efficiency": float(decision[signal].mean()),
                    "primary_efficiency": float(
                        decision[truth["expected_signal"] == strength].mean()
                    ),
                    "efficiency_by_expected_signal": levels,
                }
            )
    return records


def diagnostic_head_path(train, labels, validation, validation_labels, arm, config, guard):
    """Retain all fixed-budget candidates, including previously unselected failures."""
    settings = config["head"]
    dimension = train["self_split"].shape[1]
    extra = None if arm == "raw_prior" else train[arm]
    x = torch.as_tensor(feature_design(train["base"], extra, dimension), dtype=torch.float64)
    mean = x.mean(0)
    scale = x.std(0, unbiased=False).clamp_min(settings["feature_scale_floor"])
    x.sub_(mean).div_(scale).clamp_(-settings["feature_clip"], settings["feature_clip"])
    balanced_log_loss(np.zeros(len(labels)), labels)
    y = torch.as_tensor(labels, dtype=torch.float64)
    weights = torch.where(y > 0.5, 0.5 / (y > 0.5).sum(), 0.5 / (y < 0.5).sum())
    heads, records = [], []
    for penalty in settings["penalties"]:
        started = time.perf_counter()
        weight = torch.zeros(x.shape[1], dtype=torch.float64, requires_grad=True)
        intercept = torch.zeros((), dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.LBFGS(
            [weight, intercept],
            max_iter=settings["max_iterations"],
            history_size=settings["history_size"],
            tolerance_grad=settings["tolerance_grad"],
            tolerance_change=settings["tolerance_change"],
            line_search_fn="strong_wolfe",
        )

        def closure(optimizer=optimizer, weight=weight, intercept=intercept, penalty=penalty):
            optimizer.zero_grad()
            logits = x @ weight + intercept
            loss = (
                functional.binary_cross_entropy_with_logits(logits, y, reduction="none") * weights
            ).sum()
            loss = loss + 0.5 * penalty * weight.square().sum()
            loss.backward()
            guard()
            return loss

        optimizer.step(closure)
        final_loss = float(closure().detach())
        gradient = float(max(weight.grad.abs().max(), intercept.grad.abs()))
        if not np.isfinite(final_loss) or not np.isfinite(gradient):
            message = "Nonfinite diagnostic candidate; preserve the partial development path"
            raise RuntimeError(message)
        head = {
            "mean": mean.numpy().copy(),
            "scale": scale.numpy().copy(),
            "weight": weight.detach().numpy().copy(),
            "intercept": float(intercept.detach()),
            "penalty": float(penalty),
            "clip": settings["feature_clip"],
            "extra_dimension": dimension,
        }
        values = predict_head(
            head,
            validation["base"],
            None if arm == "raw_prior" else validation[arm],
            batch_size=config["runtime"]["head_batch_size"],
        )
        iterations = int(optimizer.state[weight]["n_iter"])
        heads.append(head)
        records.append(
            {
                "penalty": float(penalty),
                "validation_balanced_log_loss": balanced_log_loss(values, validation_labels),
                "training_balanced_log_loss": balanced_log_loss(
                    (x @ weight + intercept).detach().numpy(), labels
                ),
                "penalized_training_loss": final_loss,
                "max_absolute_gradient": gradient,
                "passes_original_gradient_gate": gradient <= 1e-4,
                "iterations": iterations,
                "function_evaluations": int(optimizer.state[weight]["func_evals"]),
                "iteration_budget_exhausted": iterations >= settings["max_iterations"],
                "seconds": time.perf_counter() - started,
            }
        )
    return heads, records


def permute_sensor_time_pairings(observed, bins, seed):
    """Destroy pairings while conserving each recorded half's two exact marginals."""
    rng = np.random.default_rng(seed)
    locations = observed.locations.copy()
    eligible, changed, full_changed = (np.zeros(len(observed), dtype=bool) for _ in range(3))
    for index in range(len(observed)):
        start, stop = observed.offsets[index : index + 2]
        rows = np.arange(start, stop)
        for half in (True, False):
            chosen = rows[observed.retained[start:stop] == half]
            sensor, time = np.divmod(observed.locations[chosen], bins)
            eligible[index] |= len(np.unique(sensor)) > 1 and len(np.unique(time)) > 1
            replacement = sensor * bins + rng.permutation(time)
            locations[chosen] = replacement
            changed[index] |= not np.array_equal(
                np.sort(replacement), np.sort(observed.locations[chosen])
            )
        full_changed[index] = not np.array_equal(
            np.sort(locations[start:stop]), np.sort(observed.locations[start:stop])
        )
    return PhotonWindows(
        observed.offsets.copy(), locations, observed.retained.copy(), observed.randomizers.copy()
    ), {
        "eligible": eligible,
        "half_joint_changed": changed,
        "full_joint_changed": full_changed,
    }


def pdf_hidden_indices(extra_dimension, sensor_groups, time_groups, hidden_dimension):
    """Recover the fixed PDF/hidden feature blocks, including symmetric A/B terms."""
    pdf = 2 * sensor_groups + time_groups
    half = pdf + hidden_dimension
    if extra_dimension != 2 * half + 7:
        message = "Unexpected frozen encoder feature layout"
        raise ValueError(message)
    pdf_indices = np.r_[
        np.arange(pdf), np.arange(half, half + pdf), np.arange(2 * half, extra_dimension)
    ]
    hidden_indices = np.r_[np.arange(pdf, half), np.arange(half + pdf, 2 * half)]
    return pdf_indices, hidden_indices


def ridge_timing_path(base, extra, targets, settings, penalties):
    """Fit fixed timing decoders on training data only, with no validation selection."""
    dimension = extra.shape[1]
    x = torch.as_tensor(feature_design(base, extra, dimension), dtype=torch.float64)
    mean, scale = x.mean(0), x.std(0, unbiased=False).clamp_min(settings["feature_scale_floor"])
    x.sub_(mean).div_(scale).clamp_(-settings["feature_clip"], settings["feature_clip"])
    center = x.mean(0)
    x.sub_(center)
    y = torch.as_tensor(targets, dtype=torch.float64)
    ymean, yscale = y.mean(0), y.std(0, unbiased=False).clamp_min(1e-8)
    y = (y - ymean) / yscale
    gram, rhs = x.T @ x / len(x), x.T @ y / len(x)
    identity = torch.eye(x.shape[1], dtype=torch.float64)
    heads = []
    for penalty in penalties:
        chol = torch.linalg.cholesky(gram + penalty * identity)
        weight = torch.cholesky_solve(rhs, chol)
        heads.append(
            {
                "mean": mean.numpy().copy(),
                "scale": scale.numpy().copy(),
                "center": center.numpy().copy(),
                "weight": weight.numpy().copy(),
                "target_mean": ymean.numpy().copy(),
                "target_scale": yscale.numpy().copy(),
                "clip": settings["feature_clip"],
                "extra_dimension": dimension,
                "penalty": penalty,
            }
        )
    return heads


def predict_timing(head, base, extra, *, batch_size=1024):
    """Apply a fixed decoder without changing target/feature preprocessing."""
    result = np.empty((len(base), len(head["target_mean"])))
    for start in range(0, len(base), batch_size):
        stop = min(start + batch_size, len(base))
        x = torch.as_tensor(
            feature_design(base[start:stop], extra[start:stop], head["extra_dimension"]),
            dtype=torch.float64,
        )
        x.sub_(torch.as_tensor(head["mean"])).div_(torch.as_tensor(head["scale"]))
        x.clamp_(-head["clip"], head["clip"]).sub_(torch.as_tensor(head["center"]))
        result[start:stop] = (
            (x @ torch.as_tensor(head["weight"])) * torch.as_tensor(head["target_scale"])
            + torch.as_tensor(head["target_mean"])
        ).numpy()
    return result
