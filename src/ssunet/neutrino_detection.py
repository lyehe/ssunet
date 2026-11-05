"""Finite Poisson injection/recovery statistics; no event probability is inferred."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PhotonWindows:
    """Ragged observed photons, without injection labels or latent positions."""

    offsets: np.ndarray
    locations: np.ndarray
    retained: np.ndarray
    randomizers: np.ndarray

    def __len__(self):
        """Return the number of acquisition windows, including empty ones."""
        return len(self.offsets) - 1

    def halves(self, index):
        """Return the two recorded halves without adding or dropping photons."""
        selected = slice(self.offsets[index], self.offsets[index + 1])
        locations, retained = self.locations[selected], self.retained[selected]
        return locations[retained], locations[~retained]

    def dense_batch(self, start, stop, shape):
        """Materialize only the requested dense batch with integer multiplicity."""
        a = np.zeros((stop - start, *shape), dtype=np.int64)
        b = np.zeros_like(a)
        for row, index in enumerate(range(start, stop)):
            left, right = self.halves(index)
            np.add.at(a[row].ravel(), left, 1)
            np.add.at(b[row].ravel(), right, 1)
        return a, b


def sample_windows(pdfs, signals, background, support_size, rng, *, p=0.5):
    """Draw independent signal/background Poisson detections on fixed support.

    ``pdfs`` yields one normalized signal PDF per requested acquisition window;
    a background-only window may use None. Latent information stays outside the
    returned observation object. Thinning is applied to complete observed photons.
    """
    if not np.isfinite(background) or background <= 0 or not 0 < p < 1:
        message = "Use positive finite background and a split probability in (0,1)"
        raise ValueError(message)
    signals = np.asarray(signals, dtype=float)
    if np.any(signals < 0) or not np.isfinite(signals).all() or support_size < 1:
        message = "Signal means must be finite and nonnegative on nonempty support"
        raise ValueError(message)
    offsets, locations, retained = [0], [], []
    for signal, pdf in zip(signals, pdfs, strict=True):
        noise = rng.integers(support_size, size=rng.poisson(background))
        if signal > 0:
            pdf = np.asarray(pdf).ravel()
            if len(pdf) != support_size or not np.isfinite(pdf).all() or np.any(pdf < 0):
                message = "Signal PDF must be finite and nonnegative on the whole support"
                raise ValueError(message)
            if not np.isclose(pdf.sum(), 1, atol=1e-10):
                message = "Signal PDF must be normalized"
                raise ValueError(message)
            # Inverse CDF sampling avoids rebuilding a large multinomial table.
            cumulative = np.cumsum(pdf)
            cumulative[-1] = 1.0
            detected = np.searchsorted(cumulative, rng.random(rng.poisson(signal)))
        else:
            detected = np.empty(0, dtype=np.int64)
        observed = np.concatenate([noise, detected]).astype(np.int32)
        locations.append(observed)
        retained.append(rng.random(len(observed)) < p)
        offsets.append(offsets[-1] + len(observed))
    return PhotonWindows(
        np.asarray(offsets, dtype=np.int64),
        np.concatenate(locations) if locations else np.empty(0, dtype=np.int32),
        np.concatenate(retained) if retained else np.empty(0, dtype=bool),
        rng.random(len(signals)),
    )


def count_glr(totals, background):
    """Profile a nonnegative extra Poisson mean against a known background mean."""
    totals = np.asarray(totals, dtype=float)
    if (
        background <= 0
        or not np.isfinite(background)
        or np.any(totals < 0)
        or not np.isfinite(totals).all()
    ):
        message = "Poisson GLR requires positive finite background and nonnegative counts"
        raise ValueError(message)
    ratio = np.maximum(totals / background, 1)
    return np.maximum(totals * np.log(ratio) - np.maximum(totals - background, 0), 0)


def cross_shape_score(left, right, q_left_at_right, q_right_at_left, support_size):
    """Average two independent-half shape scores, retaining photon multiplicity."""
    if len(q_left_at_right) != len(right) or len(q_right_at_left) != len(left):
        message = "Every observed photon must receive exactly one cross-prediction score"
        raise ValueError(message)
    forward = np.log(np.asarray(q_left_at_right) * support_size).sum()
    reverse = np.log(np.asarray(q_right_at_left) * support_size).sum()
    if not np.isfinite(forward + reverse):
        message = "Detection requires strictly positive normalized predictive probabilities"
        raise ValueError(message)
    return float(0.5 * (forward + reverse))


def normal_kernel(coordinates, width):
    """Normalize physical Gaussian kernels per source location, including identity."""
    coordinates = np.asarray(coordinates, dtype=float)
    if width == 0:
        return np.eye(len(coordinates))
    if not np.isfinite(width) or width < 0:
        message = "Kernel width must be finite and nonnegative"
        raise ValueError(message)
    squared = ((coordinates[:, None] - coordinates[None, :]) ** 2).sum(-1)
    weights = np.exp(-0.5 * squared / width**2)
    return weights / weights.sum(1, keepdims=True)


def kernel_at_photons(observed, queried, prior, sensor_kernel, time_kernel, fraction):
    """Evaluate a normalized geometry/time estimate only at queried photons."""
    if not 0 < fraction <= 1:
        message = "A positive prior fraction is required"
        raise ValueError(message)
    fallback = prior.ravel()[queried]
    if len(observed) == 0:
        return fallback
    bins = prior.shape[1]
    sensor, time = np.divmod(observed, bins)
    target_sensor, target_time = np.divmod(queried, bins)
    density = sensor_kernel[sensor[:, None], target_sensor[None, :]]
    density *= time_kernel[time[:, None], target_time[None, :]]
    return (1 - fraction) * density.mean(0) + fraction * fallback


def template_glr(locations, templates, signal_grid, background):
    """Search fixed source templates/amplitudes without seeing injection truth."""
    signal_grid = np.asarray(signal_grid, dtype=float)
    if background <= 0 or np.any(signal_grid <= 0) or not np.isfinite(signal_grid).all():
        message = "Template amplitudes and background must be positive and finite"
        raise ValueError(message)
    ratio = templates[:, locations] * templates.shape[1] / background
    values = np.log1p(signal_grid[:, None, None] * ratio[None]).sum(-1)
    return float(max(0, np.max(values - signal_grid[:, None])))


def calibrated_tail_probabilities(calibration, scores, randomizers):
    """Compute smoothed empirical upper ranks, handling discrete ties fairly."""
    calibration = np.asarray(calibration, dtype=float)
    scores, randomizers = np.asarray(scores), np.asarray(randomizers)
    if calibration.ndim != 1 or len(calibration) == 0 or scores.shape != randomizers.shape:
        message = "Calibration must be nonempty and scores/randomizers must match"
        raise ValueError(message)
    if not np.isfinite(calibration).all() or not np.isfinite(scores).all():
        message = "Calibration and test scores must be finite"
        raise ValueError(message)
    if not np.isfinite(randomizers).all() or np.any((randomizers < 0) | (randomizers >= 1)):
        message = "Tie randomizers must be in [0,1)"
        raise ValueError(message)
    ordered = np.sort(calibration)
    left = np.searchsorted(ordered, scores, side="left")
    right = np.searchsorted(ordered, scores, side="right")
    return (len(ordered) - right + randomizers * (right - left + 1)) / (len(ordered) + 1)


def wilson_interval(successes, trials):
    """Return a descriptive two-sided 95% Wilson binomial interval."""
    if trials <= 0 or not 0 <= successes <= trials:
        message = "Binomial intervals require positive trials and valid successes"
        raise ValueError(message)
    z = 1.959963984540054
    rate = successes / trials
    denominator = 1 + z**2 / trials
    center = (rate + z**2 / (2 * trials)) / denominator
    width = z * np.sqrt(rate * (1 - rate) / trials + z**2 / (4 * trials**2)) / denominator
    return [float(max(0, center - width)), float(min(1, center + width))]


def summarize_detection(probabilities, truth, config):
    """Report false alarms and paired source-group efficiency intervals."""
    levels = np.asarray(config["generator"]["signal_expected_photons"], dtype=float)
    groups = np.asarray(truth["source_group"])
    signal = np.asarray(truth["expected_signal"])
    background = signal == 0
    unique = np.unique(groups[~background])
    draws = config["evaluation"]["bootstrap_draws"]
    rng = np.random.default_rng(config["evaluation"]["bootstrap_seed"])
    # Reuse source draws across levels/methods and keep calibration fixed.
    weights = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)), size=draws)
    result = {}
    for alpha in config["evaluation"]["false_positive_rates"]:
        detected = {method: values <= alpha for method, values in probabilities.items()}
        methods = {}
        for method, decision in detected.items():
            false_positives = int(decision[background].sum())
            entry = {
                "false_positives": false_positives,
                "background_windows": int(background.sum()),
                "false_positive_rate": float(decision[background].mean()),
                "false_positive_interval": wilson_interval(false_positives, int(background.sum())),
                "efficiency": [],
            }
            for level in levels:
                selected = signal == level
                group_means = np.asarray(
                    [decision[selected & (groups == group)].mean() for group in unique]
                )
                boot = weights @ group_means / len(unique)
                row = {
                    "expected_signal": float(level),
                    "windows": int(selected.sum()),
                    "efficiency": float(decision[selected].mean()),
                    "interval": np.quantile(boot, [0.025, 0.975]).tolist(),
                }
                if method != "self_split":
                    other = detected["self_split"]
                    difference = other.astype(float) - decision
                    delta_groups = np.asarray(
                        [difference[selected & (groups == group)].mean() for group in unique]
                    )
                    boot_delta = weights @ delta_groups / len(unique)
                    row.update(
                        model_minus_control=float(difference[selected].mean()),
                        difference_interval=np.quantile(boot_delta, [0.025, 0.975]).tolist(),
                        control_missed_model_found=int(
                            (~decision[selected] & other[selected]).sum()
                        ),
                        control_found_model_missed=int(
                            (decision[selected] & ~other[selected]).sum()
                        ),
                    )
                entry["efficiency"].append(row)
            methods[method] = entry
        result[str(alpha)] = methods
    return result
