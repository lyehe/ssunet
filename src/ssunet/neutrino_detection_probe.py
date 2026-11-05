"""Observation-only detector features and separately supervised frozen-encoder heads."""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as functional

from .neutrino_detection import count_glr, cross_shape_score, kernel_at_photons, template_glr

HEAD_ARMS = ("raw_prior", "self_split", "untrained")
BASE_SCALARS = (
    "count",
    "log_count",
    "empty",
    "occupied_sensors",
    "occupied_times",
    "count_glr",
    "prior_log_ratio_sum",
    "prior_log_ratio_mean",
    "prior_log_ratio_max",
    "prior_profile_glr",
    "kernel_cross_shape",
    "same_sensor_pairs",
    "same_time_pairs",
    "mean_time",
    "std_time",
    "centroid_x",
    "centroid_y",
    "centroid_z",
)


def observation_features(windows, start, stop, a, b, reference):
    """Summarize full recorded counts and fixed references without receiving labels."""
    counts = a + b
    total = counts.sum((1, 2))
    sensor, temporal = counts.sum(2), counts.sum(1)
    coordinate = np.linspace(-1, 1, counts.shape[2])
    first = (counts * coordinate).sum(2) / np.maximum(sensor, 1)
    prior, background = reference["prior"], reference["background"]
    count_score = count_glr(total, background)
    scalars = np.zeros((len(counts), len(BASE_SCALARS)), dtype=np.float64)
    controls = {name: count_score.copy() for name in ("count", "static_prior", "geometry_kernel")}
    for row, index in enumerate(range(start, stop)):
        left, right = windows.halves(index)
        all_photons = np.concatenate([left, right])
        log_ratio = np.log(prior.ravel()[all_photons] * prior.size)
        densities = [
            kernel_at_photons(
                observed,
                queried,
                prior,
                reference["sensor_kernel"],
                reference["time_kernel"],
                reference["mixture_fraction"],
            )
            for observed, queried in ((left, right), (right, left))
        ]
        kernel_shape = cross_shape_score(left, right, *densities, prior.size)
        controls["static_prior"][row] += 0.5 * log_ratio.sum()
        controls["geometry_kernel"][row] += kernel_shape
        mean = float(temporal[row] @ coordinate / max(total[row], 1))
        second = float(temporal[row] @ coordinate**2 / max(total[row], 1))
        centroid = sensor[row] @ reference["positions"] / max(total[row], 1) / 500
        scalars[row] = [
            total[row],
            np.log1p(total[row]),
            total[row] == 0,
            (sensor[row] > 0).sum(),
            (temporal[row] > 0).sum(),
            count_score[row],
            log_ratio.sum(),
            log_ratio.sum() / max(total[row], 1),
            log_ratio.max() if len(log_ratio) else 0,
            template_glr(all_photons, prior.reshape(1, -1), reference["amplitudes"], background),
            kernel_shape,
            (sensor[row] * (sensor[row] - 1)).sum() / 2,
            (temporal[row] * (temporal[row] - 1)).sum() / 2,
            mean,
            np.sqrt(max(second - mean**2, 0)),
            *centroid,
        ]
    features = np.concatenate([np.log1p(sensor), np.log1p(temporal), first, scalars], axis=1)
    return features.astype(np.float32), controls


def pdf_summary(probability, sensor_groups, time_groups):
    """Compress a normalized PDF on fixed groups, never groups chosen from detections."""
    batch, sensors, bins = probability.shape
    if sensors % sensor_groups or bins % time_groups:
        message = "Feature groups must divide the fixed sensor/time support"
        raise ValueError(message)
    grouped = probability.reshape(batch, sensor_groups, sensors // sensor_groups, bins).sum(2)
    sensor = grouped.sum(2)
    temporal = probability.reshape(batch, sensors, time_groups, bins // time_groups).sum((1, 3))
    moments = (grouped * np.linspace(-1, 1, bins)).sum(2) / np.maximum(sensor, 1e-300)
    return np.concatenate([sensor, temporal, moments], axis=1)


def encoder_features(windows, start, stop, qa, qb, ha, hb, prior, settings):
    """Combine symmetric predicted PDFs/hidden features and observed cross scores."""
    left_summary = np.concatenate(
        [pdf_summary(qa, settings["sensor_groups"], settings["time_groups"]), ha], axis=1
    )
    right_summary = np.concatenate(
        [pdf_summary(qb, settings["sensor_groups"], settings["time_groups"]), hb], axis=1
    )
    scalars = np.zeros((stop - start, 7), dtype=np.float64)
    cross_scores = np.zeros(stop - start)
    for row, index in enumerate(range(start, stop)):
        left, right = windows.halves(index)
        cross_scores[row] = cross_shape_score(
            left, right, qa[row].ravel()[right], qb[row].ravel()[left], prior.size
        )
        pa, pb = qa[row], qb[row]
        average = 0.5 * (pa + pb)
        loga, logb, logaverage = np.log(pa), np.log(pb), np.log(average)
        entropy = -0.5 * ((pa * loga).sum() + (pb * logb).sum())
        js = 0.5 * ((pa * (loga - logaverage)).sum() + (pb * (logb - logaverage)).sum())
        prior_kl = 0.5 * ((pa * (loga - np.log(prior))).sum() + (pb * (logb - np.log(prior))).sum())
        scalars[row] = [
            cross_scores[row],
            cross_scores[row] / max(len(left) + len(right), 1),
            entropy,
            js,
            np.abs(pa - pb).sum(),
            0.5 * (pa.max() + pb.max()),
            prior_kl,
        ]
    values = np.concatenate(
        [0.5 * (left_summary + right_summary), np.abs(left_summary - right_summary), scalars],
        axis=1,
    )
    if not np.isfinite(values).all():
        message = "Encoder features must remain finite on every retained window"
        raise ValueError(message)
    return values.astype(np.float32), cross_scores


@torch.inference_mode()
def encoder_features_tensor(a, b, qa, qb, ha, hb, prior, settings):
    """Reduce PDFs on their device, equivalent to the sparse NumPy reference above."""
    sensor_groups, time_groups = settings["sensor_groups"], settings["time_groups"]
    batch, sensors, bins = qa.shape
    if sensors % sensor_groups or bins % time_groups:
        message = "Feature groups must divide the fixed sensor/time support"
        raise ValueError(message)

    def summary(probability):
        grouped = probability.reshape(batch, sensor_groups, sensors // sensor_groups, bins).sum(2)
        sensor = grouped.sum(2)
        temporal = probability.reshape(batch, sensors, time_groups, bins // time_groups).sum((1, 3))
        time_axis = torch.linspace(-1, 1, bins, device=qa.device, dtype=qa.dtype)
        moment = (grouped * time_axis).sum(2) / sensor.clamp_min(1e-300)
        return torch.cat([sensor, temporal, moment], dim=1)

    left = torch.cat([summary(qa), ha], dim=1)
    right = torch.cat([summary(qb), hb], dim=1)
    a = torch.as_tensor(a, device=qa.device)
    b = torch.as_tensor(b, device=qa.device)
    loga, logb = qa.log(), qb.log()
    log_support = np.log(sensors * bins)
    cross = 0.5 * ((b * (loga + log_support)).sum((1, 2)) + (a * (logb + log_support)).sum((1, 2)))
    total = (a + b).sum((1, 2)).clamp_min(1)
    average = 0.5 * (qa + qb)
    entropy = -0.5 * ((qa * loga).sum((1, 2)) + (qb * logb).sum((1, 2)))
    js = -(average * average.log()).sum((1, 2)) - entropy
    logprior = torch.as_tensor(prior, device=qa.device, dtype=qa.dtype).log()
    prior_kl = -entropy - (average * logprior).sum((1, 2))
    scalars = torch.stack(
        [
            cross,
            cross / total,
            entropy,
            js,
            (qa - qb).abs().sum((1, 2)),
            0.5 * (qa.amax((1, 2)) + qb.amax((1, 2))),
            prior_kl,
        ],
        dim=1,
    )
    features = torch.cat([0.5 * (left + right), (left - right).abs(), scalars], dim=1)
    return features.float().cpu().numpy(), cross.cpu().numpy()


def feature_design(base, extra, extra_dimension):
    """Keep one head layout, including zero padding for the raw/prior arm."""
    if extra is None:
        extra = np.zeros((len(base), extra_dimension), dtype=base.dtype)
    if extra.shape != (len(base), extra_dimension):
        message = "All heads require the same extra-feature dimension and row identity"
        raise ValueError(message)
    result = np.concatenate([base, extra], axis=1)
    if not np.isfinite(result).all():
        message = "Classifier features must be finite"
        raise ValueError(message)
    return result


def balanced_log_loss(scores, labels):
    """Weight event presence/absence equally, independently of an artificial class mix."""
    labels = np.asarray(labels)
    if not np.isin(labels, [0, 1]).all() or not (labels == 0).any() or not (labels == 1).any():
        message = "Binary training/validation labels must include both event classes"
        raise ValueError(message)
    loss = np.logaddexp(0, scores) - labels * scores
    return float(0.5 * (loss[labels == 0].mean() + loss[labels == 1].mean()))


def predict_head(head, base, extra, *, batch_size=1024):
    """Apply training-only scaling and a frozen linear score in bounded batches."""
    values = np.empty(len(base), dtype=np.float64)
    weight = torch.as_tensor(head["weight"], dtype=torch.float64)
    mean = torch.as_tensor(head["mean"], dtype=torch.float64)
    scale = torch.as_tensor(head["scale"], dtype=torch.float64)
    for start in range(0, len(base), batch_size):
        stop = min(start + batch_size, len(base))
        matrix = feature_design(
            base[start:stop], None if extra is None else extra[start:stop], head["extra_dimension"]
        )
        matrix = torch.as_tensor(matrix, dtype=torch.float64)
        matrix.sub_(mean).div_(scale).clamp_(-head["clip"], head["clip"])
        values[start:stop] = (matrix @ weight + head["intercept"]).numpy()
    return values


def fit_head_candidates(train, labels, validation, validation_labels, arm, config, guard):
    """Fit the fixed penalty grid and select only by balanced validation log loss."""
    settings = config["head"]
    extra_dimension = train["self_split"].shape[1]
    extra = None if arm == "raw_prior" else train[arm]
    matrix = feature_design(train["base"], extra, extra_dimension)
    x = torch.as_tensor(matrix, dtype=torch.float64)
    mean, scale = x.mean(0), x.std(0, unbiased=False).clamp_min(settings["feature_scale_floor"])
    x.sub_(mean).div_(scale).clamp_(-settings["feature_clip"], settings["feature_clip"])
    y = torch.as_tensor(labels, dtype=torch.float64)
    balanced_log_loss(np.zeros(len(labels)), labels)  # Validate both classes before optimization.
    weights = torch.where(y > 0.5, 0.5 / (y > 0.5).sum(), 0.5 / (y < 0.5).sum())
    records, heads = [], []
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
        head = {
            "mean": mean.numpy().copy(),
            "scale": scale.numpy().copy(),
            "weight": weight.detach().numpy().copy(),
            "intercept": float(intercept.detach()),
            "penalty": float(penalty),
            "clip": settings["feature_clip"],
            "extra_dimension": int(extra_dimension),
        }
        values = predict_head(
            head,
            validation["base"],
            None if arm == "raw_prior" else validation[arm],
            batch_size=config["runtime"]["head_batch_size"],
        )
        gradient = float(max(weight.grad.abs().max(), intercept.grad.abs()))
        iterations = optimizer.state[weight]["n_iter"]
        records.append(
            {
                "penalty": float(penalty),
                "validation_balanced_log_loss": balanced_log_loss(values, validation_labels),
                "training_balanced_log_loss": balanced_log_loss(
                    (x @ weight + intercept).detach().numpy(), labels
                ),
                "penalized_training_loss": final_loss,
                "max_absolute_gradient": gradient,
                "iterations": int(iterations),
                "function_evaluations": int(optimizer.state[weight]["func_evals"]),
                "iteration_budget_exhausted": bool(iterations >= settings["max_iterations"]),
                "seconds": time.perf_counter() - started,
            }
        )
        heads.append(head)
    selected = int(np.argmin([row["validation_balanced_log_loss"] for row in records]))
    if (
        not np.isfinite(records[selected]["penalized_training_loss"])
        or not np.isfinite(records[selected]["max_absolute_gradient"])
        or records[selected]["max_absolute_gradient"] > 1e-4
    ):
        message = "Selected detector head failed the finite development convergence gate"
        raise RuntimeError(message)
    return heads[selected], {"arm": arm, "selected_candidate": selected, "candidates": records}
