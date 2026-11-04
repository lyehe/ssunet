"""Tune classical photon estimators within training data, then score the frozen pilot."""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import yaml

if __package__ == "src.tools":
    from ..ssunet.datasets.neutrino import NeutrinoPhotonDataset, shard_sha256
    from ..ssunet.neutrino_classical import fit_population_prior, kernel_counts, prior_shrinkage
    from ..ssunet.neutrino_evaluation import (
        FrozenPhotonView,
        PhotonScoreRecords,
        freeze_photon_view,
        paired_photon_interval,
        photon_score_summary,
        probability_nll,
    )
    from ..ssunet.neutrino_training import _claim_run
    from ..ssunet.poisson_tv import poisson_tv
else:
    from ssunet.datasets.neutrino import NeutrinoPhotonDataset, shard_sha256
    from ssunet.neutrino_classical import fit_population_prior, kernel_counts, prior_shrinkage
    from ssunet.neutrino_evaluation import (
        FrozenPhotonView,
        PhotonScoreRecords,
        freeze_photon_view,
        paired_photon_interval,
        photon_score_summary,
        probability_nll,
    )
    from ssunet.neutrino_training import _claim_run
    from ssunet.poisson_tv import poisson_tv

LOGGER = logging.getLogger(__name__)


def write_json(path, data):
    """Atomically replace only files owned by this comparison."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def synchronize(device):
    """Include CUDA work in reported classical estimation times."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def inner_group_split(configuration_ids, fraction, seed):
    """Make a deterministic inner fit/tuning split without sharing configurations."""
    groups = np.unique(configuration_ids)
    if len(groups) < 2 or not 0 < fraction < 1:
        message = "Inner classical tuning needs at least two training configurations"
        raise ValueError(message)
    order = np.random.default_rng(seed).permutation(groups)
    size = min(len(order) - 1, max(1, int(np.ceil(len(order) * fraction))))
    tuning = np.isin(configuration_ids, order[:size])
    return np.flatnonzero(~tuning), np.flatnonzero(tuning)


def subset_view(view, indices):
    """Keep shared instrument metadata while selecting only declared training events."""
    return FrozenPhotonView(
        inputs=view.inputs[indices],
        targets=view.targets[indices],
        positions_m=view.positions_m,
        live=view.live,
        p=view.p,
        event_ids=view.event_ids[indices],
        configuration_ids=view.configuration_ids[indices],
    )


def mean_event_cross_entropy(losses, totals):
    """Give each nonempty target one vote, aligned with GAP target normalization."""
    active = totals > 0
    if not active.any():
        message = "The inner tuning partition has no held-out photons"
        raise ValueError(message)
    return float(np.mean(losses[active] / totals[active]))


def density_candidates(config, family):
    """Enumerate the complete declared search grid before outer validation access."""
    if family == "dirichlet":
        yield {"kind": "raw"}
    elif family in {"temporal_kde", "spatiotemporal_kde"}:
        spatial = [0] if family == "temporal_kde" else config["grid"]["space_bandwidth_m"]
        for space, temporal, adaptive in itertools.product(
            spatial, config["grid"]["time_bandwidth_ns"], [False, True]
        ):
            if adaptive and space == 0 and temporal == 0:
                continue
            yield {
                "kind": "kde",
                "space_bandwidth_m": space,
                "time_bandwidth_ns": temporal,
                "adaptive": adaptive,
            }
    elif family == "poisson_tv":
        yield {"kind": "poisson_tv", "regularization": 0, "vertical_strength": 0}
        for beta, vertical in itertools.product(
            config["grid"]["tv_regularization"], config["grid"]["tv_vertical_strength"]
        ):
            if beta > 0:
                yield {"kind": "poisson_tv", "regularization": beta, "vertical_strength": vertical}
    else:
        message = "Unknown declared classical estimator family"
        raise ValueError(message)


def estimate_density(counts, positions, times, live, settings, solver):
    """Estimate a density shape from A alone, returning numerical solver diagnostics."""
    kind = settings["kind"]
    options = {
        key: value for key, value in settings.items() if key not in {"kind", "concentration"}
    }
    if kind == "raw":
        return counts.double().masked_fill(~live[None, :, None], 0), {"converged": True}
    if kind == "kde":
        return kernel_counts(counts, positions, times, live, **options), {"converged": True}
    return poisson_tv(counts, positions, live, **options, **solver)


def tune_on_training(config, training_view, times_ns, root, device):
    """Fit population priors and select parameters without opening outer validation photons."""
    split = config["inner_split"]
    fit_ids, tune_ids = inner_group_split(
        training_view.configuration_ids, split["tuning_fraction"], split["seed"]
    )
    tuning = subset_view(training_view, tune_ids)
    positions = training_view.positions_m.to(device)
    live = training_view.live.to(device)
    times = times_ns.to(device)
    full_counts = training_view.inputs + training_view.targets
    fitted = full_counts[fit_ids].to(device)
    inputs, targets = tuning.inputs.to(device), tuning.targets.to(device)
    manifest = {
        "fit_event_ids": training_view.event_ids[fit_ids].tolist(),
        "tuning_event_ids": tuning.event_ids.tolist(),
        "fit_configuration_ids": training_view.configuration_ids[fit_ids].tolist(),
        "tuning_configuration_ids": tuning.configuration_ids.tolist(),
        "selection_objective": "mean cross entropy over all nonempty inner-training targets",
        "outer_validation_photons_opened": False,
    }
    write_json(root / "inner_split.json", manifest)
    trials, prior_candidates = [], []
    for bandwidth, weighting in itertools.product(
        config["grid"]["prior_time_bandwidth_ns"], config["grid"]["prior_weighting"]
    ):
        settings = {"time_bandwidth_ns": bandwidth, "weighting": weighting}
        prior = fit_population_prior(
            fitted, positions, times, live, **settings, pseudocount=config["prior_pseudocount"]
        )
        losses = probability_nll(prior.expand_as(inputs), targets)
        score = mean_event_cross_entropy(losses, tuning.target_totals)
        prior_candidates.append((score, settings))
        trials.append({"family": "population_prior", "settings": settings, "inner_event_ce": score})
    _, prior_settings = min(prior_candidates, key=lambda item: item[0])
    prior = fit_population_prior(
        fitted, positions, times, live, **prior_settings, pseudocount=config["prior_pseudocount"]
    )
    selected = {}
    for family in config["families"]:
        selected[family] = _tune_family(
            config, family, inputs, targets, positions, times, live, prior, tuning, trials, root
        )
    locked = {
        "prior": prior_settings,
        "estimators": selected,
        "fit_events": len(fit_ids),
        "tuning_events": len(tune_ids),
        "selection_objective": manifest["selection_objective"],
        "outer_validation_photons_opened_during_selection": False,
        "tuning_trials": len(trials),
    }
    write_json(root / "tuning_trials.json", trials)
    return locked


def _tune_family(
    config, family, inputs, targets, positions, times, live, prior, tuning, trials, root
):
    """Evaluate each predeclared density and concentration against inner B only."""
    best = None
    LOGGER.info("Selecting %s using training-only photon splits", family)
    for index, settings in enumerate(density_candidates(config, family)):
        density, diagnostic = estimate_density(
            inputs, positions, times, live, settings, config["tv_solver"]
        )
        if not diagnostic["converged"]:
            LOGGER.info(
                "Excluded nonconverged %s density=%s diagnostics=%s", family, index, diagnostic
            )
            trials.append(
                {"family": family, "settings": settings, "excluded_nonconverged": diagnostic}
            )
            continue
        for concentration in config["grid"]["prior_concentration"]:
            q = prior_shrinkage(density, inputs, prior, live, concentration)
            losses = probability_nll(q, targets)
            score = mean_event_cross_entropy(losses, tuning.target_totals)
            candidate = {**settings, "concentration": concentration}
            trials.append({"family": family, "settings": candidate, "inner_event_ce": score})
            if best is None or score < best["inner_event_ce"]:
                best = {"settings": candidate, "inner_event_ce": score, "solver": diagnostic}
        if index % 5 == 0:
            LOGGER.info(
                "%s density=%s best_inner_event_ce=%.4f", family, index, best["inner_event_ce"]
            )
            write_json(
                root / "progress.json",
                {"phase": "training_only_selection", "family": family, "density": index},
            )
    if best is None:
        message = "No numerically converged candidate in a declared classical family"
        raise RuntimeError(message)
    write_json(root / "tuning_trials.json", trials)
    LOGGER.info("Locked %s settings=%s", family, best["settings"])
    return best


def study_identity(config):
    """Bind the count data, classical implementations and already audited comparison."""
    source = Path(__file__).parents[1]
    paths = [Path(__file__)] + [
        source / name
        for name in (
            "ssunet/neutrino_classical.py",
            "ssunet/poisson_tv.py",
            "ssunet/datasets/neutrino.py",
            "ssunet/neutrino_evaluation.py",
            "ssunet/neutrino_training.py",
            "ssunet/distributions.py",
            "ssunet/photon_reference.py",
        )
    ]
    previous = Path(config["reference_study"])
    return {
        "config": config,
        "dataset_sha256": shard_sha256(config["dataset"]),
        "source_sha256": {str(path): shard_sha256(path) for path in paths},
        "reference_sha256": {
            name: shard_sha256(previous / name)
            for name in (
                "config.json",
                "identity.json",
                "results.json",
                "per_event_scores.npz",
                "arrival_examples.npz",
                "evidence_audit.json",
            )
        },
        "torch_version": str(torch.__version__),
    }


def predict_classical(config, view, positions, times, prior, selected, device):
    """Produce full validation PMFs with locked parameters using only A at inference."""
    inputs = view.inputs.to(device)
    live = view.live.to(device)
    probabilities, losses, diagnostics = {}, {}, {}
    for name, entry in selected.items():
        synchronize(device)
        begin = time.perf_counter()
        density, solver = estimate_density(
            inputs, positions, times, live, entry["settings"], config["tv_solver"]
        )
        if not solver["converged"]:
            refined = {**config["tv_solver"], "max_iterations": config["tv_refine_max_iterations"]}
            density, solver = estimate_density(
                inputs, positions, times, live, entry["settings"], refined
            )
        if not solver["converged"]:
            message = "Selected Poisson-TV estimator did not meet its declared numerical tolerance"
            raise RuntimeError(message)
        q = prior_shrinkage(density, inputs, prior, live, entry["settings"]["concentration"])
        synchronize(device)
        seconds = time.perf_counter() - begin
        losses[name] = probability_nll(q, view.targets.to(device))
        probabilities[name] = q.cpu().numpy()
        diagnostics[name] = {**solver, "prediction_seconds": seconds}
        LOGGER.info("Scored locked %s on %s validation observations", name, len(inputs))
    q = prior.expand_as(inputs)
    probabilities["population_prior"] = q.cpu().numpy()
    losses["population_prior"] = probability_nll(q, view.targets.to(device))
    return probabilities, losses, diagnostics


def mean_event_scores(values, validation):
    """Average normalized event scores only where a held-out target exists."""
    result = {}
    totals = validation.target_totals
    for name, mask in validation.strata.items():
        active = mask & (totals > 0)
        result[name] = float(np.mean(values[active] / totals[active])) if active.any() else None
    return result


def event_contrasts(config, validation, means, classical_names):
    """Bootstrap the same configuration groups with one unit per nonempty target."""
    totals = validation.target_totals
    records = PhotonScoreRecords(
        event_ids=validation.event_ids,
        configuration_ids=validation.configuration_ids,
        retained_totals=validation.inputs.sum((1, 2)).numpy(),
        target_totals=(totals > 0).astype(np.int64),
    )
    results = {}
    for name in classical_names:
        delta = (means["conv4d"] - means[name]) / np.maximum(totals, 1)
        results[name] = {}
        for stratum, mask in validation.strata.items():
            value = paired_photon_interval(
                delta,
                records,
                mask & (totals > 0),
                seed=config["bootstrap_seed"],
                draws=config["bootstrap_draws"],
            )
            if value is not None:
                value["model_minus_control_mean_event_cross_entropy"] = value.pop(
                    "model_minus_control_nats_per_photon"
                )
            results[name][stratum] = value
    return results


def summarize(config, validation, classical_scores, selection):
    """Compare deterministic classical predictors with the frozen neural evidence."""
    with np.load(Path(config["reference_study"]) / "per_event_scores.npz") as saved:
        scores = {
            name: saved[name].copy()
            for name in (
                "conv4d",
                "published_gap_matched",
                "published_gap_paper",
                "full_time_reference",
                "previously_tuned_mixture",
            )
        }
        expected = {
            "event_ids": validation.event_ids,
            "configuration_ids": validation.configuration_ids,
            "target_totals": validation.target_totals,
            "retained_totals": validation.inputs.sum((1, 2)).numpy(),
        }
        if any(not np.array_equal(saved[key], value) for key, value in expected.items()):
            message = (
                "Classical evaluation observations or splits differ from the neural comparator"
            )
            raise ValueError(message)
    scores.update(classical_scores)
    means = {
        name: values.mean(0) if values.ndim == 2 else values for name, values in scores.items()
    }
    differences = {
        name: {
            stratum: paired_photon_interval(
                means["conv4d"] - value,
                validation,
                mask,
                seed=config["bootstrap_seed"],
                draws=config["bootstrap_draws"],
            )
            for stratum, mask in validation.strata.items()
        }
        for name, value in means.items()
        if name in classical_scores
    }
    report = {
        "scope": config["scope"],
        "selection": selection,
        "selection_scope": "inner training groups; parameters locked before outer validation",
        "test_or_truth_used": False,
        "scores": {name: photon_score_summary(value, validation) for name, value in means.items()},
        "event_mean_nll": {
            name: mean_event_scores(value, validation) for name, value in means.items()
        },
        "conv4d_minus_classical": differences,
        "event_mean_contrasts": event_contrasts(config, validation, means, classical_scores),
        "classical_estimators_deterministic": True,
        "neural_seed_aggregation": "mean event NLL, not prediction ensembling or extra events",
        "limitations": config["limitations"],
    }
    return report, scores


def run_study(config, device_name="cuda:0", *, resume=False):
    """Lock training-only selection before accessing any outer validation photon arrays."""
    if config["observation_kind"] != "photon_counts":
        message = "This classical likelihood comparison requires photon-count observations"
        raise ValueError(message)
    root = Path(config["output"])
    identity = study_identity(config)
    if identity["dataset_sha256"] != config["dataset_sha256"]:
        message = "Classical comparison dataset differs from its declared source"
        raise ValueError(message)
    if root.exists():
        if not resume or json.loads((root / "identity.json").read_text()) != identity:
            message = "Use --resume only with identical classical data, configuration and source"
            raise ValueError(message)
    else:
        root.mkdir(parents=True)
        write_json(root / "identity.json", identity)
        write_json(root / "config.json", config)
    _claim_run(root)
    torch.set_num_threads(config["cpu_threads"])
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    options = {"lazy": True, "seed": config["photon_split_seed"], "p": config["split_probability"]}
    dataset = NeutrinoPhotonDataset(config["dataset"], "train", training=False, **options)
    training = freeze_photon_view(dataset)
    times = torch.as_tensor((dataset.edges_ns[:-1] + dataset.edges_ns[1:]) / 2)
    dataset.close()
    started = time.perf_counter()
    lock = root / "selection_locked.json"
    if lock.exists():
        selection = json.loads(lock.read_text())
    else:
        with torch.inference_mode():
            selection = tune_on_training(config, training, times, root, device)
        write_json(lock, selection)
    selection_sha = shard_sha256(lock)
    synchronize(device)
    selection_seconds = time.perf_counter() - started
    positions = training.positions_m.to(device)
    live = training.live.to(device)
    with torch.inference_mode():
        prior = fit_population_prior(
            (training.inputs + training.targets).to(device),
            positions,
            times.to(device),
            live,
            **selection["prior"],
            pseudocount=config["prior_pseudocount"],
        )
    write_json(
        root / "validation_access.json",
        {
            "selection_sha256": selection_sha,
            "selection_locked_before_validation_open": lock.exists(),
            "stage": "opening validation only after training selection is locked",
        },
    )
    validation_data = NeutrinoPhotonDataset(config["dataset"], "validation", **options)
    validation = freeze_photon_view(validation_data)
    validation_data.close()
    with torch.inference_mode():
        probabilities, losses, diagnostics = predict_classical(
            config,
            validation,
            positions,
            times.to(device),
            prior,
            selection["estimators"],
            device,
        )
    report, all_scores = summarize(config, validation, losses, selection)
    report["selection_sha256"] = selection_sha
    report["training_only_selection_seconds"] = selection_seconds
    report["numerical_diagnostics"] = diagnostics
    report["peak_gpu_allocated_gib"] = (
        torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
    )
    write_json(root / "results.json", report)
    np.savez_compressed(
        root / "per_event_scores.npz",
        event_ids=validation.event_ids,
        configuration_ids=validation.configuration_ids,
        retained_totals=validation.inputs.sum((1, 2)).numpy(),
        target_totals=validation.target_totals,
        **all_scores,
    )
    np.savez_compressed(
        root / "predictions.npz",
        event_ids=validation.event_ids,
        inputs=validation.inputs.numpy(),
        held_out=validation.targets.numpy(),
        prior=prior.cpu().numpy(),
        **probabilities,
    )
    write_json(root / "process.json", {"status": "complete", "selection_locked": True})
    return report


def main():
    """Run the declared classical-count benchmark or resume locked evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/neutrino_classical.yml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run_study(yaml.safe_load(Path(args.config).read_text()), args.device, resume=args.resume)


if __name__ == "__main__":
    main()
