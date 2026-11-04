"""Export pilot comparisons, event predictions and explicitly labelled truth views."""

from __future__ import annotations

import argparse
import base64
import html
import json
import shutil
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

if __package__ == "src.tools":
    from ..ssunet.datasets.neutrino import NeutrinoPhotonDataset, shard_sha256
    from ..ssunet.models.neutrino_dense import DensePhotonReference
    from ..ssunet.neutrino_evaluation import freeze_photon_view
    from ..ssunet.photon_reference import photon_count_log_probabilities
else:
    from ssunet.datasets.neutrino import NeutrinoPhotonDataset, shard_sha256
    from ssunet.models.neutrino_dense import DensePhotonReference
    from ssunet.neutrino_evaluation import freeze_photon_view
    from ssunet.photon_reference import photon_count_log_probabilities

COLORS = {"model": "#c96c27", "fixed": "#64748b", "tuned": "#167d8d", "target": "#208354"}


def _save(figure, output, name):
    """Export both shareable raster and vector versions."""
    for extension in ("png", "svg"):
        figure.savefig(output / f"{name}.{extension}", dpi=180, bbox_inches="tight")
    plt.close(figure)


def _comparisons(summary, short, baseline, output):
    """Compare photon-weighted scores without treating seed SD as an event CI."""
    labels = [
        "Short model: 110 steps",
        "Fixed 50/50 mixture",
        "Previously tuned mixture",
        "Long model: 5,000 steps",
    ]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for axis, stratum, title in zip(
        axes,
        ("all", "retained_1_to_10"),
        ("All validation photons", "1 to 10 retained photons"),
        strict=True,
    ):
        final = summary["across_seeds"]["final"][stratum]
        values = [
            short["strata"][stratum]["nats_per_photon"]["model"],
            baseline["fixed_mixture"][stratum]["nats_per_photon"],
            baseline["previously_tuned_mixture"][stratum]["nats_per_photon"],
            final["mean_seed_nats_per_photon"],
        ]
        axis.barh(
            np.arange(4),
            values,
            color=["#d9a477", COLORS["fixed"], COLORS["tuned"], COLORS["model"]],
        )
        axis.errorbar(
            values[-1],
            3,
            xerr=final["sample_std_across_seeds"],
            fmt="none",
            color="black",
            capsize=4,
        )
        axis.set_yticks(np.arange(4), labels)
        axis.invert_yaxis()
        axis.set_xlim(0, 6.8)
        axis.set_title(title)
        axis.set_xlabel("Held-out photon NLL (nats/photon; lower is better)")
        axis.grid(axis="x", alpha=0.2)
        axis.set_axisbelow(True)
        for index, value in enumerate(values):
            axis.text(value + 0.09, index, f"{value:.3f}", va="center")
    figure.suptitle(
        "Longer training improves low-count prediction; the tuned mixture still leads overall"
    )
    figure.supxlabel(
        "Validation only. Long-model bars: mean of three seed NLLs; "
        "error bar: seed SD, not an event confidence interval.",
        fontsize=9,
    )
    _save(figure, output, "score_comparison")

    rows = [
        ("Low count vs fixed mixture", "retained_1_to_10", "fixed_mixture"),
        ("Low count vs tuned mixture", "retained_1_to_10", "previously_tuned_mixture"),
        ("All photons vs fixed mixture", "all", "fixed_mixture"),
        ("All photons vs tuned mixture", "all", "previously_tuned_mixture"),
    ]
    figure, axis = plt.subplots(figsize=(9, 4.0), layout="constrained")
    for index, (_label, stratum, control) in enumerate(rows):
        result = summary["across_seeds"]["final"][stratum]["mean_seed_model_minus_control"][control]
        value = result["model_minus_control_nats_per_photon"]
        lower, upper = result["descriptive_95_percent_interval"]
        axis.errorbar(
            value,
            index,
            xerr=[[value - lower], [upper - value]],
            fmt="o",
            capsize=5,
            color=COLORS["model"] if index < 2 else COLORS["fixed"],
        )
        axis.text(
            0.72, index, f"{value:+.3f} [{lower:+.3f}, {upper:+.3f}]", va="center", fontsize=9
        )
    axis.axvline(0, color="black", linestyle="--", linewidth=1)
    axis.set_yticks(np.arange(len(rows)), [row[0] for row in rows])
    axis.invert_yaxis()
    axis.set_xlim(-1.6, 1.85)
    axis.set_xlabel("Model minus baseline NLL (negative favors the model)")
    axis.set_title("Paired configuration-bootstrap differences: final models averaged across seeds")
    axis.grid(axis="x", alpha=0.2)
    figure.supxlabel(
        "Descriptive 95% intervals from 10,000 group resamples; "
        "validation selection uncertainty is not included.",
        fontsize=9,
    )
    _save(figure, output, "paired_intervals")


def _strata_plot(summary, baseline, output):
    """Show small-sample strata and photon concentration beside their scores."""
    names = [
        "retained_0",
        "retained_1_to_5",
        "retained_6_to_10",
        "retained_11_to_30",
        "retained_above_30",
    ]
    labels = ["0", "1-5", "6-10", "11-30", ">30"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    values = np.asarray(
        [
            [run["final"]["strata"][name]["scores"]["model"]["nats_per_photon"] for name in names]
            for run in summary["runs"]
        ]
    )
    axes[0].errorbar(
        np.arange(5),
        values.mean(0),
        yerr=values.std(0, ddof=1),
        marker="o",
        color=COLORS["model"],
        label="Long model (mean +/- seed SD)",
    )
    for key, label, color in (
        ("fixed_mixture", "Fixed mixture", COLORS["fixed"]),
        ("previously_tuned_mixture", "Tuned mixture", COLORS["tuned"]),
    ):
        axes[0].plot(
            np.arange(5),
            [baseline[key][name]["nats_per_photon"] for name in names],
            "o--",
            label=label,
            color=color,
        )
    axes[0].set_ylabel("NLL (nats per target photon)")
    axes[0].set_title("Prediction quality by retained input count")
    axes[0].legend(fontsize=8)
    score = summary["runs"][0]["final"]["strata"]
    photons = [score[name]["scores"]["model"]["held_out_photons"] for name in names]
    events = [score[name]["scores"]["model"]["events_with_target"] for name in names]
    axes[1].bar(np.arange(5), photons, color=["#94a3b8"] * 4 + ["#bd7140"])
    for index, (number, count) in enumerate(zip(photons, events, strict=True)):
        axes[1].text(
            index, number + 12, f"{number} photons\n{count} scored events", ha="center", fontsize=8
        )
    axes[1].set_ylim(0, max(photons) * 1.22)
    axes[1].set_ylabel("Number of held-out photons")
    axes[1].set_title("Five bright events supply 80% of scored photons")
    for axis in axes:
        axis.set_xticks(np.arange(5), labels)
        axis.set_xlabel("Retained input photon count")
        axis.grid(axis="y", alpha=0.15)
        axis.set_axisbelow(True)
    figure.suptitle("Input-count strata expose differences hidden by the overall average")
    _save(figure, output, "count_strata")


@torch.inference_mode()
def _event_context(root, summary):
    """Compute physical-event predictions before loading evaluation-only truth."""
    config = summary["config"]
    path = Path(config["dataset"])
    if shard_sha256(path) != config["dataset_sha256"]:
        message = "Observation data differ from the recorded pilot"
        raise ValueError(message)
    torch.set_num_threads(4)
    dataset = NeutrinoPhotonDataset(path, "validation", seed=config["photon_split_seed"])
    view = freeze_photon_view(dataset)
    checkpoint_path = root / "longer_pilot_v1/seed_42/step_005000.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["dataset_sha256"] != config["dataset_sha256"]:
        message = "Checkpoint data identity differs from the observation shard"
        raise ValueError(message)
    model = DensePhotonReference(checkpoint["model_channels"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    probability = []
    for begin in range(0, len(view.inputs), 16):
        counts = view.inputs[begin : begin + 16]
        logits = model(counts, view.positions_m, view.live, torch.full((len(counts),), view.p))
        probability.append(photon_count_log_probabilities(logits, view.live).exp().numpy())
    probability = np.concatenate(probability)
    np.testing.assert_allclose(probability.sum((1, 2)), 1, atol=2e-6)
    recorded = summary["runs"][0]["final"]["per_event"]
    for index, row in enumerate(recorded):
        if int(view.event_ids[index]) != row["event_id"]:
            message = "Saved scores and visualization event IDs disagree"
            raise ValueError(message)
        target = view.targets[index].numpy()
        scored = target > 0
        actual = float(-(target[scored] * np.log(probability[index][scored])).sum())
        np.testing.assert_allclose(actual, row["nll_by_method"]["model"], rtol=2e-6, atol=1e-5)
    training = NeutrinoPhotonDataset(path, "train", training=False)
    observed = np.zeros_like(probability[0], dtype=np.float64)
    for index in range(len(training)):
        observed += training.counts(index)[0]
    settings = config["baselines"]
    original = observed + settings["original_prior_pseudocount"]
    original /= original.sum()
    centers = (dataset.edges_ns[:-1] + dataset.edges_ns[1:]) / 2
    kernel = np.exp(
        -0.5
        * ((centers[:, None] - centers[None, :]) / settings["tuned_prior_time_bandwidth_ns"]) ** 2
    )
    kernel /= kernel.sum(1, keepdims=True)
    tuned = observed @ kernel + settings["tuned_prior_pseudocount"]
    tuned /= tuned.sum()
    # Truth is read only after model/baseline inference inputs and probabilities
    # have been fixed. It supplies display context, never a forward argument.
    truth_path = root / "prometheus_pone_reference_1000.evaluation_only.json"
    truth = {row["event_no"]: row for row in json.loads(truth_path.read_text())["events"]}
    return {
        "dataset": dataset,
        "view": view,
        "q": probability,
        "original_prior": original,
        "tuned_prior": tuned,
        "truth": truth,
        "checkpoint": checkpoint_path,
    }


def _select_cases(context):
    """Select documented count categories without inspecting model quality."""
    view = context["view"]
    retained = view.inputs.sum((1, 2)).numpy()
    target = view.target_totals
    candidates = [
        ("Empty observation", (retained == 0) & (target == 0)),
        ("Empty retained input", (retained == 0) & (target > 0)),
        ("1-5 retained", (retained >= 1) & (retained <= 5) & (target > 0)),
        ("6-10 retained", (retained >= 6) & (retained <= 10) & (target > 0)),
        ("11-30 retained", (retained >= 11) & (retained <= 30) & (target > 0)),
        ("Brightest retained input", retained == retained.max()),
    ]
    cases = []
    for label, mask in candidates:
        indices = np.flatnonzero(mask)
        index = sorted(indices, key=lambda i: view.event_ids[i])[len(indices) // 2]
        a, b = view.inputs[index].numpy(), view.targets[index].numpy()
        fixed_pdf = a / a.sum() if a.sum() else context["original_prior"]
        tuned_pdf = a / a.sum() if a.sum() else context["tuned_prior"]
        cases.append(
            {
                "label": label,
                "event_id": int(view.event_ids[index]),
                "A": a,
                "B": b,
                "fixed": 0.5 * fixed_pdf + 0.5 * context["original_prior"],
                "tuned": 0.75 * tuned_pdf + 0.25 * context["tuned_prior"],
                "q": context["q"][index],
                "truth": context["truth"][int(view.event_ids[index])],
            }
        )
    return cases


def _event_heatmaps(context, cases, output):
    """Display observed detections and normalized PDFs with distinct units."""
    figure, axes = plt.subplots(len(cases), 5, figsize=(16, 13), layout="constrained")
    count_max = max(float(case[key].max()) for case in cases for key in ("A", "B"))
    probability_max = max(
        float(case[key].max()) for case in cases for key in ("fixed", "tuned", "q")
    )
    count_norm = LogNorm(1, max(2, count_max))
    probability_norm = LogNorm(1e-7, probability_max)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#f5f7fa")
    limit = int(np.searchsorted(context["dataset"].edges_ns, 2000))
    titles = [
        "Retained A\nobserved counts",
        "Held-out B\nobserved counts",
        "Fixed mixture\nprobability / bin",
        "Tuned mixture\nprobability / bin",
        "Learned q, seed 42\nprobability / bin",
    ]
    for row, case in enumerate(cases):
        for column, key in enumerate(("A", "B", "fixed", "tuned", "q")):
            value = case[key][:, :limit]
            shown = np.ma.masked_where(value == 0, value)
            rendered = axes[row, column].imshow(
                shown,
                origin="lower",
                aspect="auto",
                extent=[0, 2, -0.5, 59.5],
                norm=count_norm if column < 2 else probability_norm,
                cmap=cmap,
            )
            if row == 0:
                axes[row, column].set_title(titles[column], fontsize=10)
            axes[row, column].set_yticks([0, 20, 40, 59])
            if column == 0:
                axes[row, column].set_ylabel(
                    f"{case['label']}\nID {case['event_id']} | "
                    f"A={case['A'].sum()}, B={case['B'].sum()}\nSensor index",
                    fontsize=8,
                )
            if row == len(cases) - 1:
                axes[row, column].set_xlabel("Source time (microseconds)")
            for boundary in (19.5, 39.5):
                axes[row, column].axhline(boundary, color="white", alpha=0.3, linewidth=0.7)
            if column < 2 and not value.any():
                axes[row, column].text(
                    1, 30, "No detections", ha="center", color="#58657a", fontsize=9
                )
            if column == 1:
                count_image = rendered
            if column == 4:
                probability_image = rendered
    figure.colorbar(
        count_image,
        ax=axes[:, :2],
        label="Observed photons per bin (log colors)",
        shrink=0.6,
        pad=0.01,
    )
    figure.colorbar(
        probability_image,
        ax=axes[:, 2:],
        label="Probability per bin (log colors)",
        shrink=0.6,
        pad=0.01,
    )
    figure.suptitle(
        "Validation events: observations, simple estimators and the learned arrival distribution"
    )
    figure.supxlabel(
        "Display-only zoom: 0-2 microseconds. Every PDF is normalized over all 60 "
        "sensors and the full 0-6.4 microsecond window.\n"
        "B is a single held-out photon realization, not clean expected-intensity GT. "
        "Cases were selected by counts and event IDs, not NLL.",
        fontsize=9,
    )
    _save(figure, output, "event_comparisons")


def _geometry(context, cases, output):
    """Show simulator initial vertices as truth context, not a reconstruction."""
    positions = context["view"].positions_m.numpy()
    figure = plt.figure(figsize=(12, 7), layout="constrained")
    for index, case in enumerate(cases):
        axis = figure.add_subplot(2, 3, index + 1, projection="3d")
        sensor_mass = case["q"].sum(1)
        scatter = axis.scatter(
            *positions.T,
            c=sensor_mass,
            s=15 + 10 * np.log1p(case["A"].sum(1)),
            cmap="plasma",
            vmin=0,
            vmax=1,
        )
        vertex = [case["truth"]["initial_state_" + coordinate] for coordinate in ("x", "y", "z")]
        axis.scatter(
            *vertex, marker="*", s=160, c="#18a56c", edgecolors="black", label="MC initial vertex"
        )
        axis.set_title(f"{case['label']} | ID {case['event_id']}", fontsize=9)
        axis.set_xlabel("x (m)", fontsize=8)
        axis.set_ylabel("y (m)", fontsize=8)
        axis.set_zlabel("z (m)", fontsize=8)
        axis.tick_params(labelsize=7)
        axis.view_init(elev=18, azim=35)
    figure.colorbar(
        scatter,
        ax=figure.axes,
        label="Predicted probability per sensor (sum over time)",
        shrink=0.6,
        pad=0.02,
    )
    figure.suptitle(
        "P-ONE geometry: simulated initial vertex (green star) and predicted sensor probabilities"
    )
    figure.supxlabel(
        "Marker size follows retained hits. Vertex truth is shown only as context; "
        "no vertex reconstruction or clean arrival-PDF GT is inferred.",
        fontsize=9,
    )
    _save(figure, output, "simulation_truth_geometry")


def _analytic_truth(root, output):
    """Reconstruct the generator's known expected PDFs strictly for display."""
    dataset = NeutrinoPhotonDataset(root / "analytic_reference_1000.h5", "validation")
    config = dataset.provenance["config"]
    levels = np.asarray(config["analytic"]["expected_signal_photons"])
    background = config["analytic"]["expected_background_photons"]
    truth_path = root / "analytic_reference_1000.evaluation_only.npz"
    centers = (dataset.edges_ns[:-1] + dataset.edges_ns[1:]) / 2
    figure, axes = plt.subplots(3, 4, figsize=(14, 8.5), layout="constrained")
    truth_records = []
    with np.load(truth_path) as truth:
        for row, level in enumerate((0, 5, 30)):
            candidates = [
                i
                for i in range(len(dataset))
                if levels[dataset.event_ids[dataset.indices[i]] % len(levels)] == level
            ]
            index = candidates[0]
            source = dataset.indices[index]
            event_id = int(dataset.event_ids[source])
            group = int(dataset.configuration_ids[source])
            distance = np.linalg.norm(
                dataset.positions_m - truth["latent_source_position_m"][group], axis=1
            )
            delay = centers[None, :] - (500 + distance / 0.225)[:, None]
            temporal = np.exp(-0.5 * (delay / 60) ** 2)
            temporal += 0.35 * np.exp(-np.maximum(delay, 0) / 200) * (delay >= 0)
            spatial = np.exp(-distance / 300) / (distance**2 + 50**2)
            signal_pdf = spatial[:, None] * temporal
            signal_pdf /= signal_pdf.sum()
            intensity = level * signal_pdf + background / signal_pdf.size
            np.testing.assert_allclose(
                intensity.sum(), truth["expected_total_photons"][source], rtol=1e-10
            )
            probability = intensity / intensity.sum()
            item = dataset[index]
            a, b = item["input"].numpy(), item["target"].numpy()
            truth_image = axes[row, 0].imshow(
                probability,
                origin="lower",
                aspect="auto",
                extent=[0, 6.4, 0, 256],
                norm=LogNorm(1e-7, 0.01),
                cmap="magma",
            )
            for column, counts in ((1, a), (2, b)):
                sensor, time = np.nonzero(counts)
                axes[row, column].scatter(
                    centers[time] / 1000,
                    sensor,
                    c=counts[sensor, time],
                    s=14 + 9 * counts[sensor, time],
                    cmap="Blues",
                    vmin=0,
                    vmax=6,
                    edgecolors="#4682a8",
                    linewidths=0.3,
                )
                axes[row, column].set_xlim(0, 6.4)
                axes[row, column].set_ylim(0, 256)
                if not counts.any():
                    axes[row, column].text(3.2, 128, "No detections", ha="center", color="#64748b")
            axes[row, 0].set_ylabel(
                f"Expected signal {level} + background {background:g}\n"
                f"ID {event_id} | A={a.sum()}, B={b.sum()}\nSensor index",
                fontsize=8,
            )
            axes[row, 3].plot(
                centers / 1000, probability.sum(0), color="#9b284e", label="Known expected PDF (GT)"
            )
            for counts, label, color in (
                (a, "Retained A (empirical)", "#3678b5"),
                (b, "Held-out B (empirical)", COLORS["target"]),
            ):
                if counts.sum():
                    axes[row, 3].plot(
                        centers / 1000,
                        counts.sum(0) / counts.sum(),
                        color=color,
                        alpha=0.75,
                        linewidth=1,
                        label=label,
                    )
            axes[row, 3].set_xlim(0, 6.4)
            axes[row, 3].set_ylabel("Probability per time bin", fontsize=8)
            if row == 0:
                for axis, title in zip(
                    axes[row],
                    (
                        "Known expected location PDF",
                        "Observed retained photons A",
                        "Observed held-out photons B",
                        "Time marginal: expected vs realized",
                    ),
                    strict=True,
                ):
                    axis.set_title(title, fontsize=9)
                axes[row, 3].legend(fontsize=7)
            for axis in axes[row]:
                axis.set_xlabel("Time (microseconds)", fontsize=8)
                axis.tick_params(labelsize=8)
            truth_records.append(
                {
                    "event_id": event_id,
                    "expected_signal_photons": level,
                    "expected_background_photons": background,
                    "observed_A": int(a.sum()),
                    "observed_B": int(b.sum()),
                    "expected_pdf_mass": float(probability.sum()),
                }
            )
    figure.colorbar(
        truth_image,
        ax=axes[:, 0],
        label="Known expected probability / bin (log colors)",
        shrink=0.65,
        pad=0.01,
    )
    figure.suptitle("Analytic reference: known expected PDF and two random photon realizations")
    figure.supxlabel(
        "The constructed analytic generator supplies an exact expected PDF. "
        "No learned model is shown here; this is not a P-ONE model benchmark.\n"
        "GT is reconstructed from evaluation-only source parameters; it is never "
        "read by the training loader. Expected PDF colors are logarithmic; "
        "dots show photon counts.",
        fontsize=9,
    )
    _save(figure, output, "analytic_ground_truth")
    return truth_records


def _roadmap(output):
    """Draw the planned phases and their dependency gates."""
    figure, axis = plt.subplots(figsize=(9, 10))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    boxes = [
        (
            0.5,
            0.93,
            "0. Freeze evidence and protocol",
            "Primary metric, budgets and decision rules",
        ),
        (
            0.5,
            0.80,
            "1. Audit data and grouping",
            "Unit photons, clocks, selection, related configurations",
        ),
        (0.25, 0.65, "2. Bound execution memory", "Stream counts, donors, scoring and bootstrap"),
        (
            0.75,
            0.65,
            "3. Build persistent cohorts",
            "100k cohort; nested 10k engineering qualification",
        ),
        (
            0.5,
            0.49,
            "4. Train and select on validation",
            "Three seeds; strong controls; fixed training budgets",
        ),
        (
            0.5,
            0.35,
            "5. Evaluate locked independent test",
            "Low-count NLL, paired intervals and resource checks",
        ),
        (
            0.5,
            0.21,
            "6. Explain limits with targeted studies",
            "Fine time resolution, supervision and response effects",
        ),
        (
            0.5,
            0.07,
            "7. Real data and physics utility",
            "Calibrated response; run-separated/downstream evaluation",
        ),
    ]
    for index, (x, y, title, subtitle) in enumerate(boxes):
        width = 0.44 if index in (2, 3) else 0.78
        axis.add_patch(
            FancyBboxPatch(
                (x - width / 2, y - 0.044),
                width,
                0.088,
                boxstyle="round,pad=0.01",
                facecolor="#eef5f9",
                edgecolor="#476c83",
                linewidth=1.3,
            )
        )
        axis.text(x, y + 0.013, title, ha="center", va="center", fontsize=10, weight="bold")
        axis.text(x, y - 0.019, subtitle, ha="center", va="center", fontsize=7.5)
    for begin, end in ((0, 1), (1, 2), (1, 3), (2, 4), (3, 4), (4, 5), (5, 6), (6, 7)):
        x0, y0 = boxes[begin][:2]
        x1, y1 = boxes[end][:2]
        axis.add_patch(
            FancyArrowPatch(
                (x0, y0 - 0.055),
                (x1, y1 + 0.055),
                arrowstyle="-|>",
                mutation_scale=13,
                color="#526579",
                linewidth=1.3,
            )
        )
    axis.set_title(
        "Phased study roadmap\nCurrent pilot complete; larger-study phases are planned",
        fontsize=14,
        pad=20,
    )
    _save(figure, output, "phased_roadmap")


def _add_empirical_marginals(figure, case, times):
    """Add observed-location marginals only when their normalization is defined."""
    import plotly.graph_objects as go

    for key, label, color in (
        ("A", "Retained A (empirical)", "#3678b5"),
        ("B", "Held-out B (empirical)", COLORS["target"]),
    ):
        if not case[key].sum():
            continue
        normalized = case[key] / case[key].sum()
        for axis, column in ((1, 1), (0, 2)):
            figure.add_trace(
                go.Scatter(
                    x=np.arange(60) if axis == 1 else times,
                    y=normalized.sum(axis),
                    mode="lines",
                    name=label,
                    legendgroup=key,
                    showlegend=axis == 1,
                    line={"color": color, "dash": "dot"},
                ),
                row=2,
                col=column,
            )


def _interactive_events(context, cases):
    """Build a self-contained event selector with interactive geometry/marginals."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    figure = make_subplots(
        rows=2,
        cols=3,
        specs=[[{}, {}, {}], [{}, {}, {"type": "scene"}]],
        subplot_titles=(
            "Retained A (observed counts)",
            "Held-out B (single realization)",
            "Learned q (full-support PDF)",
            "Sensor marginal",
            "Time marginal (full window)",
            "Geometry and MC initial vertex",
        ),
        vertical_spacing=0.16,
    )
    positions = context["view"].positions_m.numpy()
    times = (context["dataset"].edges_ns[:-1] + context["dataset"].edges_ns[1:]) / 2000
    limit = int(np.searchsorted(context["dataset"].edges_ns, 2000))
    intervals = []
    for index, case in enumerate(cases):
        begin = len(figure.data)
        for column, key in enumerate(("A", "B", "q"), 1):
            value = case[key][:, :limit]
            z = np.log10(value.clip(1e-8)) if key == "q" else np.log10(1 + value)
            figure.add_trace(
                go.Heatmap(
                    x=times[:limit],
                    y=np.arange(60),
                    z=z,
                    customdata=value,
                    colorscale="Magma" if key == "q" else "Blues",
                    zmin=-8 if key == "q" else 0,
                    zmax=0 if key == "q" else 3,
                    showscale=False,
                    hovertemplate=(
                        "Time %{x:.3f} us<br>Sensor %{y}<br>Value %{customdata:.5g}<extra></extra>"
                    ),
                    name=key,
                ),
                row=1,
                col=column,
            )
        for key, label, color in (
            ("q", "Learned q", COLORS["model"]),
            ("fixed", "Fixed mixture", COLORS["fixed"]),
            ("tuned", "Tuned mixture", COLORS["tuned"]),
        ):
            for axis, row, column in ((1, 2, 1), (0, 2, 2)):
                figure.add_trace(
                    go.Scatter(
                        x=np.arange(60) if axis == 1 else times,
                        y=case[key].sum(axis),
                        mode="lines",
                        name=label,
                        legendgroup=key,
                        showlegend=axis == 1,
                        line={"color": color},
                    ),
                    row=row,
                    col=column,
                )
        _add_empirical_marginals(figure, case, times)
        figure.add_trace(
            go.Scatter3d(
                x=positions[:, 0],
                y=positions[:, 1],
                z=positions[:, 2],
                mode="markers",
                marker={
                    "color": case["q"].sum(1),
                    "colorscale": "Plasma",
                    "cmin": 0,
                    "cmax": 1,
                    "size": 4 + 2 * np.log1p(case["A"].sum(1)),
                },
                text=[f"Sensor {i}" for i in range(60)],
                name="Sensors: color=predicted mass",
                showlegend=False,
            ),
            row=2,
            col=3,
        )
        vertex = [case["truth"]["initial_state_" + coordinate] for coordinate in ("x", "y", "z")]
        figure.add_trace(
            go.Scatter3d(
                x=[vertex[0]],
                y=[vertex[1]],
                z=[vertex[2]],
                mode="markers",
                marker={"size": 8, "symbol": "diamond", "color": "#18a56c"},
                name="MC initial vertex (context)",
                showlegend=True,
            ),
            row=2,
            col=3,
        )
        end = len(figure.data)
        intervals.append((begin, end))
        for trace in figure.data[begin:end]:
            trace.visible = index == 0
    buttons = []
    for case, (begin, end) in zip(cases, intervals, strict=True):
        buttons.append(
            {
                "label": (
                    f"{case['label']} | ID {case['event_id']} | "
                    f"A={case['A'].sum()}, B={case['B'].sum()}"
                ),
                "method": "update",
                "args": [
                    {"visible": [begin <= i < end for i in range(len(figure.data))]},
                    {
                        "title": (
                            f"{case['label']} | validation ID {case['event_id']} | "
                            "seed 42, step 5000"
                        )
                    },
                ],
            }
        )
    figure.update_layout(
        height=850,
        title="Select a validation event; rotate the detector geometry",
        template="plotly_white",
        margin={"t": 150, "b": 90},
        legend={"orientation": "h", "y": -0.17},
        updatemenus=[{"buttons": buttons, "x": 0, "y": 1.21, "xanchor": "left"}],
        scene={"xaxis_title": "x (m)", "yaxis_title": "y (m)", "zaxis_title": "z (m)"},
    )
    for column in (1, 2, 3):
        figure.update_xaxes(title_text="Time (microseconds; display zoom)", row=1, col=column)
        figure.update_yaxes(title_text="Sensor index", row=1, col=column)
    figure.update_xaxes(title_text="Sensor index", row=2, col=1)
    figure.update_xaxes(title_text="Time (microseconds)", row=2, col=2)
    figure.update_yaxes(title_text="Probability mass", row=2, col=1)
    figure.update_yaxes(title_text="Probability / time bin", row=2, col=2)
    return figure.to_html(
        full_html=False,
        include_plotlyjs=False,
        config={"responsive": True, "displaylogo": False},
        div_id="event-explorer",
    )


def _viewer(output, summary, context, cases, plan):
    """Embed all assets and Plotly itself so the report works without a network."""
    from plotly.offline import get_plotlyjs

    def picture(name, caption):
        encoded = base64.b64encode((output / f"{name}.png").read_bytes()).decode("ascii")
        return (
            f'<figure><img alt="{html.escape(caption)}" '
            f'src="data:image/png;base64,{encoded}">'
            f"<figcaption>{html.escape(caption)}</figcaption></figure>"
        )

    sections = [
        picture(
            "score_comparison",
            "Aggregate validation scores. Lower NLL is better; error bars show seed SD.",
        ),
        picture(
            "paired_intervals",
            "Paired event/configuration uncertainty. "
            "The tuned-baseline low-count advantage narrowly excludes zero.",
        ),
        picture("learning_curves", "The same architecture and loss improve with longer training."),
        picture("count_strata", "The overall average is dominated by a few bright observations."),
        '<h2 id="events">Event explorer</h2><p>Model: seed 42, final step 5000. '
        "Top maps show 0-2 us; PDFs remain normalized over 0-6.4 us. Counts and "
        "probabilities have different units. Hover for values, select an event, "
        "and rotate the detector. The green diamond is simulator vertex truth, "
        "not a predicted vertex or photon-PDF GT.</p>",
        _interactive_events(context, cases),
        picture(
            "event_comparisons",
            "Representative cases selected by counts and IDs, without using prediction "
            "quality. B is one held-out photon realization.",
        ),
        picture(
            "simulation_truth_geometry",
            "Simulation truth context: green stars mark initial vertices; "
            "sensor colors show learned photon probability.",
        ),
        '<h2 id="gt">Known analytic GT</h2><p>The analytic generator supplies an '
        "exact expected arrival PDF. These plots compare that PDF with random photon "
        "realizations; no learned model is shown. The physical P-ONE shard does not "
        "supply this clean latent PDF.</p>",
        picture(
            "analytic_ground_truth",
            "GT is evaluation-only analytic intensity, not an individual photon image "
            "and not a training target.",
        ),
        '<h2 id="plan">Phased plan</h2>',
        picture(
            "phased_roadmap",
            "Phases are prospective. The larger 100k-event independent comparison has not started.",
        ),
        "<details><summary>Read the detailed phased plan</summary><pre>"
        + html.escape(plan.read_text(encoding="utf-8"))
        + "</pre></details>",
    ]
    low = summary["across_seeds"]["final"]["retained_1_to_10"]["mean_seed_nats_per_photon"]
    document = (
        """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Photon self-splitting: pilot results and phased plan</title><style>
body{font-family:system-ui,sans-serif;color:#243448;background:#f3f6fa;
margin:0;line-height:1.5}
main{max-width:1440px;margin:auto;padding:28px}
header,section{background:white;border:1px solid #dce3eb;border-radius:12px;
padding:24px;margin-bottom:22px}
h1{font-size:29px;margin:0 0 10px}h2{margin-top:38px}
nav{display:flex;gap:20px;flex-wrap:wrap}a{color:#166579}
.badge{display:inline-block;padding:5px 10px;border-radius:6px;
background:#e8f2f5;margin:8px 8px 0 0}
figure{margin:25px 0}img{width:100%;height:auto}
figcaption{font-size:14px;color:#56667a}
pre{white-space:pre-wrap;font-size:13px;background:#f6f8fb;padding:18px}
summary{cursor:pointer;font-weight:600}
.note{border-left:4px solid #d58b38;padding:12px 16px;background:#fff8ee}
</style><script>"""
        + get_plotlyjs()
        + """</script></head><body><main><header>
<h1>Photon self-splitting: pilot results</h1>
<p>Longer training improves low-count validation. The tuned simple mixture still
leads on the overall point estimate.</p><nav>
<a href="#scores">Score comparisons</a><a href="#events">Event predictions</a>
<a href="#gt">Analytic GT</a><a href="#plan">Detailed phased plan</a></nav>"""
    )
    document += (
        f'<span class="badge">Low-count model NLL: {low:.3f}</span>'
        '<span class="badge">3 seeds, 5000 steps each</span>'
        '<span class="badge">Validation only: 100 events</span>'
        '<span class="badge">No clean-PDF supervision</span></header>'
    )
    document += (
        '<section id="scores"><p class="note">The low-count result uses 120 target '
        "photons in 36 scored events. Its bootstrap interval is exploratory; the "
        "reserved test was not scored. A held-out photon realization is not a "
        "clean expected-arrival ground truth.</p>"
        + "".join(sections)
        + "</section></main></body></html>"
    )
    (output / "viewer.html").write_text(document, encoding="utf-8")


def create_visualizations(root: Path, output: Path, plan: Path) -> dict:
    """Create the requested figures without training or opening the test partition."""
    if output.exists():
        message = f"Refusing to overwrite visualization artifacts: {output}"
        raise FileExistsError(message)
    summary_path = root / "longer_pilot_v1/summary.json"
    summary = json.loads(summary_path.read_text())
    short = json.loads((root / "pone_pilot_evaluation.json").read_text())
    baseline = json.loads((root / "longer_pilot_v1/baseline_scores.json").read_text())
    context = _event_context(root, summary)
    cases = _select_cases(context)
    output.mkdir(parents=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    _comparisons(summary, short, baseline, output)
    _strata_plot(summary, baseline, output)
    _event_heatmaps(context, cases, output)
    _geometry(context, cases, output)
    analytic = _analytic_truth(root, output)
    _roadmap(output)
    for extension in ("png", "svg"):
        shutil.copyfile(
            root / f"longer_pilot_v1/learning_curves.{extension}",
            output / f"learning_curves.{extension}",
        )
    _viewer(output, summary, context, cases, plan)
    manifest = {
        "scope": "existing_pilot_visualization; no_new_training_or_test_scoring",
        "physical_dataset_sha256": summary["config"]["dataset_sha256"],
        "summary_sha256": shard_sha256(summary_path),
        "visualization_code_sha256": shard_sha256(__file__),
        "checkpoint": str(context["checkpoint"]),
        "checkpoint_sha256": shard_sha256(context["checkpoint"]),
        "plan_sha256": shard_sha256(plan),
        "event_map_model": "seed_42_final_5000_steps_not_probability_ensemble",
        "event_selection": (
            "median_event_ID_in_count_category_with_nonempty_B; empty_observation_category; "
            "maximum_retained_count_category; never_uses_NLL"
        ),
        "physical_clean_expected_pdf_GT_available": False,
        "physical_GT_display": "MC initial vertex only; not a predicted reconstruction",
        "analytic_GT_display": (
            "known expected location PDF reconstructed from evaluation-only generator "
            "parameters; no learned model shown"
        ),
        "truth_sidecars_used_only_for_visualization": True,
        "test_partition_scored": False,
        "probability_display_window_ns": [0, 2000],
        "probability_normalization_window_ns": [0, 6400],
        "physical_events": [
            {
                "event_id": case["event_id"],
                "category": case["label"],
                "retained_photons": int(case["A"].sum()),
                "held_out_photons": int(case["B"].sum()),
                "predicted_probability_mass": float(case["q"].sum()),
                "probability_mass_in_display_window": float(case["q"][:, :80].sum()),
            }
            for case in cases
        ],
        "analytic_events": analytic,
        "png_exports": sorted(path.name for path in output.glob("*.png")),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output / "README.txt").write_text(
        "Open viewer.html for the self-contained offline results/GT/plan viewer.\n"
        "PNG and SVG versions accompany each figure. Training and test evaluation were not run.\n"
        "Aggregate scores average seed NLLs; event PDFs use seed 42, step 5000.\n"
        "B is held-out observed photons, not clean photon-intensity GT. Physical MC truth\n"
        "shows the initial vertex only. Analytic GT is an exact expected PDF in a separate\n"
        "constructed dataset; no learned model is compared to that GT in these figures.\n"
        "Truth was loaded for visualization only, after physical predictions were fixed.\n"
        "Event maps zoom to 0-2 us without changing the full 0-6.4 us normalization.\n"
        "manifest.json records input identities, chosen cases and display conventions.\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    """Export figures and an offline interactive viewer from existing pilot artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/neutrino/controlled_reference"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=Path("plans/neutrino_phased_plan.txt"))
    args = parser.parse_args()
    create_visualizations(args.root, args.output, args.plan)


if __name__ == "__main__":
    main()
