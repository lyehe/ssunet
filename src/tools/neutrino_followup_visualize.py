"""Export completed mechanism, response, real-pulse and labelled-probe comparisons."""

from __future__ import annotations

import argparse
import base64
import json
import shutil
import zipfile
from pathlib import Path

import matplotlib
import numpy as np
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from ..ssunet.models.neutrino_pulses import DenseMarkedPulseReference
from ..ssunet.neutrino_evaluation import photon_score_summary
from ..ssunet.neutrino_pulse_baselines import MarkedPulseBaselines
from ..ssunet.photon_reference import photon_count_log_probabilities
from .neutrino_prepare import write_json
from .neutrino_pulse_study import pulse_dataset
from .neutrino_study import _read_scalar_scores
from .neutrino_study_visualize import save_figure


def _read(path):
    """Load a completed scalar report, never start another evaluation."""
    return json.loads(Path(path).read_text())


def _mechanism_figures(root, precision, continuation, output):
    """Keep identical-data architecture comparisons separate from the failure fork."""
    records, values = _read_scalar_scores(root / "test_mean_seed_scores.npz")
    baseline = _read(root / "baseline_selection.json")["selected_method"]
    methods = [baseline, "dense_fresh", "dense_fixed", "fine_fresh", "residual_fresh"]
    summaries = {key: photon_score_summary(value, records) for key, value in values.items()}
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for axis, stratum in zip(axes, ("all", "retained_1_to_10"), strict=True):
        numbers = [summaries[key][stratum]["nats_per_photon"] for key in methods]
        axis.barh(methods, numbers, color=["#168496", "#657a90", "#91a3b1", "#cc6b29", "#915a98"])
        axis.invert_yaxis()
        axis.set_xlim(0, max(numbers) * 1.13)
        for index, number in enumerate(numbers):
            axis.text(number + 0.04, index, f"{number:.3f}", va="center")
        axis.set_title(stratum)
        axis.set_xlabel("Held-out NLL (nats/photon; lower is better)")
    figure.suptitle("Fresh-family mechanism test: matched observations, epochs and seed budgets")
    save_figure(figure, output, "mechanism_comparison")
    fork = _read(continuation / "test_results.json")
    small = _read(precision / "test_results.json")
    rows = [
        (
            "Small cohort: FP32 vs BF16",
            small["versus_mixed_precision"]["contrasts"]["retained_1_to_10"]["dense_fresh"],
        ),
        (
            "Failure seed: matched epoch 5",
            fork["fixed_epoch_contrast"]["contrasts"]["retained_1_to_10"]["bf16_epoch5"],
        ),
        (
            "Failure seed: selected checkpoints",
            fork["validation_selected_contrast"]["contrasts"]["retained_1_to_10"][
                "bf16_selected_epoch2"
            ],
        ),
    ]
    figure, axis = plt.subplots(figsize=(11, 4), layout="constrained")
    for index, (_, interval) in enumerate(rows):
        point = interval["model_minus_control_nats_per_photon"]
        lo, hi = interval["descriptive_95_percent_interval"]
        axis.errorbar(
            point, index, xerr=[[point - lo], [hi - point]], fmt="o", capsize=4, color="#cc6b29"
        )
        axis.annotate(f" {point:+.3f} [{lo:+.3f}, {hi:+.3f}]", (hi, index), va="center", fontsize=8)
    axis.set_yticks(range(len(rows)), [row[0] for row in rows])
    axis.invert_yaxis()
    axis.axvline(0, color="black", linestyle="--")
    axis.margins(x=0.45)
    axis.set_xlabel("FP32 model minus BF16 control NLL (nats/photon)")
    axis.set_title("Precision diagnostics on fresh families")
    figure.supxlabel(
        "The failure continuation is one preidentified seed with a shared BF16 prefix; "
        "no multi-seed causal generalization is inferred.",
        fontsize=8,
    )
    save_figure(figure, output, "precision_comparison")


def _response_figure(root, output):
    """Display within-response paired contrasts without claiming a physical calibration."""
    result = _read(root / "response_diagnostics/summary.json")
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    reports = result["reports"]
    for axis, stratum in zip(axes, ("all", "retained_1_to_10"), strict=True):
        for index, (name, report) in enumerate(reports.items()):
            interval = next(iter(report["contrasts"][stratum].values()))
            point = interval["model_minus_control_nats_per_photon"]
            lo, hi = interval["descriptive_95_percent_interval"]
            color = (
                "#cc6b29"
                if name in result["independent_Poisson_assumption_violations"]
                else "#168496"
            )
            axis.errorbar(
                point, index, xerr=[[point - lo], [hi - point]], fmt="o", capsize=4, color=color
            )
        axis.set_yticks(range(len(reports)), list(reports))
        axis.invert_yaxis()
        axis.axvline(0, color="black", linestyle="--")
        axis.set_xlabel("Model minus selected baseline NLL (nats/photon)")
        axis.set_title(stratum)
    figure.suptitle("Analytic response shifts: seed 42, no refitting")
    figure.supxlabel(
        "Orange: merging/dead time/afterpulses violate the independent-Poisson observation "
        "contract.\nIntervals are within-response exploratory group contrasts; "
        "strata can change membership under a response shift.",
        fontsize=8,
    )
    save_figure(figure, output, "response_diagnostics")


def _probe_figure(root, output):
    """Show physics-label probe results without relabelling them self-supervised targets."""
    result = _read(root / "downstream_probe/test_results.json")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    for axis, task, unit in zip(
        axes,
        ("energy", "direction"),
        ("Mean absolute log10-energy error", "Mean angular error (degrees)"),
        strict=True,
    ):
        values = result["metrics"][task]["mean_errors"]
        methods = ["raw", "baseline", "mean_seed_model"]
        axis.bar(methods, [values[key] for key in methods], color=["#657a90", "#168496", "#cc6b29"])
        axis.set_ylabel(unit)
        axis.set_title(task)
        axis.tick_params(axis="x", labelsize=8)
    figure.suptitle("Separate labelled linear probe: frozen pretraining, fresh test families")
    figure.supxlabel(
        "All methods receive equal-dimensional summaries and retained total. "
        "Physics labels are used only by the probe.\nCurated-MC linear-probe errors "
        "do not establish real-detector reconstruction benefit.",
        fontsize=8,
    )
    save_figure(figure, output, "downstream_probe")


def _real_score_figure(root, output):
    """Compare marked-pulse scores only within the same real data/support."""
    result = _read(root / "test_results.json")
    methods = list(result["scores"])
    numbers = [result["scores"][key]["all"]["nats_per_pulse"] for key in methods]
    figure, axis = plt.subplots(figsize=(10, 4.5), layout="constrained")
    axis.barh(
        methods, numbers, color=["#cc6b29" if key == "model" else "#168496" for key in methods]
    )
    axis.invert_yaxis()
    for index, value in enumerate(numbers):
        axis.text(value + 0.05, index, f"{value:.3f}", va="center")
    axis.set_xlim(0, max(numbers) * 1.12)
    axis.set_xlabel("Held-out joint location/time/charge-category NLL (nats/pulse)")
    axis.set_title("Run-separated real pulse test: calibrated observations, no photon GT")
    figure.supxlabel(
        "5,480 fixed spatial cells; 64 time bins of 1 us; 4 PE mark categories. "
        "Candidate geometry is not a certified live mask.\nTest is one bright selected run; "
        "no real low-count/no-interaction conclusion or cross-dataset NLL ranking is supported.",
        fontsize=8,
    )
    save_figure(figure, output, "real_marked_pulse_comparison")


@torch.inference_mode()
def _real_events(config, output):  # noqa: C901
    """Display whole-pulse observations and arrival marginals selected independently of fit."""
    root = Path(config["data"]["output"])
    torch.set_num_threads(4)
    lock = _read(root / "analysis_lock.json")
    model = DenseMarkedPulseReference(4, config["model"]["channels"])
    model.load_state_dict(
        torch.load(lock["checkpoints"]["42"]["path"], map_location="cpu", weights_only=True)[
            "state_dict"
        ]
    )
    model.eval()
    with np.load(root / "training_reference.npz", allow_pickle=False) as data:
        observed, training_a, training_ids = (
            data["observed_counts"],
            data["retained_totals"],
            data["event_ids"],
        )
    bank = MarkedPulseBaselines(
        observed, config["baselines"], selected=_read(root / "baseline_selection.json")
    )
    cases = []
    rules = [
        ("validation", "Empty retained validation input", "empty"),
        ("train", "Low-count training example (display only)", "low"),
        ("test", "Moderate retained test input", "moderate"),
        ("test", "Bright retained test input", "bright"),
    ]
    for partition, label, category in rules:
        dataset = pulse_dataset(config, partition)
        if partition == "train":
            ids, totals = training_ids, training_a
        else:
            source = root / (
                "validation_baseline_scores.npz"
                if partition == "validation"
                else "test_seed_42.npz"
            )
            records, _ = _read_scalar_scores(source)
            ids, totals = records.event_ids, records.retained_totals
        mask = {
            "empty": totals == 0,
            "low": (totals >= 1) & (totals <= 10),
            "moderate": (totals >= 100) & (totals <= 500),
            "bright": totals > 500,
        }[category]
        candidates = np.flatnonzero(mask)
        if not len(candidates):
            dataset.close()
            continue
        ordered = candidates[np.argsort(ids[candidates])]
        index = int(ordered[len(ordered) // 2])
        item = dataset[index]
        a, b = item["input"], item["target"]
        q = (
            photon_count_log_probabilities(
                model(a[None], item["positions_m"], item["live"], item["p"].reshape(1)),
                item["live"],
            )
            .exp()[0]
            .numpy()
            .reshape(-1, 4, a.shape[-1])
            .sum(1)
        )
        baseline = (
            bank.probabilities(a[None])[bank.winner][0].numpy().reshape(-1, 4, a.shape[-1]).sum(1)
        )
        a, b = (
            a.numpy().reshape(-1, 4, a.shape[-1]).sum(1),
            b.numpy().reshape(-1, 4, b.shape[-1]).sum(1),
        )
        selected = (
            np.argsort(-a.sum(1), kind="stable")[:20]
            if a.any()
            else np.linspace(0, len(a) - 1, 20, dtype=int)
        )
        cases.append(
            {
                "label": label,
                "event_id": int(ids[index]),
                "A": a,
                "B": b,
                "q": q,
                "baseline": baseline,
                "selected": selected,
            }
        )
        dataset.close()
    figure, axes = plt.subplots(
        len(cases), 5, figsize=(16, 2.7 * len(cases)), layout="constrained", squeeze=False
    )
    counts_norm = LogNorm(1, max(2, max(float(c[key].max()) for c in cases for key in ("A", "B"))))
    pdf_norm = LogNorm(1e-9, max(float(c[key].max()) for c in cases for key in ("q", "baseline")))
    for row, case in enumerate(cases):
        for column, key in enumerate(("A", "B", "baseline", "q")):
            values = case[key][case["selected"]]
            rendered = axes[row, column].imshow(
                np.ma.masked_where(values == 0, values),
                origin="lower",
                aspect="auto",
                extent=[0, 64, -0.5, 19.5],
                norm=counts_norm if column < 2 else pdf_norm,
                cmap="viridis",
            )
            if column == 1:
                count_image = rendered
            if column == 3:
                probability_image = rendered
            axes[row, column].set_yticks(
                [0, 5, 10, 15], [str(i) for i in case["selected"][[0, 5, 10, 15]]]
            )
            axes[row, column].set_xlabel("Source time (us)", fontsize=8)
            if row == 0:
                axes[row, column].set_title(
                    {
                        "A": "Retained A: pulse counts",
                        "B": "Held-out B: pulse counts",
                        "baseline": "Baseline: probability/bin",
                        "q": "Model: probability/bin",
                    }[key],
                    fontsize=9,
                )
        axes[row, 0].set_ylabel(
            f"{case['label']}\nA={case['A'].sum()}, B={case['B'].sum()}\nSpatial cell index",
            fontsize=8,
        )
        times = np.linspace(0.5, 63.5, 64)
        for key, color in (("baseline", "#168496"), ("q", "#cc6b29")):
            axes[row, 4].plot(times, case[key].sum(0), color=color, label=key)
        axes[row, 4].set_xlabel("Source time (us)")
        axes[row, 4].set_ylabel("Probability / time bin", fontsize=8)
        if row == 0:
            axes[row, 4].legend(fontsize=8)
    figure.suptitle("Real reconstructed pulses: arrival marginals, no clean-photon GT")
    figure.colorbar(
        count_image, ax=axes[:, :2], shrink=0.55, pad=0.01, label="Observed pulses/bin (log colors)"
    )
    figure.colorbar(
        probability_image,
        ax=axes[:, 2:4],
        shrink=0.55,
        pad=0.01,
        label="Arrival probability/bin (log colors)",
    )
    figure.supxlabel(
        "Heatmaps display the 20 cells with largest A counts (fixed spread for empty A); "
        "PDFs stay normalized over all cells/time.\nCharge marks are summed only for display. "
        "Model/controls use A only. Low-count training views are display context, "
        "not an independent test result.",
        fontsize=8,
    )
    save_figure(figure, output, "real_event_distributions")
    return [
        {
            "event_id": c["event_id"],
            "category": c["label"],
            "A": int(c["A"].sum()),
            "B": int(c["B"].sum()),
            "pdf_mass": float(c["q"].sum()),
            "displayed_pdf_mass": float(c["q"][c["selected"]].sum()),
        }
        for c in cases
    ]


def create_followup_figures(output):
    """Render all completed follow-ups into an offline report without new model fitting."""
    mechanism = Path("data/neutrino/mechanism_study")
    precision = Path("data/neutrino/mechanism_precision")
    continuation = Path("data/neutrino/precision_continuation")
    real_config = yaml.safe_load(Path("config/neutrino_pulses.yml").read_text())
    real = Path(real_config["data"]["output"])
    analytic = Path("data/neutrino/analytic_study")
    required = [
        mechanism / "test_results.json",
        precision / "test_results.json",
        continuation / "test_results.json",
        real / "test_results.json",
        mechanism / "downstream_probe/test_results.json",
    ]
    if not all(path.exists() for path in required):
        message = "Complete the sealed follow-up results before producing their final comparisons"
        raise ValueError(message)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    _mechanism_figures(mechanism, precision, continuation, output)
    _response_figure(analytic, output)
    _probe_figure(mechanism, output)
    _real_score_figure(real, output)
    examples = _real_events(real_config, output)
    write_json(
        output / "manifest.json",
        {
            "scope": "completed_followup_visualization_only_no_new_test_selection",
            "real_examples": examples,
            "GT_policy": (
                "Only the separate analytic generator supplies expected arrival PDF GT; "
                "real pulse B is an observation, not GT"
            ),
            "source_reports": [str(path) for path in required],
        },
    )
    page = (
        '<!doctype html><meta charset="utf-8"><title>Photon self-splitting follow-ups</title>'
        '<body style="max-width:1600px;margin:30px auto;padding:20px;font:16px system-ui">'
        "<h1>Mechanism, response and real-data comparisons</h1><p>Dataset-specific scores "
        "and units stay separate. These follow-ups are exploratory; physics labels are used "
        "only by a separate frozen-feature probe. Real data provide no clean photon-arrival GT.</p>"
    )
    for path in output.glob("*.png"):
        encoded = base64.b64encode(path.read_bytes()).decode()
        page += (
            f'<img style="width:100%;margin:20px 0" src="data:image/png;base64,{encoded}" '
            f'alt="{path.stem}">'
        )
    (output / "index.html").write_text(page + "</body>", encoding="utf-8")
    shutil.copyfile(Path(__file__), output / "visualization_source.py")
    with zipfile.ZipFile(
        output.parent / f"{output.name}.zip", "x", compression=zipfile.ZIP_DEFLATED
    ) as bundle:
        for path in output.iterdir():
            bundle.write(path, arcname=f"{output.name}/{path.name}")


def main():
    """Export the final scientific follow-up figures as PNG/SVG/HTML/ZIP artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("data/neutrino/study/followup_visualizations")
    )
    args = parser.parse_args()
    create_followup_figures(args.output)


if __name__ == "__main__":
    main()
