"""Figures and a local viewer for the published-GAP detector transfer comparison."""

import csv
import html
import json
from pathlib import Path

import numpy as np
import torch

from . import neutrino_gap4d_retry as reference
from . import neutrino_published_gap as study

if __package__ == "src.tools":
    from ..ssunet.photon_reference import photon_count_log_probabilities
else:
    from ssunet.photon_reference import photon_count_log_probabilities

LABELS = {
    "conv4d": "4D GAP (ours)",
    "published_gap_paper": "Published GAP: paper setup",
    "published_gap_matched": "Published GAP: matched setup",
    "full_time_reference": "Full-time temporal control",
    "previously_tuned_mixture": "Frozen tuned mixture",
}
COLORS = {"conv4d": "#288765", "published_gap_paper": "#aa604c", "published_gap_matched": "#6b69b2"}


def comparison_rows(report):
    """Export both photon-weighted and nonempty-event-weighted comparisons."""
    return [
        {
            "method": name,
            "label": LABELS[name],
            "photon_nll_all": report["scores"][name]["all"]["nats_per_photon"],
            "photon_nll_low": report["scores"][name]["retained_1_to_10"]["nats_per_photon"],
            "event_cross_entropy_all": report["event_mean_nll"][name]["all"],
            "event_cross_entropy_low": report["event_mean_nll"][name]["retained_1_to_10"],
        }
        for name in LABELS
    ]


def _training_diagnostics(root, report, config):
    """Expose late-training degradation without turning it into checkpoint selection."""
    previous = json.loads((Path(config["reference_study"]) / "results.json").read_text())
    records, rows = {}, []
    for name in COLORS:
        source = previous if name == "conv4d" else report
        runs = [run for run in source["runs"] if run["model"] == name]
        values = np.asarray(
            [[point["retained_1_to_10"] for point in run["curve"]] for run in runs]
        ).mean(0)
        best = int(values.argmin())
        point = runs[0]["curve"][best]
        examples = point["step"] * runs[0].get("batch_size", 16)
        records[name] = {
            "best_recorded_mean_low_count_nll": float(values[best]),
            "training_examples_at_that_point": examples,
            "final_mean_low_count_nll": float(values[-1]),
            "scope": "retrospective curve diagnostic; no selected checkpoint or independent score",
        }
        rows.append(
            f"<tr><td>{LABELS[name]}</td><td>{values[best]:.4f}</td>"
            f"<td>{examples:,}</td><td>{values[-1]:.4f}</td></tr>"
        )
    (root / "learning_diagnostics.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )
    return (
        "<h2>Training-duration sensitivity</h2><p>The large GAP network's validation "
        "scores worsen late in training. Its best recorded mean low-count score is "
        "closer to 4D than its final score. The table below is a retrospective diagnostic "
        "over the saved curve points, not a selected-model result or an independent test. "
        "No paired interval is available for these earlier points. A superiority claim "
        "over a validation-selected or fully tuned GAP model remains unestablished.</p>"
        "<table><tr><th>Method</th><th>Best recorded mean low NLL</th>"
        "<th>Training examples at that point</th><th>Fixed-final low NLL</th></tr>"
        f"{''.join(rows)}</table>"
    )


def _learning_curves(root, report, config, plt):
    """Use training examples on the x axis to account for different batch sizes."""
    previous = json.loads((Path(config["reference_study"]) / "results.json").read_text())
    groups = {
        name: [run for run in source["runs"] if run["model"] == name]
        for name, source in (
            ("conv4d", previous),
            ("published_gap_paper", report),
            ("published_gap_matched", report),
        )
    }
    metrics = (
        ("all", "All held-out photons", "Nats / photon"),
        ("retained_1_to_10", "1-10 retained photons", "Nats / photon"),
        ("event_mean_all", "All nonempty target events", "Mean event cross entropy"),
        ("event_mean_low", "1-10 retained photons: equal events", "Mean event cross entropy"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    for axis, (metric, title, ylabel) in zip(axes.flat, metrics, strict=True):
        for name, runs in groups.items():
            batch_size = runs[0].get("batch_size", 16)
            examples = [point["step"] * batch_size for point in runs[0]["curve"]]
            values = np.asarray([[point[metric] for point in run["curve"]] for run in runs])
            mean, spread = values.mean(0), values.std(0)
            axis.plot(examples, mean, label=LABELS[name], color=COLORS[name])
            axis.fill_between(
                examples, mean - spread, mean + spread, color=COLORS[name], alpha=0.15
            )
        axis.set(title=title, xlabel="Training examples seen", ylabel=f"{ylabel} (lower better)")
        axis.legend(fontsize=8)
    figure.savefig(root / "learning_curves.png", dpi=160)
    figure.savefig(root / "learning_curves.svg")
    plt.close(figure)


def _example_predictions(root, config, view, positions, device):
    """Use the same input-count-defined examples and first seed in every model."""
    chosen = np.concatenate(
        [
            np.flatnonzero(view.strata[stratum])[:2]
            for stratum in ("retained_0", "retained_1_to_10", "retained_above_30")
        ]
    )
    previous_root = Path(config["reference_study"])
    previous = json.loads((previous_root / "config.json").read_text())
    settings = next(row for row in previous["models"] if row["name"] == "conv4d")
    models = [("conv4d", previous_root, settings)] + [
        (settings["name"], root, settings) for settings in config["models"]
    ]
    probabilities = {}
    for name, source, settings in models:
        factory = reference.make_model if name == "conv4d" else study.make_model
        model = factory(settings, positions).to(device)
        saved = torch.load(
            source / name / f"seed_{config['training']['seeds'][0]}" / "last.pt",
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(saved["state_dict"])
        model.eval()
        with torch.inference_mode():
            logits = model(
                view.inputs[chosen].to(device),
                view.positions_m.to(device),
                view.live.to(device),
                torch.full((len(chosen),), view.p, device=device),
            )
            probabilities[name] = (
                photon_count_log_probabilities(logits, view.live.to(device)).exp().cpu().numpy()
            )
    np.savez_compressed(
        root / "arrival_examples.npz",
        event_ids=view.event_ids[chosen],
        input=view.inputs[chosen].numpy(),
        held_out=view.targets[chosen].numpy(),
        **probabilities,
    )
    return chosen, probabilities


def _arrival_examples(root, config, view, positions, device, plt):
    """Plot arrival-time marginals with raw A/B counts on a separate axis."""
    chosen, predictions = _example_predictions(root, config, view, positions, device)
    manifest = json.loads((Path(config["reference_study"]) / "manifest.json").read_text())
    width = manifest["time_bin_width_ns"]
    times = manifest["time_window_ns"][0] + (np.arange(view.inputs.shape[-1]) + 0.5) * width
    figure, axes = plt.subplots(
        len(chosen), 1, figsize=(12, 2.6 * len(chosen)), squeeze=False, layout="constrained"
    )
    for row, index in enumerate(chosen):
        axis = axes[row, 0]
        for name, q in predictions.items():
            axis.plot(times, q[row].sum(0), color=COLORS[name], label=LABELS[name])
        raw = axis.twinx()
        observed = view.inputs[index].sum(0).numpy()
        held_out = view.targets[index].sum(0).numpy()
        raw.scatter(
            times[observed > 0],
            observed[observed > 0],
            marker="|",
            color="#8b939b",
            label="A counts",
        )
        raw.scatter(
            times[held_out > 0],
            held_out[held_out > 0],
            marker="x",
            color="#30383d",
            s=18,
            label="B counts (noisy held-out photons)",
        )
        axis.set(
            xlabel="Arrival time (ns)",
            ylabel=f"Probability per {width:g} ns bin",
            title=(
                f"Event {view.event_ids[index]}: A={int(observed.sum())}, B={int(held_out.sum())}"
            ),
        )
        axis.set_ylim(bottom=0)
        raw.set_ylim(0, 1.15 * max(1, int(observed.max()), int(held_out.max())))
        raw.set_ylabel("Observed photon counts")
        handles, labels = axis.get_legend_handles_labels()
        other, other_labels = raw.get_legend_handles_labels()
        axis.legend(handles + other, labels + other_labels, fontsize=7)
    figure.savefig(root / "arrival_examples.png", dpi=160)
    plt.close(figure)


def _contrast_table(report):
    """Show each configuration's paired result without choosing a winning setup."""
    rows = []
    for name, strata in report["conv4d_minus_published"].items():
        for stratum, label in (("retained_1_to_10", "Low counts"), ("all", "All photons")):
            value = strata[stratum]
            low, high = value["descriptive_95_percent_interval"]
            rows.append(
                f"<tr><td>{LABELS[name]}</td><td>{label}</td>"
                f"<td>{value['model_minus_control_nats_per_photon']:+.4f}</td>"
                f"<td>[{low:+.4f}, {high:+.4f}]</td></tr>"
            )
    return (
        "<h2>4D minus published GAP</h2><p>Negative differences favor 4D. "
        "Intervals resample whole persisted detector configurations and condition on the "
        "fitted models. They omit seed and selection uncertainty.</p>"
        "<table><tr><th>Published setup</th><th>Observations</th><th>NLL difference</th>"
        f"<th>Descriptive 95% interval</th></tr>{''.join(rows)}</table>"
    )


def _cost_table(report, config):
    """Compare actual resource use at the same training-example budget."""
    previous = json.loads((Path(config["reference_study"]) / "results.json").read_text())
    rows = []
    for name in COLORS:
        runs = [run for run in report["runs"] + previous["runs"] if run["model"] == name]
        seconds = np.mean([run["training_seconds"] for run in runs])
        peak = max(run["peak_gpu_allocated_gib"] for run in runs)
        rows.append(
            f"<tr><td>{LABELS[name]}</td><td>{runs[0]['parameters']:,}</td>"
            f"<td>{seconds:.1f}</td><td>{peak:.3f}</td></tr>"
        )
    return (
        "<h2>Training cost</h2><p>All models see 80,000 training examples per seed. "
        "Times exclude validation/checkpoint overhead and use shared GPU hardware. "
        "The paper setup uses batch 32; the other setups use batch 16.</p>"
        "<table><tr><th>Method</th><th>Parameters</th><th>Mean training seconds/seed</th>"
        f"<th>Peak allocated GPU GiB</th></tr>{''.join(rows)}</table>"
    )


def render_comparison(root, report, config, view, positions, device):
    """Export the complete released-network comparison with explicit domain adaptations."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _learning_curves(root, report, config, plt)
    _arrival_examples(root, config, view, positions, device, plt)
    rows = comparison_rows(report)
    with (root / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    table = "".join(
        f"<tr><td>{row['label']}</td><td>{row['photon_nll_low']:.4f}</td>"
        f"<td>{row['photon_nll_all']:.4f}</td><td>{row['event_cross_entropy_low']:.4f}</td>"
        f"<td>{row['event_cross_entropy_all']:.4f}</td></tr>"
        for row in rows
    )
    limits = "".join(f"<li>{html.escape(value)}</li>" for value in report["limitations"])
    low = report["scores"]["conv4d"]["retained_1_to_10"]
    page = (
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<title>Published GAP versus 4D</title>"
        "<style>body{font:16px system-ui;max-width:1250px;margin:35px auto;padding:0 20px;"
        "line-height:1.6}img{width:100%}td,th{padding:8px 14px;text-align:left}"
        "table{border-collapse:collapse}tr{border-bottom:1px solid #ddd}</style>"
        "<h1>Published GAP network versus 4D photon prediction</h1>"
        "<p>The released GAP residual U-Net is used unchanged at its paper dimensions: "
        "six levels, 28 initial channels, ten raw-count sine scales and 33,443,425 parameters. "
        "The same 800 training and 100 validation P-ONE simulation observations are used. "
        "Three seeds and 80,000 training examples per seed are shared across setups.</p>"
        "<p><a href='https://github.com/krulllab/GAP/tree/135b8c67ce5a6ab4c44bcdd0fbe756f75526e249'>"
        "Pinned official code</a> &middot; "
        "<a href='https://arxiv.org/html/2307.06607v2'>GAP paper</a></p>"
        "<p><strong>Detector-data transfer:</strong> observations become geometry-ordered "
        "sensor/time images. Padding is removed before probability normalization. Whole events "
        "replace microscopy crops, image augmentation is omitted, and p=.5 is fixed. "
        "This comparison does not reproduce the paper's microscopy benchmark.</p>"
        "<h2>Final distribution scores</h2><p>Lower is better. Photon NLL weights photons "
        "equally; event cross entropy gives each nonempty target event equal weight. "
        "Predictions are joint sensor/time probability distributions.</p>"
        "<table><tr><th>Method</th><th>Low-count photon NLL</th><th>All-photon NLL</th>"
        f"<th>Low-count event CE</th><th>All-event CE</th></tr>{table}</table>"
        f"<p>The 1-10 retained-photon subset contains {low['events']} observations, "
        f"{low['events_with_target']} nonempty targets and {low['held_out_photons']} held-out "
        "photons. These are reused validation observations; no clean photon GT, PSNR, "
        "reserved test records or MC truth are used.</p>"
        f"{_training_diagnostics(root, report, config)}"
        f"{_contrast_table(report)}"
        "<h2>Learning curves</h2><p>Curves use training examples seen because batch sizes "
        "differ. Shading shows one standard deviation across seeds.</p>"
        "<img src='learning_curves.png' alt='Distribution scores versus training examples'>"
        f"{_cost_table(report, config)}"
        "<h2>Arrival distributions</h2><p>Fixed input-stratum and record-order selection, "
        "using seed 42 for every model. Crosses are noisy held-out photons. "
        "Empty-input predictions are priors. There is no clean distribution ground truth.</p>"
        "<img src='arrival_examples.png' alt='Arrival probability marginals and noisy A/B counts'>"
        f"<h2>Adaptations and limits</h2><ul>{limits}</ul>"
        "<p>The original image notebook uses virtual epochs of 16,000 random crops. "
        "Here 100 detector-data epochs mean 80,000 total examples; the complete original "
        "image-training schedule is not reproduced.</p>"
        "<p><a href='results.json'>Full numerical results</a> &middot; "
        "<a href='comparison.csv'>Comparison CSV</a> &middot; "
        "<a href='evidence_audit.json'>Checkpoint and source audit</a> &middot; "
        "<a href='source_snapshot.json'>Frozen source snapshot</a> &middot; "
        "<a href='protocol.txt'>Protocol</a> &middot; "
        "<a href='learning_diagnostics.json'>Training-duration diagnostic</a> &middot; "
        "<a href='arrival_examples.npz'>Example arrays</a></p></html>"
    )
    (root / "index.html").write_text(page, encoding="utf-8")
