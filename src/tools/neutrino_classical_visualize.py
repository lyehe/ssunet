"""Comparison figures for training-selected traditional photon estimators."""

import csv
import html
import json
from pathlib import Path

import h5py
import numpy as np

LABELS = {
    "conv4d": "4D GAP (ours)",
    "published_gap_matched": "Released GAP: matched setup",
    "published_gap_paper": "Released GAP: paper setup",
    "full_time_reference": "Full-time temporal control",
    "previously_tuned_mixture": "Historical tuned mixture",
    "dirichlet": "Bayesian Dirichlet shrinkage",
    "temporal_kde": "Temporal Gaussian KDE",
    "spatiotemporal_kde": "Geometry-aware spatial/time KDE",
    "poisson_tv": "Poisson-TV + prior shrinkage",
    "population_prior": "Training-selected population prior",
}
COLORS = {
    "conv4d": "#278363",
    "published_gap_matched": "#7569ab",
    "dirichlet": "#8c8172",
    "temporal_kde": "#397da4",
    "spatiotemporal_kde": "#c18c3b",
    "poisson_tv": "#ad5e59",
}


def comparison_rows(report):
    """Keep all declared estimators and both score weightings in the exported table."""
    return [
        {
            "method": name,
            "label": label,
            "photon_nll_low": report["scores"][name]["retained_1_to_10"]["nats_per_photon"],
            "photon_nll_all": report["scores"][name]["all"]["nats_per_photon"],
            "event_ce_low": report["event_mean_nll"][name]["retained_1_to_10"],
            "event_ce_all": report["event_mean_nll"][name]["all"],
        }
        for name, label in LABELS.items()
    ]


def _comparison_plot(root, rows, plt):
    """Show absolute photon NLL on zero-based axes for the shared observation cohort."""
    figure, axes = plt.subplots(1, 2, figsize=(14, 7), layout="constrained")
    y = np.arange(len(rows))
    colors = [COLORS.get(row["method"], "#9da5ac") for row in rows]
    for axis, key, title in zip(
        axes,
        ("photon_nll_low", "photon_nll_all"),
        ("1-10 retained photons", "All photons"),
        strict=True,
    ):
        values = [row[key] for row in rows]
        axis.barh(y, values, color=colors)
        axis.set_yticks(y, labels=[row["label"] for row in rows], fontsize=8)
        axis.invert_yaxis()
        axis.set(
            title=title,
            xlabel="Nats / held-out photon (lower better)",
            xlim=(0, max(values) * 1.13),
        )
        for row, value in enumerate(values):
            axis.text(value + 0.04, row, f"{value:.3f}", va="center", fontsize=8)
    figure.savefig(root / "comparison.png", dpi=160)
    figure.savefig(root / "comparison.svg")
    plt.close(figure)


def _arrival_examples(root, config, view, probabilities, plt):
    """Reuse the original input-defined examples and archived neural probability fields."""
    source = Path(config["reference_study"]) / "arrival_examples.npz"
    with np.load(source) as neural:
        indices = np.array(
            [int(np.flatnonzero(view.event_ids == event)[0]) for event in neural["event_ids"]]
        )
        if not np.array_equal(neural["input"], view.inputs[indices].numpy()) or not np.array_equal(
            neural["held_out"], view.targets[indices].numpy()
        ):
            message = "Classical and archived neural example observations differ"
            raise ValueError(message)
        fields = {
            "conv4d": neural["conv4d"].copy(),
            "published_gap_matched": neural["published_gap_matched"].copy(),
            **{name: probabilities[name][indices] for name in config["families"]},
        }
    with h5py.File(config["dataset"], "r") as data:
        edges = data["time_edges_ns"][:]
    times = (edges[:-1] + edges[1:]) / 2
    width = float(edges[1] - edges[0])
    figure, axes = plt.subplots(
        len(indices), 1, figsize=(12, len(indices) * 2.6), squeeze=False, layout="constrained"
    )
    for row, index in enumerate(indices):
        axis = axes[row, 0]
        for name, field in fields.items():
            axis.plot(times, field[row].sum(0), label=LABELS[name], color=COLORS[name])
        counts = axis.twinx()
        a = view.inputs[index].sum(0).numpy()
        b = view.targets[index].sum(0).numpy()
        counts.scatter(times[a > 0], a[a > 0], marker="|", color="#86909a", label="A counts")
        counts.scatter(
            times[b > 0], b[b > 0], marker="x", color="#222b32", s=18, label="B counts (noisy)"
        )
        axis.set(xlabel="Arrival time (ns)", ylabel=f"Probability per {width:g} ns bin")
        axis.set_title(f"Event {view.event_ids[index]}: A={int(a.sum())}, B={int(b.sum())}")
        axis.set_ylim(bottom=0)
        counts.set_ylabel("Observed photon counts")
        counts.set_ylim(0, 1.15 * max(1, int(a.max()), int(b.max())))
        handles, labels = axis.get_legend_handles_labels()
        other, other_labels = counts.get_legend_handles_labels()
        axis.legend(handles + other, labels + other_labels, fontsize=6)
    figure.savefig(root / "arrival_examples.png", dpi=160)
    plt.close(figure)
    np.savez_compressed(root / "arrival_examples.npz", event_ids=view.event_ids[indices], **fields)


def _contrast_table(report):
    """Display conditional paired differences with their actual resampling unit."""
    rows = []
    for name, strata in report["conv4d_minus_classical"].items():
        for stratum, label in (("retained_1_to_10", "Low counts"), ("all", "All photons")):
            value = strata[stratum]
            low, high = value["descriptive_95_percent_interval"]
            rows.append(
                f"<tr><td>{LABELS[name]}</td><td>{label}</td>"
                f"<td>{value['model_minus_control_nats_per_photon']:+.4f}</td>"
                f"<td>[{low:+.4f}, {high:+.4f}]</td></tr>"
            )
    return (
        "<h2>4D minus classical estimator</h2><p>Negative differences favor 4D. "
        "Intervals resample whole persisted configurations and condition on the fitted "
        "estimators, omitting selection and model-seed uncertainty.</p>"
        "<table><tr><th>Classical estimator</th><th>Observations</th><th>NLL difference</th>"
        f"<th>Descriptive 95% interval</th></tr>{''.join(rows)}</table>"
    )


def render_comparison(root, report, config, view, probabilities):
    """Export every declared classical result and its training-only selected parameters."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = comparison_rows(report)
    _comparison_plot(root, rows, plt)
    _arrival_examples(root, config, view, probabilities, plt)
    with (root / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    table = "".join(
        f"<tr><td>{row['label']}</td><td>{row['photon_nll_low']:.4f}</td>"
        f"<td>{row['photon_nll_all']:.4f}</td><td>{row['event_ce_low']:.4f}</td>"
        f"<td>{row['event_ce_all']:.4f}</td></tr>"
        for row in rows
    )
    settings = "".join(
        f"<tr><td>{LABELS[name]}</td><td><code>"
        f"{html.escape(json.dumps(value['settings'], sort_keys=True))}</code></td>"
        f"<td>{value['inner_event_ce']:.4f}</td></tr>"
        for name, value in report["selection"]["estimators"].items()
    )
    low = report["scores"]["conv4d"]["retained_1_to_10"]
    selection = report["selection"]
    limits = "".join(f"<li>{html.escape(value)}</li>" for value in report["limitations"])
    page = (
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<title>Traditional versus neural photon estimators</title>"
        "<style>body{font:16px system-ui;max-width:1300px;margin:35px auto;padding:0 20px;"
        "line-height:1.6}img{width:100%}td,th{padding:8px 12px;text-align:left}"
        "table{border-collapse:collapse}tr{border-bottom:1px solid #ddd}"
        "code{font-size:12px}</style>"
        "<h1>Traditional versus neural photon-arrival prediction</h1>"
        "<p>Bayesian Dirichlet shrinkage, temporal Gaussian KDE, geometry-aware Gaussian "
        "KDE and Poisson-TV are compared with the frozen neural predictors. Every output "
        "is one normalized distribution over live detector/time bins. There is no clean "
        "photon GT, PSNR, test-photon access or absolute event-rate inference.</p>"
        "<h2>Training-only parameter selection</h2>"
        f"<p>Population priors were fitted on {selection['fit_events']} training observations "
        f"and parameters selected on {selection['tuning_events']} separate training "
        "observations, keeping configuration groups disjoint. Settings were locked before "
        "the 100 outer validation observations were opened. After selection the prior was "
        "refitted on all 800 training observations. Selection minimizes cross entropy averaged "
        "over nonempty inner-training targets.</p>"
        "<p>Selected population prior: "
        f"<code>{html.escape(json.dumps(selection['prior']))}</code>. "
        f"Evaluated inner combinations: {selection['tuning_trials']}. "
        f"Selection took {report['training_only_selection_seconds']:.2f} seconds "
        "on this system.</p><table><tr><th>Estimator</th><th>Locked settings</th>"
        f"<th>Inner event CE</th></tr>{settings}</table>"
        "<h2>Outer validation scores</h2><p>Lower is better. Photon NLL weights photons "
        "equally; event CE gives each nonempty target observation one vote. Neural scores "
        "are averaged over their paired training seeds; classical estimates are deterministic.</p>"
        "<table><tr><th>Method</th><th>Low-count photon NLL</th><th>All-photon NLL</th>"
        f"<th>Low-count event CE</th><th>All-event CE</th></tr>{table}</table>"
        f"<p>The retained-A 1-10 count subset has {low['events']} observations, "
        f"{low['events_with_target']} nonempty targets and {low['held_out_photons']} held-out "
        "photons. The same previously explored simulation validation cohort is reused.</p>"
        "<img src='comparison.png' alt='All declared traditional and neural photon NLL scores'>"
        f"{_contrast_table(report)}"
        "<h2>Arrival-time probability marginals</h2><p>Examples use the same input-defined "
        "strata and record order as the published-method comparison. Neural examples use "
        "the same fixed seed 42. Crosses show noisy held-out B counts; empty-input estimates "
        "are population priors.</p>"
        "<img src='arrival_examples.png' "
        "alt='Traditional and neural arrival distributions with noisy counts'>"
        f"<h2>Scope and numerical limits</h2><ul>{limits}</ul>"
        "<p><a href='results.json'>Numerical results</a> &middot; "
        "<a href='comparison.csv'>Comparison CSV</a> &middot; "
        "<a href='selection_locked.json'>Locked settings</a> &middot; "
        "<a href='inner_split.json'>Fit/tuning groups</a> &middot; "
        "<a href='tuning_trials.json'>All inner trials</a> &middot; "
        "<a href='evidence_audit.json'>Prediction audit</a> &middot; "
        "<a href='source_snapshot.json'>Source snapshot</a> &middot; "
        "<a href='protocol.txt'>Protocol and sources</a></p></html>"
    )
    (root / "index.html").write_text(page, encoding="utf-8")
