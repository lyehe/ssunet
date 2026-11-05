"""Report a completed frozen-feature probe without changing fitting or calibration."""

from __future__ import annotations

import html

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = {
    "raw_prior": "Head: raw + prior",
    "self_split": "Head: raw + prior + self-split",
    "untrained": "Head: raw + prior + untrained",
    "physics_template": "Physics-template search",
    "static_prior": "Static prior statistic",
    "frozen_score": "Previous frozen-model statistic",
    "geometry_kernel": "Geometry/time kernel",
    "count": "Count only",
}
COLORS = {
    "raw_prior": "#52677d",
    "self_split": "#cc6b29",
    "untrained": "#8c65b5",
    "physics_template": "#2f935c",
}


def _save(output, figure, name):
    for extension in ("png", "svg"):
        figure.savefig(output / f"{name}.{extension}", dpi=160, bbox_inches="tight")
    plt.close(figure)


def _primary(config, result):
    methods = result["operating_points"][str(config["evaluation"]["primary_false_positive_rate"])]
    strength = config["evaluation"]["primary_expected_signal"]
    rows = {
        name: next(row for row in entry["efficiency"] if row["expected_signal"] == strength)
        for name, entry in methods.items()
    }
    return methods, rows


def _interpretation(config, result):
    methods, rows = _primary(config, result)
    comparisons = []
    for name in ("raw_prior", "untrained"):
        row = rows[name]
        lo, hi = row["difference_interval"]
        comparisons.append(
            f"Self-split head minus {LABELS[name]}: {100 * row['model_minus_control']:+.2f} "
            f"percentage points, conditional 95% interval [{100 * lo:+.2f}, {100 * hi:+.2f}]."
        )
    positive = all(rows[name]["difference_interval"][0] > 0 for name in ("raw_prior", "untrained"))
    if positive:
        decision = (
            "This exploratory frozen-feature probe supports further matched-pretraining-objective "
            "testing: both registered feature-control contrasts are positive. It does not isolate "
            "self-splitting from other pretraining objectives or validate real event recovery."
        )
    else:
        decision = (
            "This probe does not establish an added detection benefit over both registered "
            "feature controls. Preserve this test and diagnose using development data; "
            "the result also depends on this linear head, features and validation selection rule."
        )
    measured = ", ".join(
        f"{LABELS[name]} {100 * methods[name]['false_positive_rate']:.3f}%"
        for name in ("raw_prior", "self_split", "untrained", "physics_template")
    )
    return "\n".join(
        [
            decision,
            *comparisons,
            f"Measured test false-positive rates: {measured}.",
            "Intervals condition on frozen encoders, selected heads and background calibration; "
            "they exclude procedure/model fitting and calibration uncertainty. "
            "Rates are not exactly equal.",
            "The small head uses injection-presence labels; "
            "encoder pretraining remains self-supervised. No clean photon reference, "
            "physical posterior or decoded event parameters are asserted.",
        ]
    )


def create_probe_report(output, config, result, selections, probabilities, truth):
    """Export fixed primary/secondary comparisons, actual false alarms and fit records."""
    methods, rows = _primary(config, result)
    alpha = config["evaluation"]["primary_false_positive_rate"]
    strength = config["evaluation"]["primary_expected_signal"]
    levels = config["generator"]["signal_expected_photons"]
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    for name, color in COLORS.items():
        values = methods[name]["efficiency"]
        efficiency = np.asarray([row["efficiency"] for row in values])
        ci = np.asarray([row["interval"] for row in values])
        axes[0].plot(levels, efficiency, "o-", color=color, label=LABELS[name], markersize=4)
        axes[0].fill_between(levels, ci[:, 0], ci[:, 1], color=color, alpha=0.1)
    for name in ("raw_prior", "untrained"):
        values = methods[name]["efficiency"]
        delta = np.asarray([row["model_minus_control"] for row in values])
        ci = np.asarray([row["difference_interval"] for row in values])
        axes[1].plot(
            levels, delta, "o-", color=COLORS[name], label=f"Self-split minus {LABELS[name]}"
        )
        axes[1].fill_between(levels, ci[:, 0], ci[:, 1], color=COLORS[name], alpha=0.15)
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xlabel("Expected signal photons / window")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("Detection efficiency")
    axes[0].set_ylim(0, 1.02)
    axes[1].set_ylabel("Paired efficiency difference (conditional 95%)")
    axes[1].axhline(0, color="black", linewidth=0.7)
    figure.suptitle(f"New frozen-feature test: target false-positive rate {alpha:g} / window")
    _save(output, figure, "probe_efficiency")

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    for name, color in COLORS.items():
        entry = methods[name]
        index = list(COLORS).index(name)
        rate, ci = entry["false_positive_rate"], entry["false_positive_interval"]
        axes[0].errorbar(index, rate, yerr=[[rate - ci[0]], [ci[1] - rate]], fmt="o", color=color)
    axes[0].axhline(alpha, linestyle="--", color="black", linewidth=0.8)
    axes[0].set_xticks(
        np.arange(len(COLORS)), [LABELS[name] for name in COLORS], rotation=25, ha="right"
    )
    axes[0].set_ylabel("Test false-positive rate / window\nWilson 95% interval")
    null = truth["expected_signal"] == 0
    signal = truth["expected_signal"] == strength
    grid = np.geomspace(0.0001, 0.1, 160)
    for name, color in COLORS.items():
        values = probabilities[name]
        fpr = [(values[null] <= level).mean() for level in grid]
        efficiency = [(values[signal] <= level).mean() for level in grid]
        axes[1].plot(fpr, efficiency, color=color, label=LABELS[name])
    axes[1].set_xscale("log")
    axes[1].set_xlim(0.0001, 0.1)
    axes[1].set_xlabel("Measured test false-positive rate / window")
    axes[1].set_ylabel(f"Efficiency at expected signal {strength:g}")
    axes[1].legend(fontsize=8)
    figure.suptitle("Independent background check and low-FPR ROC; no exposure-based rate claim")
    _save(output, figure, "probe_false_alarms")

    figure, axis = plt.subplots(figsize=(7.5, 4), layout="constrained")
    for name in ("raw_prior", "self_split", "untrained"):
        candidates = selections[name]["candidates"]
        x = [row["penalty"] for row in candidates]
        y = [row["validation_balanced_log_loss"] for row in candidates]
        axis.plot(x, y, "o-", color=COLORS[name], label=LABELS[name])
        selected = selections[name]["selected_candidate"]
        axis.scatter([x[selected]], [y[selected]], marker="*", s=180, color=COLORS[name])
    axis.set_xscale("log")
    axis.set_xlabel("L2 penalty (fixed candidate grid)")
    axis.set_ylabel("Balanced validation log loss")
    axis.legend(fontsize=8)
    axis.set_title("Selection uses validation only; stars identify frozen choices")
    _save(output, figure, "probe_selection")
    interpretation = _interpretation(config, result)
    (output / "evaluation_notes.txt").write_text(interpretation + "\n", encoding="utf-8")
    table = []
    for name, label in LABELS.items():
        entry, row = methods[name], rows[name]
        ci = row["interval"]
        difference = "&mdash;"
        if name != "self_split":
            lo, hi = row["difference_interval"]
            difference = (
                f"{100 * row['model_minus_control']:+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}]"
            )
        table.append(
            f"<tr><td>{html.escape(label)}</td>"
            f"<td>{100 * entry['false_positive_rate']:.3f}%</td>"
            f"<td>{100 * row['efficiency']:.2f}% [{100 * ci[0]:.2f}, {100 * ci[1]:.2f}]</td>"
            f"<td>{difference}</td></tr>"
        )
    notes = "".join(f"<p>{html.escape(line)}</p>" for line in interpretation.splitlines())
    document = (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Frozen-feature detection probe</title><style>"
        "body{max-width:1200px;margin:30px auto;padding:20px;font:17px system-ui;"
        "line-height:1.5;color:#213345}img{width:100%}td,th{padding:9px;text-align:left;"
        "border-bottom:1px solid #ddd}table{border-collapse:collapse}.note{background:"
        "#fff0d5;padding:15px}a{color:#176b95}</style>"
        "<h1>Do frozen self-splitting features add detection information?</h1>"
        '<div class="note"><b>Fresh exact-Poisson toy, labelled linear heads.</b>'
        + notes
        + "</div>"
        f"<p>{result['test_windows']:,} new test windows; "
        f"{result['test_sources']} source groups excluded from every earlier study and fold. "
        f"All three heads have {result['head_parameters_each']:,} coefficients including the "
        "intercept. Raw/prior uses zero padding; the random-feature arm controls for the "
        "additional varying feature capacity. "
        "Both recorded halves are available to all methods.</p>"
        f"<h2>Expected signal {strength:g}; "
        f"nominal false-positive rate {100 * alpha:g}%</h2>"
        "<table><tr><th>Method</th><th>Measured false-positive rate</th>"
        "<th>Efficiency [conditional 95%]</th><th>Self-split head minus reference (pp)</th></tr>"
        + "".join(table)
        + "</table>"
        '<h2>Efficiency and paired feature controls</h2><a href="probe_efficiency.svg">'
        '<img src="probe_efficiency.png" alt="Frozen feature head comparison"></a>'
        '<h2>False alarms and ROC</h2><a href="probe_false_alarms.svg">'
        '<img src="probe_false_alarms.png" alt="Background false alarms and low FPR ROC"></a>'
        '<h2>Validation-only head selection</h2><a href="probe_selection.svg">'
        '<img src="probe_selection.png" alt="All fixed regularization candidates"></a>'
        '<p><a href="test_results.json">All numerical results</a> &middot; '
        '<a href="detection_results.csv">Comparison CSV</a> &middot; '
        '<a href="head_selection.json">Selection and convergence</a> &middot; '
        '<a href="protocol.txt">Fixed protocol</a> &middot; '
        '<a href="analysis_lock.json">Head/calibration lock</a> &middot; '
        '<a href="../detection_study_v1/index.html">Earlier fixed detector and GT examples</a> '
        '&middot; <a href="../study/results_index.html">Results hub</a></p></html>'
    )
    (output / "index.html").write_text(document, encoding="utf-8")
