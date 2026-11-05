"""Scientific figures for development diagnosis, with no test-performance claims."""

from __future__ import annotations

import html

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"raw_prior": "#52677d", "self_split": "#cc6b29", "untrained": "#8c65b5"}
LABELS = {
    "raw_prior": "Raw + prior",
    "self_split": "Added self-split",
    "untrained": "Added untrained",
}
TIMING_LABELS = {
    "constant_training_mean": "Constant train mean",
    "raw_prior": "Raw + prior",
    "self_full": "Raw + self-split",
    "untrained_full": "Raw + untrained",
    "self_pdf_only": "Self-split PDF only",
    "untrained_pdf_only": "Untrained PDF only",
    "self_hidden_only": "Self-split hidden only",
    "untrained_hidden_only": "Untrained hidden only",
}


def _save(output, figure, name):
    for extension in ("png", "svg"):
        figure.savefig(output / f"{name}.{extension}", dpi=160, bbox_inches="tight")
    plt.close(figure)


def diagnostic_notes(config, result):
    """Describe the numerical observations without selecting or confirming a new method."""
    lines = [
        "Exploratory development diagnosis: training and validation only. "
        "No completed test or calibration outcome is read or changed.",
        "Selection ranges reuse the same validation windows and are not confidence intervals. "
        "Tail-preferred choices are optimistic development suggestions, "
        "not confirmed improvements.",
    ]
    for arm, result_arm in result["selection"].items():
        tail = result_arm["tail_selection"]["0.01"]
        lines.append(
            f"{LABELS[arm]}: log-loss penalty {tail['log_loss_preferred_penalty']:g}; "
            f"validation-tail-preferred penalty {tail['tail_preferred_penalty']:g}; "
            f"mean cross-calibrated two-photon efficiency difference "
            f"{100 * tail['tail_minus_log_loss_mean_efficiency']:+.2f} percentage points."
        )
    primary = config["timing"]["primary_penalty"]
    for method in (
        "constant_training_mean",
        "raw_prior",
        "self_full",
        "untrained_full",
        "self_pdf_only",
        "self_hidden_only",
    ):
        row = next(row for row in result["timing"][method] if row["penalty"] in (None, primary))
        lines.append(
            f"Timing {TIMING_LABELS[method]}: signal-mean validation MAE "
            f"{row['strata']['all']['mae_ns'][0]:.2f} ns."
        )
    coupling = result["coupling"]
    lines.append(
        f"Pairing intervention: {coupling['changed_windows']}/{coupling['selected_windows']} "
        "windows change a recorded-half joint map; each A/B sensor and time marginal is "
        "verified unchanged exactly."
    )
    for arm, values in coupling["methods"].items():
        changed = values["changed"]
        if changed["windows"]:
            lines.append(
                f"{LABELS[arm]} on changed windows: mean PDF TV "
                f"{changed['pdf_tv']['mean']:.6g}; hidden relative L2 "
                f"{changed['hidden_relative_l2']['mean']:.6g}; mean absolute logit response "
                f"raw {changed['absolute_raw_logit_change']['mean']:.6g}, "
                f"added {changed['absolute_extra_logit_change']['mean']:.6g}."
            )
    lines.append(
        "Sensitivity is not usefulness. The fixed clock, time averaging, small photon counts "
        "and nonlinear local features affect these diagnostics. Any revised score, architecture "
        "or pretraining comparison needs a new protocol and fresh held-out evaluation."
    )
    return "\n".join(lines)


def create_development_report(output, config, result):
    """Render metric tradeoffs, timing retention and pairing-response diagnostics."""
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout="constrained")
    for arm, color in COLORS.items():
        item = result["selection"][arm]
        candidates = item["candidates"]
        x = [row["penalty"] for row in candidates]
        loss = [row["validation_balanced_log_loss"] for row in candidates]
        selected = item["log_loss_selected_index"]
        axes[0].plot(x, loss, "o-", color=color, label=LABELS[arm])
        axes[0].scatter([x[selected]], [loss[selected]], marker="*", s=140, color=color)
        tail = item["tail_selection"]["0.01"]
        mean, low, high = (
            np.asarray(tail[name])
            for name in (
                "efficiency_by_penalty_mean",
                "efficiency_by_penalty_min",
                "efficiency_by_penalty_max",
            )
        )
        axes[1].plot(x, mean, "o-", color=color, label=LABELS[arm])
        axes[1].fill_between(x, low, high, alpha=0.15, color=color)
        axes[1].scatter([x[selected]], [mean[selected]], marker="*", s=140, color=color)
        failed = np.flatnonzero(~np.asarray(tail["candidate_passes_gradient_gate"]))
        if len(failed):
            axes[0].scatter(
                np.asarray(x)[failed], np.asarray(loss)[failed], marker="x", s=65, color="#b34b43"
            )
            axes[1].scatter(np.asarray(x)[failed], mean[failed], marker="x", s=65, color="#b34b43")
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xlabel("L2 penalty")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Balanced validation log loss")
    axes[1].set_ylabel("Validation efficiency at expected signal two")
    figure.suptitle("Development only: stars select log loss; shaded partition ranges are not CIs")
    figure.supxlabel(
        "Red crosses: original gradient gate failed; excluded from tail recommendations", fontsize=9
    )
    _save(output, figure, "selection_diagnosis")

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    methods = list(TIMING_LABELS)
    for target, axis in zip((0, 2), axes, strict=True):
        values = []
        for method in methods:
            row = next(
                row
                for row in result["timing"][method]
                if row["penalty"] in (None, config["timing"]["primary_penalty"])
            )
            values.append(row["strata"]["all"]["mae_ns"][target])
        axis.bar(
            np.arange(len(methods)),
            values,
            color=[
                "#8593a2",
                "#52677d",
                "#cc6b29",
                "#8c65b5",
                "#db9b61",
                "#b29acf",
                "#aa572a",
                "#6f489c",
            ],
        )
        axis.set_xticks(
            np.arange(len(methods)),
            [TIMING_LABELS[name] for name in methods],
            rotation=45,
            ha="right",
            fontsize=8,
        )
        axis.set_ylabel("Validation timing MAE (ns)")
        axis.set_title(("Expected signal mean", "Expected signal median")[target == 2])
    figure.suptitle(
        "Fixed ridge timing probes: positive injections, including zero recorded photons"
    )
    _save(output, figure, "timing_diagnosis")

    figure, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    arms = ("self_split", "untrained")
    for index, key in enumerate(("pdf_tv", "hidden_relative_l2")):
        values = [result["coupling"]["methods"][arm]["changed"][key]["mean"] for arm in arms]
        axes[index].bar(np.arange(2), values, color=[COLORS[arm] for arm in arms])
        axes[index].set_xticks([0, 1], [LABELS[arm] for arm in arms], rotation=20, ha="right")
        axes[index].set_ylabel(
            ("Mean probability total variation", "Mean relative hidden-feature L2")[index]
        )
    for index, arm in enumerate(arms):
        row = result["coupling"]["methods"][arm]["changed"]
        axes[2].bar(
            index - 0.16,
            row["absolute_raw_logit_change"]["mean"],
            width=0.3,
            color="#52677d",
            label="Raw feature contribution" if index == 0 else None,
        )
        axes[2].bar(
            index + 0.16,
            row["absolute_extra_logit_change"]["mean"],
            width=0.3,
            color=COLORS[arm],
            label="Added feature contribution" if index == 0 else None,
        )
    axes[2].set_xticks([0, 1], [LABELS[arm] for arm in arms], rotation=20, ha="right")
    axes[2].set_ylabel("Mean absolute logit change")
    axes[2].legend(fontsize=8)
    figure.suptitle("Changed validation windows: sensor/time marginals fixed, pairings shuffled")
    _save(output, figure, "coupling_diagnosis")
    notes = diagnostic_notes(config, result)
    (output / "evaluation_notes.txt").write_text(notes + "\n", encoding="utf-8")
    document = (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Development-only timing and selection diagnosis</title><style>"
        "body{max-width:1200px;margin:30px auto;padding:20px;font:17px system-ui;"
        "line-height:1.5;color:#213345}img{width:100%}.note{background:#fff0d5;padding:15px}"
        "a{color:#176b95}</style><h1>Development diagnosis: timing features and tail selection</h1>"
        '<div class="note"><b>Training and validation only. No new test claim.</b>'
        + "".join(f"<p>{html.escape(line)}</p>" for line in notes.splitlines())
        + "</div>"
        '<h2>Selection criterion</h2><a href="selection_diagnosis.svg">'
        '<img src="selection_diagnosis.png" alt="Validation log loss and tail efficiency"></a>'
        '<h2>Expected timing retained in features</h2><a href="timing_diagnosis.svg">'
        '<img src="timing_diagnosis.png" alt="Fixed auxiliary timing probes"></a>'
        '<h2>Joint sensor-time sensitivity</h2><a href="coupling_diagnosis.svg">'
        '<img src="coupling_diagnosis.png" alt="Pairing intervention and logit decomposition"></a>'
        '<p><a href="results.json">All diagnostic results</a> &middot; '
        '<a href="data_access.json">Development-only access log</a> &middot; '
        '<a href="protocol.txt">Diagnostic protocol</a> &middot; '
        '<a href="../study/results_index.html">Results hub</a></p></html>'
    )
    (output / "index.html").write_text(document, encoding="utf-8")
