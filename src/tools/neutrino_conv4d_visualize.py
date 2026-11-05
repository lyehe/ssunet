"""Figures for the fixed 4D photon retry, separate from training identities."""

import csv
import json

import numpy as np
import torch

if __package__ == "src.tools":
    from ..ssunet.photon_reference import photon_count_log_probabilities
else:
    from ssunet.photon_reference import photon_count_log_probabilities


def _render_event_metrics(root, report, colors, plt):
    """Add diagnostics that weight each nonempty target event equally."""
    event_table = ""
    event_figure = ""
    if "event_mean_nll" in report:
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
        for axis, stratum, curve_key in zip(
            axes,
            ("all", "retained_1_to_10"),
            ("event_mean_all", "event_mean_low"),
            strict=True,
        ):
            for name, color in colors.items():
                curves = [run["curve"] for run in report["runs"] if run["model"] == name]
                steps = [point["step"] for point in curves[0]]
                values = np.asarray([[point[curve_key] for point in curve] for curve in curves])
                mean, spread = values.mean(0), values.std(0)
                axis.plot(steps, mean, label=name, color=color)
                axis.fill_between(steps, mean - spread, mean + spread, color=color, alpha=0.14)
            for name, style in (("fixed_mixture", ":"), ("previously_tuned_mixture", "--")):
                axis.axhline(
                    report["event_mean_nll"][name][stratum],
                    linestyle=style,
                    color="#60656c",
                    label=name,
                )
            axis.set(
                xlabel="Matched optimizer updates",
                ylabel="Mean event cross entropy (nats; lower better)",
                title="All nonempty targets" if stratum == "all" else "1-10 retained photons",
            )
            axis.legend(fontsize=7)
        figure.savefig(root / "event_learning_curves.png", dpi=160)
        plt.close(figure)
        event_rows = "".join(
            f"<tr><td>{name}</td><td>{values['all']:.4f}</td>"
            f"<td>{values['retained_1_to_10']:.4f}</td></tr>"
            for name, values in report["event_mean_nll"].items()
            if not name.endswith("_donor")
        )
        event_table = (
            "<h2>Equal event weighting</h2><p>Each nonempty held-out target contributes "
            "one cross entropy, matching GAP's target weighting. Empty targets do not "
            "have an observed cross entropy. The table above weights each held-out "
            "photon equally. Both tables average scores over seeds.</p>"
            "<table><tr><th>Architecture / control</th><th>All nonempty targets</th>"
            f"<th>1-10 retained photons</th></tr>{event_rows}</table>"
        )
        event_figure = "<img src='event_learning_curves.png' alt='Equal event learning curves'>"
    return event_table, event_figure


def _method_note(report, manifest):
    """Describe the declared acquisition policy and the actual fitted objective."""
    if not report.get("method"):
        return ""
    observation = (
        "multi-photon counts"
        if manifest["observation_kind"] == "photon_counts"
        else "one-bit counts"
    )
    return (
        f"<p><strong>Method: {report['method'].upper()}.</strong> "
        "Declared one-bit acquisitions select Bit2Bit; photon-count acquisitions "
        f"select GAP. This dataset contains {observation}. This run uses "
        "ten-scale sine inputs and per-event unit-mean targets. Outputs are a joint "
        "unit-sum probability distribution over live sensor/time bins. "
        "The small detector backbones and optimizer are adaptations of the papers.</p>"
        "<p><a href='../conv4d_retry_v1/index.html'>Archived count-weighted comparison</a> "
        "(different input encoding and training weights).</p>"
    )


def _comparison_rows(report):
    """Export both score weightings when the paper-aligned report provides them."""
    rows = []
    for name, values in report["scores"].items():
        if name.endswith("_donor"):
            continue
        row = {
            "method": name,
            **{
                stratum: values[stratum]["nats_per_photon"]
                for stratum in ("all", "retained_1_to_10", "retained_0")
            },
        }
        if "event_mean_nll" in report:
            row.update(
                {
                    f"{stratum}_event_cross_entropy": report["event_mean_nll"][name][stratum]
                    for stratum in ("all", "retained_1_to_10", "retained_0")
                }
            )
        rows.append(row)
    return rows


def render_results(root, report, config, view, positions, device, model_factory):
    """Export directly comparable likelihood curves and noisy-observation examples."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    colors = {"dense_reference": "#497eae", "full_time_reference": "#c38935", "conv4d": "#288765"}
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for axis, stratum in zip(axes, ("all", "retained_1_to_10"), strict=True):
        for name, color in colors.items():
            curves = [run["curve"] for run in report["runs"] if run["model"] == name]
            steps = [point["step"] for point in curves[0]]
            values = np.asarray([[point[stratum] for point in curve] for curve in curves])
            mean, spread = values.mean(0), values.std(0)
            axis.plot(steps, mean, label=name, color=color)
            axis.fill_between(steps, mean - spread, mean + spread, color=color, alpha=0.14)
        for name, style in (("fixed_mixture", ":"), ("previously_tuned_mixture", "--")):
            axis.axhline(
                report["scores"][name][stratum]["nats_per_photon"],
                linestyle=style,
                color="#60656c",
                label=name,
            )
        axis.set(
            xlabel="Matched optimizer updates",
            ylabel="Nats / held-out photon (lower better)",
            title="All validation photons" if stratum == "all" else "1-10 retained photons",
        )
        axis.legend(fontsize=7)
    figure.savefig(root / "learning_curves.png", dpi=160)
    figure.savefig(root / "learning_curves.svg")
    plt.close(figure)
    event_table, event_figure = _render_event_metrics(root, report, colors, plt)
    chosen = np.concatenate(
        [
            np.flatnonzero(view.strata[stratum])[:2]
            for stratum in ("retained_0", "retained_1_to_10", "retained_above_30")
        ]
    )
    predictions = {}
    for settings in config["models"]:
        model = model_factory(settings, positions).to(device)
        checkpoint = torch.load(
            root / settings["name"] / f"seed_{config['training']['seeds'][0]}" / "last.pt",
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(checkpoint["state_dict"])
        with torch.inference_mode():
            logits = model(
                view.inputs[chosen].to(device),
                view.positions_m.to(device),
                view.live.to(device),
                torch.full((len(chosen),), view.p, device=device),
            )
            predictions[settings["name"]] = (
                photon_count_log_probabilities(logits, view.live.to(device)).exp().cpu().numpy()
            )
    figure, axes = plt.subplots(
        len(chosen), 1, figsize=(12, len(chosen) * 2.5), squeeze=False, layout="constrained"
    )
    width = manifest["time_bin_width_ns"]
    times = manifest["time_window_ns"][0] + (np.arange(view.inputs.shape[-1]) + 0.5) * width
    for row, index in enumerate(chosen):
        axis = axes[row, 0]
        for name, probabilities in predictions.items():
            axis.plot(
                times,
                probabilities[row].sum(0),
                label=f"{name} probability",
                color=colors[name],
            )
        raw_axis = axis.twinx()
        observed = view.inputs[index].sum(0).numpy()
        raw_axis.scatter(
            times[observed > 0],
            observed[observed > 0],
            marker="|",
            color="#89929b",
            label="Retained A photons",
            s=35,
        )
        target = view.targets[index].sum(0).numpy()
        if target.sum():
            raw_axis.scatter(
                times[target > 0],
                target[target > 0],
                marker="x",
                color="#333333",
                label="Held-out B photons (noisy; not GT)",
                s=20,
            )
        axis.set(
            title=f"Event {view.event_ids[index]}: A={int(view.inputs[index].sum())}, "
            f"B={int(view.targets[index].sum())}; fixed first seed",
            xlabel="Arrival time (ns)",
            ylabel=f"Probability per {width:g} ns bin",
        )
        raw_axis.set_ylabel("Observed photon counts")
        axis.set_ylim(bottom=0)
        raw_axis.set_ylim(0, 1.15 * max(1, int(observed.max()), int(target.max())))
        handles, labels = axis.get_legend_handles_labels()
        raw_handles, raw_labels = raw_axis.get_legend_handles_labels()
        axis.legend(handles + raw_handles, labels + raw_labels, fontsize=7)
    figure.savefig(root / "arrival_examples.png", dpi=160)
    plt.close(figure)
    np.savez_compressed(
        root / "arrival_examples.npz",
        event_ids=view.event_ids[chosen],
        input=view.inputs[chosen].numpy(),
        held_out=view.targets[chosen].numpy(),
        **predictions,
    )
    rows = _comparison_rows(report)
    with (root / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    table = "".join(
        f"<tr><td>{row['method']}</td><td>{row['all']:.4f}</td>"
        f"<td>{row['retained_1_to_10']:.4f}</td></tr>"
        for row in rows
    )
    primary = report["scores"]["conv4d"]["retained_1_to_10"]
    contrast = report["conv4d_minus_control"]["full_time_reference"]["retained_1_to_10"]
    interval = contrast["descriptive_95_percent_interval"]
    all_contrast = report["conv4d_minus_control"]["full_time_reference"]["all"]
    all_interval = all_contrast["descriptive_95_percent_interval"]
    all_comparison = (
        "The all-photon contrast with the full-time reference is uncertain. "
        if all_interval[0] <= 0 <= all_interval[1]
        else "The all-photon contrast and its interval are included in the full results. "
    )
    all_comparison += (
        "All-photon 4D minus full-time NLL: "
        f"{all_contrast['model_minus_control_nats_per_photon']:+.3f}, "
        f"descriptive interval [{all_interval[0]:.3f}, {all_interval[1]:.3f}]. "
    )
    method_note = _method_note(report, manifest)
    (root / "index.html").write_text(
        "<!doctype html><html><meta charset='utf-8'><title>4D convolution retry</title>"
        "<style>body{font:16px system-ui;max-width:1200px;margin:35px auto;padding:0 20px}"
        "img{width:100%}td,th{padding:8px 18px;text-align:left}table{border-collapse:collapse}"
        "tr{border-bottom:1px solid #ddd}</style><h1>Photon-splitting with 4D convolution</h1>"
        f"<p>Exploratory P-ONE simulation validation: {manifest['train_events']} training / "
        f"{manifest['validation_events']} validation events. "
        f"Same A/B photon splits, FP32, optimizer and {config['training']['steps']:,} "
        "updates per seed. "
        "Predictions are normalized sensor/time distributions; no clean photon GT or PSNR.</p>"
        f"{method_note}"
        "<table><tr><th>Method</th><th>All photons NLL</th><th>1-10 retained photons NLL</th>"
        f"</tr>{table}</table><p>The low-count comparison uses {primary['events']} validation "
        f"events and {primary['held_out_photons']} held-out photons. "
        f"4D minus full-time NLL: {contrast['model_minus_control_nats_per_photon']:.3f}, "
        f"descriptive configuration-bootstrap interval [{interval[0]:.3f}, {interval[1]:.3f}], "
        f"conditional on the fitted models. {all_comparison}"
        "This is reused simulation validation, not a new "
        "held-out test or evidence of recovering real undetected events.</p>"
        f"{event_table}"
        "<p>Learning curves show seed means; shaded bands show one standard deviation "
        "across seeds.</p>"
        "<img src='learning_curves.png' alt='Matched learning curves'>"
        f"{event_figure}"
        "<h2>Arrival distributions</h2><p>Examples chosen by input-count strata and record order, "
        "without selecting on target counts or prediction quality. Crosses are a random held-out "
        "photon realization, not a clean ground-truth distribution. "
        "Empty-input predictions are priors.</p>"
        "<img src='arrival_examples.png' alt='Arrival distributions and noisy held-out photons'>"
        "<p><a href='results.json'>Full results and limitations</a> · "
        "<a href='comparison.csv'>Comparison CSV</a> · "
        "<a href='arrival_examples.npz'>Example arrays</a> · "
        "<a href='evidence_audit.json'>Checkpoint and source audit</a></p>"
        "</html>",
        encoding="utf-8",
    )
