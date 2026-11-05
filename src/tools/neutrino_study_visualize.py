"""Export reproducible distribution comparisons from saved study evidence."""

from __future__ import annotations

import argparse
import base64
import csv
import html
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

from ..ssunet.datasets.neutrino import shard_sha256
from ..ssunet.models.neutrino_dense import DensePhotonReference
from ..ssunet.neutrino_evaluation import paired_photon_interval, photon_score_summary
from ..ssunet.photon_reference import photon_count_log_probabilities
from .neutrino_result_figures import (
    concentration_figure,
    interactive_distributions,
    resource_figure,
    truth_context_figure,
)
from .neutrino_study import (
    _assert_same_records,
    _bank_from_selection,
    _dataset,
    _read_scalar_scores,
    _score_bundle,
    _training_reference,
)

COLORS = {"model": "#cc6b29", "baseline": "#168496", "held_out": "#2f935c"}
STRATA = (
    "retained_0",
    "retained_1_to_5",
    "retained_6_to_10",
    "retained_11_to_30",
    "retained_above_30",
)


def save_figure(figure, output, name):
    """Write shareable raster and editable vector figures."""
    for extension in ("png", "svg"):
        figure.savefig(output / f"{name}.{extension}", dpi=180, bbox_inches="tight")
    plt.close(figure)


def select_cases(records):
    """Select median event IDs in count categories without consulting losses."""
    a, b = records.retained_totals, records.target_totals
    rules = [
        ("Empty observation", (a == 0) & (b == 0)),
        ("Empty retained input", (a == 0) & (b > 0)),
        ("1-5 retained", (a >= 1) & (a <= 5) & (b > 0)),
        ("6-10 retained", (a >= 6) & (a <= 10) & (b > 0)),
        ("11-30 retained", (a >= 11) & (a <= 30) & (b > 0)),
        ("Above 30 retained", (a > 30) & (b > 0)),
    ]
    result = []
    for label, mask in rules:
        candidates = np.flatnonzero(mask)
        if len(candidates):
            ordered = candidates[np.argsort(records.event_ids[candidates])]
            result.append((label, int(ordered[len(ordered) // 2])))
    return result


def _score_plots(  # noqa: C901
    records,
    values,
    baseline,
    curves,
    profile,
    output,
    scope,
    *,
    validation_baselines=None,
    contrasts=None,
):
    """Compare photon-weighted scores, denominators and actual resource use."""
    summaries = {key: photon_score_summary(loss, records) for key, loss in values.items()}
    methods = [
        key
        for key in (
            "uniform",
            "selected_prior",
            "fixed_mixture",
            "frozen_mixture",
            baseline,
            "model",
        )
        if key in values
    ]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for axis, stratum, title in zip(
        axes,
        ("all", "retained_1_to_10"),
        ("All target photons", "Primary: 1-10 retained"),
        strict=True,
    ):
        numbers = [summaries[key][stratum]["nats_per_photon"] for key in methods]
        axis.barh(
            methods,
            numbers,
            color=[COLORS["model"] if key == "model" else COLORS["baseline"] for key in methods],
        )
        axis.invert_yaxis()
        axis.set_xlim(0, max(numbers) * 1.16)
        for index, number in enumerate(numbers):
            axis.text(number + 0.04, index, f"{number:.3f}", va="center", fontsize=9)
        axis.set_title(title)
        axis.set_xlabel("Held-out NLL (nats/photon; lower is better)")
        axis.grid(axis="x", alpha=0.15)
        axis.set_axisbelow(True)
    figure.suptitle(scope)
    save_figure(figure, output, "score_comparison")

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for key, color in ((baseline, COLORS["baseline"]), ("model", COLORS["model"])):
        axes[0].plot(
            range(5),
            [summaries[key][s]["nats_per_photon"] for s in STRATA],
            "o-",
            label=key,
            color=color,
        )
    axes[0].set_ylabel("NLL (nats/photon)")
    axes[0].legend()
    photons = [summaries["model"][s]["held_out_photons"] for s in STRATA]
    axes[1].bar(range(5), photons, color="#7895aa")
    for index, stratum in enumerate(STRATA):
        item = summaries["model"][stratum]
        axes[1].text(
            index,
            photons[index] + max(photons) * 0.025,
            f"{item['events']:,} events\n{item['events_with_target']:,} with B\n"
            f"{photons[index]:,} photons",
            ha="center",
            fontsize=7,
        )
    axes[1].set_ylim(0, max(photons) * 1.28)
    axes[1].set_ylabel("Held-out target photons")
    for axis in axes:
        axis.set_xticks(range(5), ["0", "1-5", "6-10", "11-30", ">30"])
        axis.set_xlabel("Retained input count A")
        axis.grid(axis="y", alpha=0.15)
        axis.set_axisbelow(True)
    figure.suptitle("Count strata and evaluation denominators")
    save_figure(figure, output, "count_strata")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    for seed, curve in curves.items():
        x = [row["optimizer_step"] for row in curve]
        for axis, column in zip(
            axes, ("validation_nats_per_photon", "primary_validation_nats_per_photon"), strict=True
        ):
            axis.plot(x, [row[column] for row in curve], "o-", label=f"Seed {seed}")
    for axis, stratum in zip(axes, ("all", "retained_1_to_10"), strict=True):
        level = (
            summaries[baseline][stratum]
            if validation_baselines is None
            else validation_baselines[baseline][stratum]
        )
        axis.axhline(
            level["nats_per_photon"],
            color=COLORS["baseline"],
            linestyle="--",
            label="Selected baseline",
        )
        axis.set_xlabel("Optimizer updates")
        axis.set_ylabel("Frozen validation NLL (nats/photon; log axis)")
        axis.set_yscale("log")
        axis.set_title(stratum)
        axis.legend(fontsize=8)
        axis.grid(alpha=0.15)
    figure.suptitle("Full recorded learning curves, including late failures")
    save_figure(figure, output, "learning_curves")

    intervals = {}
    figure, axis = plt.subplots(figsize=(10, 3.8), layout="constrained")
    rows = [("retained_1_to_10", baseline, "Primary vs selected baseline")]
    if "matched_donor" in values:
        rows.append(("retained_1_to_10", "matched_donor", "Primary vs exact-count donors"))
    rows.append(("all", baseline, "All photons vs selected baseline"))
    for index, (stratum, control, _) in enumerate(rows):
        interval = (
            contrasts[stratum][control]
            if contrasts is not None
            else paired_photon_interval(
                values["model"] - values[control],
                records,
                records.strata[stratum],
                seed=20261025,
                draws=10000,
                chunk_size=64,
            )
        )
        intervals[f"{stratum}_versus_{control}"] = interval
        point = interval["model_minus_control_nats_per_photon"]
        lo, hi = interval["descriptive_95_percent_interval"]
        axis.errorbar(
            point,
            index,
            xerr=[[point - lo], [hi - point]],
            fmt="o",
            capsize=5,
            color=COLORS["model"],
        )
        axis.annotate(f" {point:+.3f} [{lo:+.3f}, {hi:+.3f}]", (hi, index), fontsize=9, va="center")
    axis.axvline(0, color="black", linestyle="--", linewidth=1)
    axis.set_yticks(range(len(rows)), [row[2] for row in rows])
    axis.invert_yaxis()
    axis.margins(x=0.45, y=0.5)
    axis.set_xlabel("Model minus selected baseline NLL (negative favors model)")
    axis.set_title("Descriptive 95% paired configuration-bootstrap intervals")
    figure.supxlabel(
        "Validation selection uncertainty is not included. "
        "Seeds are averaged, never counted as extra events.",
        fontsize=8,
    )
    save_figure(figure, output, "paired_intervals")
    return {"scores": summaries, "intervals": intervals, "engineering_profile": profile}


@torch.inference_mode()
def _event_plots(config, records, losses, checkpoint, output, partition="validation"):  # noqa: C901
    """Predict only selected validation cases from A, then display B separately."""
    training, reference = _training_reference(config)
    validation = _dataset(config, partition)
    bank = _bank_from_selection(config, training, reference, "cpu")
    model = DensePhotonReference(config["model"]["channels"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    choices = select_cases(records)
    examples = []
    for label, index in choices:
        item = validation[index]
        a, b = item["input"], item["target"]
        logits = model(a[None], item["positions_m"], item["live"], item["p"].reshape(1))
        q = photon_count_log_probabilities(logits, item["live"]).exp()[0].numpy()
        np.testing.assert_allclose(q.sum(), 1, atol=2e-6)
        scored = b.numpy() > 0
        actual = -(b.numpy()[scored] * np.log(q[scored])).sum()
        np.testing.assert_allclose(actual, losses[index], atol=0.003, rtol=2e-5)
        baseline = bank.probabilities(a[None])[bank.winner][0].numpy()
        examples.append(
            {
                "label": label,
                "event_id": int(records.event_ids[index]),
                "A": a.numpy(),
                "B": b.numpy(),
                "baseline": baseline,
                "q": q,
            }
        )
    centers = (validation.edges_ns[:-1] + validation.edges_ns[1:]) / 2000
    limit = int(np.searchsorted(validation.edges_ns, 2000))
    figure, axes = plt.subplots(
        len(examples), 5, figsize=(17, 2.1 * len(examples)), layout="constrained", squeeze=False
    )
    count_norm = LogNorm(1, max(2, max(float(e[k].max()) for e in examples for k in ("A", "B"))))
    q_norm = LogNorm(1e-7, max(float(e[k].max()) for e in examples for k in ("baseline", "q")))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#f1f4f6")
    for row, example in enumerate(examples):
        for column, key in enumerate(("A", "B", "baseline", "q")):
            shown = example[key][:, :limit]
            rendered = axes[row, column].imshow(
                np.ma.masked_where(shown == 0, shown),
                origin="lower",
                aspect="auto",
                extent=[0, 2, -0.5, len(shown) - 0.5],
                cmap=cmap,
                norm=count_norm if column < 2 else q_norm,
            )
            if row == 0:
                axes[row, column].set_title(
                    {
                        "A": "Retained A: counts",
                        "B": "Held-out B: counts",
                        "baseline": "Selected baseline: probability/bin",
                        "q": "Learned q: probability/bin",
                    }[key],
                    fontsize=9,
                )
            if column == 0:
                axes[row, column].set_ylabel(
                    f"{example['label']}\nID {example['event_id']} | "
                    f"A={example['A'].sum():g}, B={example['B'].sum():g}\nSensor index",
                    fontsize=8,
                )
            if column == 1:
                count_image = rendered
            if column == 3:
                q_image = rendered
            if column < 2 and not shown.any():
                axes[row, column].text(
                    1, len(shown) / 2, "No detections", ha="center", color="#596979", fontsize=8
                )
            axes[row, column].set_xlabel("Time (microseconds)", fontsize=8)
        axis = axes[row, 4]
        for key, color in (("baseline", COLORS["baseline"]), ("q", COLORS["model"])):
            axis.plot(centers, example[key].sum(0), label=key, color=color)
        if example["B"].sum():
            axis.plot(
                centers,
                example["B"].sum(0) / example["B"].sum(),
                color=COLORS["held_out"],
                alpha=0.5,
                label="B empirical",
            )
        axis.set_xlim(0, 2)
        axis.set_xlabel("Time (microseconds)", fontsize=8)
        axis.set_ylabel("Probability / time bin", fontsize=8)
        if row == 0:
            axis.set_title("Time marginals", fontsize=9)
            axis.legend(fontsize=7)
    figure.colorbar(
        count_image, ax=axes[:, :2], shrink=0.55, pad=0.01, label="Observed counts (log colors)"
    )
    figure.colorbar(
        q_image, ax=axes[:, 2:4], shrink=0.55, pad=0.01, label="Probability / bin (log colors)"
    )
    figure.suptitle(f"{partition.title()} distributions: count-selected examples; seed 42")
    figure.supxlabel(
        "Display zoom 0-2 microseconds; PDFs retain normalization over the full "
        "0-6.4 microsecond window.\nB is a random held-out realization, not clean "
        "photon-arrival GT. Selection uses counts and IDs, never model NLL.",
        fontsize=9,
    )
    save_figure(figure, output, "event_distributions")
    np.savez_compressed(
        output / "event_distributions.npz",
        event_ids=[e["event_id"] for e in examples],
        time_edges_ns=validation.edges_ns,
        positions_m=validation.positions_m,
        **{key: np.stack([e[key] for e in examples]) for key in ("A", "B", "baseline", "q")},
    )
    training.close()
    validation.close()
    return [
        {
            "event_id": e["event_id"],
            "category": e["label"],
            "A": int(e["A"].sum()),
            "B": int(e["B"].sum()),
            "pdf_mass": float(e["q"].sum()),
        }
        for e in examples
    ]


def _snapshot_seed(root, output, seed, partition):
    """Copy weights and score records so running checkpoints cannot change a figure."""
    directory = root / f"runs/seed_{seed}"
    if partition == "test":
        scored = root / f"test_evidence/seed_{seed}.npz"
    else:
        scored = directory / "selected_validation_evidence.npz"
        if not scored.exists():
            scored = directory / "best_validation_scores.npz"
    if not scored.exists():
        return None
    checkpoint_path = output / f"seed_{seed}_snapshot.pt"
    score_path = output / f"seed_{seed}_scores_snapshot.npz"
    shutil.copyfile(directory / "best_low_count.pt", checkpoint_path)
    shutil.copyfile(scored, score_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    identity = json.loads((root / "training_manifest.json").read_text())["identity"]
    if checkpoint["identity"] != identity:
        message = "Visualized checkpoint differs from the recorded training identity"
        raise ValueError(message)
    records, values = _read_scalar_scores(score_path)
    with np.load(score_path, allow_pickle=False) as data:
        exact = data["exact_donor"] if "exact_donor" in data else None
    result = (
        json.loads((directory / "training_result.json").read_text())
        if (directory / "training_result.json").exists()
        else None
    )
    return {
        "records": records,
        "values": values,
        "exact": exact,
        "curve": json.loads((directory / "learning_curve.json").read_text()),
        "checkpoint": checkpoint,
        "metadata": {
            "path": str(checkpoint_path),
            "sha256": shard_sha256(checkpoint_path),
            "score_sha256": shard_sha256(score_path),
            "selected_epoch": checkpoint["training_state"]["best_epoch"],
        },
        "training_result": result,
    }


def _write_comparison_csv(output, scores):
    """Export scores with their event and photon denominators."""
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "method",
                "stratum",
                "nats_per_photon",
                "events",
                "events_with_target",
                "held_out_photons",
            ],
        )
        writer.writeheader()
        for method, strata in scores.items():
            for stratum, row in strata.items():
                writer.writerow({"method": method, "stratum": stratum, **row})


def _write_report(output, result, interactive, analytic):
    """Write one offline HTML report plus a portable bundle of source exports."""
    panels = []
    names = [
        "score_comparison",
        "paired_intervals",
        "learning_curves",
        "count_strata",
        "concentration_and_empty_targets",
        "resource_measurements",
        "event_distributions",
        "mc_truth_context",
    ]
    if analytic:
        names.extend(
            ["analytic_known_pdf_comparison", "analytic_oracle_nll", "analytic_known_pdf_kl"]
        )
    for name in names:
        encoded = base64.b64encode((output / f"{name}.png").read_bytes()).decode()
        panels.append(
            f'<figure><img src="data:image/png;base64,{encoded}" '
            f'alt="{html.escape(name)}"></figure>'
        )
    page = (
        '<!doctype html><meta charset="utf-8"><title>Photon self-splitting results</title>'
        "<style>body{max-width:1600px;margin:30px auto;padding:0 20px;font:16px system-ui;"
        "color:#213345;background:#fafcfe}img{width:100%}figure{margin:25px 0;background:white;"
        "border:1px solid #dae3ea}p{max-width:1100px;line-height:1.5}"
        "summary{cursor:pointer;padding:15px}</style>"
    )
    page += (
        f"<h1>Photon-arrival distribution comparisons</h1><p>{html.escape(result['scope'])}</p>"
        f"<p>{html.escape(result['ground_truth_policy'])} Event maps show one registered seed; "
        "aggregate scores average per-event NLL across seeds. "
        "The full learning curves retain late failures. "
        "The reserved test is first opened only by the separate locked analysis, "
        "never by this report.</p>"
    )
    page += (
        "<details><summary>Interactive event distributions: "
        f"choose an event, zoom and inspect values</summary>{interactive}</details>"
    )
    page += "".join(panels)
    if analytic:
        page += (
            "<p>The analytic figures use separately trained analytic models "
            "and known latent rates. "
            "The latent oracle has information unavailable to the model. "
            "This benchmark's negative comparisons remain visible.</p>"
        )
    (output / "index.html").write_text(page, encoding="utf-8")
    for source in (Path(__file__), Path(__file__).with_name("neutrino_result_figures.py")):
        shutil.copyfile(source, output / f"source_{source.name}")
    archive = output.parent / f"{output.name}.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in output.iterdir():
            if path.is_file():
                bundle.write(path, arcname=f"{output.name}/{path.name}")


def create_study_visualizations(config, output, partition="validation"):
    """Render saved evidence, with a hard guard against opening an unscored test."""
    root = Path(config["data"]["output"])
    if partition == "test" and not (root / "test_results.json").exists():
        message = "Test visualizations require the completed locked test result first"
        raise ValueError(message)
    if partition not in ("validation", "test"):
        message = "Visualizations are limited to frozen validation or completed test"
        raise ValueError(message)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(config["training"]["cpu_threads"])
    source = (
        root / "validation_baseline_scores.npz"
        if partition == "validation"
        else root / "test_evidence/seed_42.npz"
    )
    records, values = _read_scalar_scores(source)
    snapshots = {}
    for seed in config["training"]["seeds"]:
        snapshot = _snapshot_seed(root, output, seed, partition)
        if snapshot is None:
            continue
        _assert_same_records(records, snapshot["records"])
        snapshots[seed] = snapshot
    if not snapshots:
        message = "No recorded validation checkpoints are available"
        raise FileNotFoundError(message)
    values["model"] = np.mean(
        [snapshot["values"]["model"] for snapshot in snapshots.values()], axis=0
    )
    exact = None
    if all("matched_donor" in snapshot["values"] for snapshot in snapshots.values()):
        values["matched_donor"] = np.mean(
            [snapshot["values"]["matched_donor"] for snapshot in snapshots.values()], axis=0
        )
        exact = np.logical_and.reduce([snapshot["exact"] for snapshot in snapshots.values()])
    else:
        values.pop("matched_donor", None)
    baseline = json.loads((root / "baseline_selection.json").read_text())["selected_method"]
    completed = all(
        snapshot["training_result"] is not None for snapshot in snapshots.values()
    ) and len(snapshots) == len(config["training"]["seeds"])
    scope = f"{partition.title()}: {len(snapshots)} selected seed checkpoints; " + (
        "fixed training budgets complete" if completed else "training still incomplete"
    )
    profile = json.loads((root / "engineering_profile.json").read_text())
    evidence = _score_bundle(records, values, config, baseline, exact)
    curves = {seed: snapshot["curve"] for seed, snapshot in snapshots.items()}
    validation_baselines = json.loads((root / "validation_baseline_scores.json").read_text())
    result = _score_plots(
        records,
        values,
        baseline,
        curves,
        profile,
        output,
        scope,
        validation_baselines=validation_baselines,
        contrasts=evidence["contrasts"],
    )
    result["examples"] = _event_plots(
        config,
        records,
        snapshots[42]["values"]["model"],
        snapshots[42]["checkpoint"],
        output,
        partition,
    )
    result["concentration"] = concentration_figure(records, values["model"], output, save_figure)
    resource_figure(
        [snapshot["training_result"] for snapshot in snapshots.values()],
        profile,
        config,
        output,
        save_figure,
    )
    truth_context_figure(
        root / "source_truth.evaluation_only.parquet", output, result["examples"], save_figure
    )
    interactive = interactive_distributions(output, result["examples"], partition)
    analytic = root.parent / "analytic_study/visualizations_v2"
    analytic_available = (analytic / "index.html").exists()
    if analytic_available:
        for name in ("known_pdf_comparison", "oracle_nll", "known_pdf_kl"):
            shutil.copyfile(analytic / f"{name}.png", output / f"analytic_{name}.png")
        shutil.copyfile(analytic / "index.html", output / "analytic_known_gt.html")
    result.update(
        {
            "scope": scope,
            "partition": partition,
            "opens_an_unscored_reserved_test": False,
            "checkpoints": {seed: snapshot["metadata"] for seed, snapshot in snapshots.items()},
            "exact_donor_fraction": evidence.get("exact_donor_fraction"),
            "ground_truth_policy": (
                "Physical held-out photons are observations. Known expected PDF GT exists "
                "only for the separate analytic generator. MC vertices are display context."
            ),
            "seed_combination": "mean per-event NLL, not probability ensemble",
            "event_map_seed": 42,
            "code_sha256": shard_sha256(Path(__file__)),
        }
    )
    (output / "manifest.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    _write_comparison_csv(output, result["scores"])
    _write_report(output, result, interactive, analytic_available)
    return result


def main():
    """Build a new immutable visualization snapshot directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/neutrino_study.yml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partition", choices=("validation", "test"), default="validation")
    args = parser.parse_args()
    create_study_visualizations(
        yaml.safe_load(args.config.read_text()), args.output, args.partition
    )


if __name__ == "__main__":
    main()
