"""Export true-rate GT comparisons for models trained on the analytic reference."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from ..ssunet.models.neutrino_dense import DensePhotonReference
from ..ssunet.neutrino_streaming import collect_training_reference
from ..ssunet.photon_reference import photon_count_log_probabilities
from .neutrino_analytic_study import _baseline_bank, analytic_location_pdf, open_analytic
from .neutrino_prepare import write_json
from .neutrino_study_visualize import save_figure


@torch.inference_mode()
def create_analytic_figures(config, output=None):
    """Display prediction, true PDF and observed photons with explicitly different roles."""
    torch.set_num_threads(config["training"]["cpu_threads"])
    root = Path(config["data"]["output"])
    result = json.loads((root / "test_results.json").read_text())
    output = root / "visualizations" if output is None else Path(output)
    output.mkdir(exist_ok=False)
    training, testing = open_analytic(config, "train"), open_analytic(config, "test")
    reference = collect_training_reference(training)
    selected = json.loads((root / "baseline_selection.json").read_text())
    bank = _baseline_bank(config, training, reference, "cpu", selected)
    lock = json.loads((root / "analysis_lock.json").read_text())
    model = DensePhotonReference(config["model"]["channels"])
    model.load_state_dict(
        torch.load(lock["checkpoints"]["42"]["path"], map_location="cpu", weights_only=True)[
            "state_dict"
        ]
    )
    model.eval()
    with np.load(root / "observations.evaluation_only.npz", allow_pickle=False) as truth:
        sources = truth["latent_source_position_m"]
    levels = config["generator"]["analytic"]["expected_signal_photons"]
    background = config["generator"]["analytic"]["expected_background_photons"]
    event_ids = testing.event_ids[testing.indices]
    examples = []
    for level in (0, 5, 30, 100):
        candidates = np.flatnonzero(
            np.asarray([levels[int(e) % len(levels)] == level for e in event_ids])
        )
        ordered = candidates[np.argsort(event_ids[candidates])]
        index = int(ordered[len(ordered) // 2])
        event = int(event_ids[index])
        item = testing[index]
        a = item["input"][None]
        prediction = (
            photon_count_log_probabilities(
                model(a, item["positions_m"], item["live"], item["p"].reshape(1)), item["live"]
            )
            .exp()[0]
            .numpy()
        )
        baseline = bank.probabilities(a)[bank.winner][0].numpy()
        # The model prediction is fixed above before reading this event's
        # latent parameters. They enter the diagnostic oracle only.
        pdf = analytic_location_pdf(
            testing.positions_m, testing.edges_ns, sources[event // len(levels)], level, background
        )
        examples.append(
            {
                "event_id": event,
                "signal": level,
                "A": a[0].numpy(),
                "B": item["target"].numpy(),
                "GT": pdf,
                "baseline": baseline,
                "model": prediction,
            }
        )
    figure, axes = plt.subplots(len(examples), 5, figsize=(16, 10), layout="constrained")
    norm = LogNorm(
        1e-7, max(float(e[k].max()) for e in examples for k in ("GT", "baseline", "model"))
    )
    centers = (testing.edges_ns[:-1] + testing.edges_ns[1:]) / 2000
    for row, example in enumerate(examples):
        for column, key in enumerate(("GT", "baseline", "model")):
            rendered = axes[row, column].imshow(
                example[key],
                origin="lower",
                aspect="auto",
                extent=[0, 6.4, -0.5, len(testing.live) - 0.5],
                norm=norm,
                cmap="magma",
            )
            if row == 0:
                axes[row, column].set_title(
                    {
                        "GT": "Known expected location PDF (GT)",
                        "baseline": "Selected baseline PDF",
                        "model": "Learned PDF: analytic-trained model",
                    }[key],
                    fontsize=9,
                )
            axes[row, column].set_xlabel("Time (microseconds)", fontsize=8)
        axes[row, 0].set_ylabel(
            f"Signal {example['signal']} + background {background:g}\n"
            f"ID {example['event_id']} | A={example['A'].sum()}, "
            f"B={example['B'].sum()}\nSensor index",
            fontsize=8,
        )
        for key, color in (("GT", "#7c2358"), ("baseline", "#188396"), ("model", "#cc6b29")):
            axes[row, 3].plot(centers, example[key].sum(0), label=key, color=color)
            axes[row, 4].plot(
                np.arange(len(testing.live)), example[key].sum(1), label=key, color=color
            )
        peak = max(float(example[key].sum(0).max()) for key in ("GT", "baseline", "model"))
        for key, color, offset in (("A", "#3678b5", -0.04), ("B", "#2f935c", -0.09)):
            observed = example[key].sum(0)
            active = observed > 0
            axes[row, 3].scatter(
                centers[active],
                np.full(int(active.sum()), offset * peak),
                s=10 + 8 * np.log1p(observed[active]),
                marker="|",
                color=color,
                label=f"{key} arrivals (rug)",
            )
        axes[row, 3].set_ylim(-0.14 * peak, 1.08 * peak)
        axes[row, 3].set_xlabel("Time (microseconds)")
        axes[row, 4].set_xlabel("Sensor index")
        for axis in axes[row, 3:]:
            axis.set_ylabel("Marginal probability / bin", fontsize=8)
            axis.grid(alpha=0.15)
        if row == 0:
            axes[row, 3].set_title("Time marginal")
            axes[row, 4].set_title("Sensor marginal")
            axes[row, 3].legend(fontsize=7)
    figure.colorbar(
        rendered,
        ax=axes[:, :3],
        shrink=0.65,
        pad=0.01,
        label="Probability / sensor-time bin (common log scale)",
    )
    figure.suptitle(
        "Exact analytic GT versus self-split predictions on unseen source configurations"
    )
    figure.supxlabel(
        "Seed 42; cases selected by expected signal category and median event ID, "
        "never prediction quality.\nThe latent oracle knows the source position, "
        "unlike the model. Zero signal + background is not a photon-free state.",
        fontsize=9,
    )
    save_figure(figure, output, "known_pdf_comparison")
    _metric_figures(config, result, bank.winner, output)
    np.savez_compressed(
        output / "known_pdf_examples.npz",
        event_ids=[e["event_id"] for e in examples],
        **{
            key: np.stack([e[key] for e in examples])
            for key in ("A", "B", "GT", "baseline", "model")
        },
    )
    write_json(
        output / "manifest.json",
        {
            "scope": "analytic_trained_models_independent_test_known_rate_GT",
            "event_map_seed": 42,
            "known_rate_in_training": False,
            "event_selection": "median_ID_in_prescribed_expected_signal_category",
            "cases": [
                {
                    "event_id": e["event_id"],
                    "expected_signal": e["signal"],
                    "expected_background": background,
                    "retained_photons": int(e["A"].sum()),
                    "held_out_photons": int(e["B"].sum()),
                    "model_pdf_mass": float(e["model"].sum()),
                }
                for e in examples
            ],
            "GT_is_unavailable_in_physical_data": True,
        },
    )
    images = []
    for name in ("known_pdf_comparison", "oracle_nll", "known_pdf_kl"):
        encoded = base64.b64encode((output / f"{name}.png").read_bytes()).decode()
        images.append(
            f'<img style="width:100%;margin:20px 0" src="data:image/png;base64,{encoded}" '
            f'alt="{name}">'
        )
    page = (
        '<!doctype html><meta charset="utf-8"><title>Known analytic photon GT</title>'
        '<body style="max-width:1600px;margin:30px auto;font:16px system-ui;padding:20px">'
        "<h1>Known arrival-distribution GT</h1><p>Models were trained on observed A/B splits. "
        "The true rate is used only for evaluation. "
        "Physical telescope data do not supply this GT.</p>"
    )
    page += "".join(images) + "</body>"
    (output / "index.html").write_text(page, encoding="utf-8")
    training.close()
    testing.close()


def _metric_figures(config, result, baseline, output):
    """Keep held-out log scores separate from exact expected-distribution KL."""
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    for axis, stratum in zip(axes, ("all", "retained_1_to_10"), strict=True):
        methods = [baseline, "model", "latent_oracle"]
        values = [result["scores"][method][stratum]["nats_per_photon"] for method in methods]
        axis.bar(methods, values, color=["#188396", "#cc6b29", "#7c2358"])
        axis.set_ylabel("Held-out NLL (nats/photon)")
        axis.set_title(stratum)
        axis.tick_params(axis="x", labelsize=8)
        axis.set_ylim(0, max(values) * 1.16)
        for index, value in enumerate(values):
            axis.text(index, value + 0.08, f"{value:.3f}", ha="center")
    figure.suptitle("Analytic test: predictive NLL versus oracle with latent source information")
    save_figure(figure, output, "oracle_nll")
    levels = config["generator"]["analytic"]["expected_signal_photons"]
    figure, axis = plt.subplots(figsize=(9, 4), layout="constrained")
    for key, label, color in (
        ("mean_seed_event_model_KL", "Learned (mean seed KL)", "#cc6b29"),
        ("mean_event_baseline_KL", "Selected baseline", "#188396"),
    ):
        axis.plot(
            range(len(levels)),
            [result["by_expected_signal"][str(level)][key] for level in levels],
            "o-",
            color=color,
            label=label,
        )
    axis.set_xticks(range(len(levels)), [f"{level} + 1" for level in levels])
    axis.set_xlabel("Expected signal photons + expected background photons")
    axis.set_ylabel("Event-mean KL(true location PDF || prediction), nats")
    axis.set_title("Exact expected-distribution error; no noisy photon map used as GT")
    axis.axhline(0, color="#7c2358", linestyle="--", label="Latent oracle: KL = 0")
    axis.grid(alpha=0.15)
    axis.legend()
    save_figure(figure, output, "known_pdf_kl")


def main():
    """Render a completed independent analytic benchmark without refitting models."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/neutrino_diagnostics.yml"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    create_analytic_figures(yaml.safe_load(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
