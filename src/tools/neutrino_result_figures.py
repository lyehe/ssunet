"""Supporting result figures and a portable interactive photon-distribution viewer."""

from __future__ import annotations

import numpy as np
import pandas as pd


def concentration_figure(records, losses, output, save):
    """Show target/loss concentration and empty-target rates with explicit denominators."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    report = {}
    for stratum, color in (("all", "#657a90"), ("retained_1_to_10", "#cc6b29")):
        mask = records.strata[stratum]
        totals = records.target_totals[mask]
        values = losses[mask]
        order = np.argsort(-totals, kind="stable")
        fraction = np.arange(1, len(order) + 1) / len(order)
        axes[0].plot(
            100 * fraction,
            100 * np.cumsum(totals[order]) / totals.sum(),
            color=color,
            label=f"{stratum}: target photons",
        )
        axes[0].plot(
            100 * fraction,
            100 * np.cumsum(values[order]) / values.sum(),
            color=color,
            linestyle="--",
            label=f"{stratum}: model loss",
        )
        report[stratum] = {
            "events": int(mask.sum()),
            "target_photons": int(totals.sum()),
            "zero_target_fraction": float((totals == 0).mean()),
            "top_1_percent_target_share": float(
                totals[order[: max(1, int(np.ceil(len(order) * 0.01)))]].sum() / totals.sum()
            ),
        }
    axes[0].set_xlabel("Events sorted by target count (%)")
    axes[0].set_ylabel("Cumulative target photons / NLL (%)")
    axes[0].set_title("Contribution concentration")
    axes[0].legend(fontsize=7)
    names = [
        "retained_0",
        "retained_1_to_5",
        "retained_6_to_10",
        "retained_11_to_30",
        "retained_above_30",
    ]
    empty = [100 * (records.target_totals[records.strata[name]] == 0).mean() for name in names]
    axes[1].bar(range(5), empty, color="#7895aa")
    axes[1].set_xticks(range(5), ["0", "1-5", "6-10", "11-30", ">30"])
    axes[1].set_xlabel("Retained input photon count")
    axes[1].set_ylabel("Events with zero target photons (%)")
    axes[1].set_ylim(0, 105)
    axes[1].set_title("Zero-target events stay in the catalogue")
    for index, value in enumerate(empty):
        axes[1].text(index, value + 2, f"{value:.1f}%", ha="center")
    for axis in axes:
        axis.grid(alpha=0.15)
        axis.set_axisbelow(True)
    figure.supxlabel(
        "Zero-target events contribute no location NLL. "
        "An empty observation does not establish a physical no-event state.",
        fontsize=8,
    )
    save(figure, output, "concentration_and_empty_targets")
    return report


def resource_figure(runs, profile, config, output, save):
    """Use measured process/device allocation and distinguish it from budget limits."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    rows = [row for row in runs if row is not None]
    if rows:
        host = max(row["training_state"]["peak_process_tree_rss_gib"] for row in rows)
        gpu = max(row["peak_allocated_gpu_gib"] or 0 for row in rows)
        speed = [row["training_examples_per_second"] for row in rows]
        labels = [f"Seed {row['seed']}" for row in rows]
    else:
        host, gpu = profile["peak_process_tree_rss_gib"], profile["peak_allocated_gpu_gib"]
        speed, labels = [], []
    limits = [
        config["training"]["max_process_rss_gib"],
        config["training"]["max_gpu_allocated_gib"],
    ]
    measured = [host, gpu]
    axes[0].barh(
        ["Host process peak", "GPU allocated peak"],
        [100 * value / limit for value, limit in zip(measured, limits, strict=True)],
        color=["#657a90", "#cc6b29"],
    )
    axes[0].set_xlim(0, 110)
    axes[0].axvline(100, color="black", linestyle="--", label="Registered cap")
    axes[0].set_xlabel("Measured memory / registered limit (%)")
    for index, (value, limit) in enumerate(zip(measured, limits, strict=True)):
        axes[0].text(
            100 * value / limit + 2, index, f"{value:.3f} / {limit:g} GiB", va="center", fontsize=9
        )
    axes[1].barh(
        ["Engineering", *labels],
        [profile["training_events_per_second"], *speed],
        color=["#168496", *["#cc6b29"] * len(speed)],
    )
    axes[1].set_xlabel("Training examples / measured second")
    axes[1].set_title("Data loading + optimizer updates")
    for axis in axes:
        axis.grid(axis="x", alpha=0.15)
        axis.set_axisbelow(True)
    figure.suptitle("Dense reference: measured training resources")
    figure.supxlabel(
        "Per-process measurements; GPU allocation excludes other programs. "
        "Auxiliary GPU studies may run concurrently.",
        fontsize=8,
    )
    save(figure, output, "resource_measurements")


def truth_context_figure(source_truth, output, examples, save):
    """Load MC vertex labels after predictions, purely as geometric display context."""
    import matplotlib.pyplot as plt

    with np.load(output / "event_distributions.npz", allow_pickle=False) as data:
        positions, probabilities, inputs = data["positions_m"], data["q"], data["A"]
        ids = data["event_ids"]
    columns = ["event_no", "initial_state_x", "initial_state_y", "initial_state_z"]
    truth = pd.read_parquet(
        source_truth, columns=columns, filters=[("event_no", "in", ids.tolist())]
    ).set_index("event_no")
    figure = plt.figure(figsize=(12, 7), layout="constrained")
    maximum = float(probabilities.sum(2).max())
    for index, event in enumerate(ids):
        axis = figure.add_subplot(2, 3, index + 1, projection="3d")
        scatter = axis.scatter(
            *positions.T,
            c=probabilities[index].sum(1),
            s=15 + 8 * np.log1p(inputs[index].sum(1)),
            cmap="plasma",
            vmin=0,
            vmax=maximum,
        )
        vertex = truth.loc[
            event, ["initial_state_x", "initial_state_y", "initial_state_z"]
        ].to_numpy(float)
        axis.scatter(*vertex, marker="*", s=160, c="#18a56c", edgecolors="black")
        axis.set_title(f"{examples[index]['category']} | ID {event}", fontsize=9)
        for coordinate in ("x", "y", "z"):
            getattr(axis, f"set_{coordinate}label")(
                f"{coordinate} (source geometry units)", fontsize=7
            )
        axis.tick_params(labelsize=7)
        axis.view_init(elev=18, azim=35)
    figure.colorbar(
        scatter,
        ax=figure.axes,
        label="Predicted probability per sensor (sum over time)",
        shrink=0.6,
        pad=0.02,
    )
    figure.suptitle("MC initial vertex truth (green star): geometric context only")
    figure.supxlabel(
        "No vertex reconstruction is predicted. The simulated initial vertex is not "
        "photon-arrival PDF GT.\nAll displayed probabilities were computed "
        "before these MC labels were read.",
        fontsize=9,
    )
    save(figure, output, "mc_truth_context")


def interactive_distributions(output, examples, partition):
    """Create a self-contained dropdown/zoom viewer without an external server."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    with np.load(output / "event_distributions.npz", allow_pickle=False) as data:
        maps = {key: data[key] for key in ("A", "B", "baseline", "q")}
        times = (data["time_edges_ns"][:-1] + data["time_edges_ns"][1:]) / 2000
    sensors = np.arange(maps["q"].shape[1])
    figure = make_subplots(
        rows=2,
        cols=4,
        specs=[[{}, {}, {}, {}], [{"colspan": 2}, None, {"colspan": 2}, None]],
        subplot_titles=(
            "Retained A: observed counts",
            "Held-out B: observed counts",
            "Selected baseline: probability/bin",
            "Seed 42 model: probability/bin",
            "Time marginals",
            "Sensor marginals",
        ),
        vertical_spacing=0.15,
    )
    count_max = max(2.0, float(max(maps[key].max() for key in ("A", "B"))))
    probability_max = float(max(maps[key].max() for key in ("baseline", "q")))
    for index, _example in enumerate(examples):
        for column, key in enumerate(("A", "B", "baseline", "q"), start=1):
            values = maps[key][index]
            shown = np.log10(np.maximum(values, 1e-30))
            shown = np.where(values > 0, shown, np.nan)
            figure.add_trace(
                go.Heatmap(
                    x=times,
                    y=sensors,
                    z=shown,
                    customdata=values,
                    coloraxis="coloraxis" if column < 3 else "coloraxis2",
                    visible=index == 0,
                    name=key,
                    hovertemplate=(
                        "Time %{x:.3f} us<br>Sensor %{y}<br>Value %{customdata:.6g}<extra></extra>"
                    ),
                ),
                row=1,
                col=column,
            )
        for key, color, label in (
            ("baseline", "#168496", "Selected baseline"),
            ("q", "#cc6b29", "Model seed 42"),
        ):
            for row_axis, target_column, x, axis_sum in ((2, 1, times, 0), (2, 3, sensors, 1)):
                figure.add_trace(
                    go.Scatter(
                        x=x,
                        y=maps[key][index].sum(axis_sum),
                        mode="lines",
                        line={"color": color},
                        name=label,
                        legendgroup=key,
                        showlegend=target_column == 1,
                        visible=index == 0,
                    ),
                    row=row_axis,
                    col=target_column,
                )
    trace_count = 8
    buttons = []
    for index, example in enumerate(examples):
        title = (
            f"{partition.title()} ID {example['event_id']} | {example['category']} | "
            f"A={example['A']}, B={example['B']}"
        )
        buttons.append(
            {
                "label": example["category"],
                "method": "update",
                "args": [
                    {"visible": [i // trace_count == index for i in range(len(figure.data))]},
                    {"title": {"text": title}},
                ],
            }
        )
    figure.update_layout(
        title=(
            f"{partition.title()} distribution examples; B is a held-out realization, not clean GT"
        ),
        height=850,
        margin={"t": 130, "r": 180},
        updatemenus=[{"buttons": buttons, "x": 0, "y": 1.16, "xanchor": "left"}],
        coloraxis={
            "colorscale": "Blues",
            "cmin": 0,
            "cmax": np.log10(count_max),
            "colorbar": {"title": "Counts (log10)", "x": 1.02, "y": 0.78, "len": 0.4},
        },
        coloraxis2={
            "colorscale": "Magma",
            "cmin": -7,
            "cmax": np.log10(probability_max),
            "colorbar": {"title": "Probability (log10)", "x": 1.13, "y": 0.78, "len": 0.4},
        },
        template="plotly_white",
    )
    for column in range(1, 5):
        figure.update_xaxes(title_text="Source time (microseconds)", row=1, col=column)
    figure.update_xaxes(title_text="Source time (microseconds)", row=2, col=1)
    figure.update_xaxes(title_text="Sensor index", row=2, col=3)
    figure.update_yaxes(title_text="Probability / time bin", row=2, col=1)
    figure.update_yaxes(title_text="Probability / sensor", row=2, col=3)
    figure.write_html(output / "interactive_events.html", include_plotlyjs=True, full_html=True)
    return figure.to_html(include_plotlyjs=True, full_html=False)
