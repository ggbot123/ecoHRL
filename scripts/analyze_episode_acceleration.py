"""Compute per-episode absolute-acceleration statistics from evaluation CSVs.

The script first uses an explicit acceleration column when one is present. For
HIRO and single-SAC trajectory CSVs it otherwise reconstructs the actual
acceleration from the ego ``vx`` entries in ``low_obs`` or ``obs`` and the
saved policy frequency. This is the same finite-difference acceleration used
by the environment comfort term. Samples recorded while the ego speed is below
the configured minimum are excluded.
The script writes ``acceleration_episode_metrics.csv`` next to the configured
CSV, or in the configured result directory. It also writes an all-episode
summary CSV containing both episode-equal and pooled-step aggregations. Signed
samples are retained internally so positive acceleration and negative
acceleration (deceleration) distributions are reported separately.
"""

from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_OUTPUT_ROOT = PROJECT_ROOT / "results"

# Add one evaluation-result directory (or one trajectory CSV) per experiment.
# Each experiment receives its own CSV outputs in the corresponding directory.
RESULT_PATHS = [
    # Path(r"results/eval_results/20260709_102056_HRLnew"),
    # Path(r"results/eval_results/20260707_164522_uniHERnew"),
    # Path(r"results/eval_results/20260706_125633_uniNoHERnew"),
    # Path(r"results/eval_results/20260706_175329_ruleNew"),
    # Path(r"results/eval_results/20260707_153258_HRLold"),
    Path(r"results/eval_results/ReUni_fixedHER_oldEnv"),
    # Path(r"results/eval_results/Uni_fixedHER_oldEnv"),
    # Path(r"results/eval_results/Uni_noHER_oldEnv"),
    # Path(r"results/eval_results/Uni_rule_oldEnv"),

    # Path(r"results/eval_results/20260713_170306_sacPriorNew"),
    # Path(r"results/eval_results/20260713_170344_sacPriorOld"),
    # Path(r"results/eval_results/20260713_175523_sacBaseNewWL"),
    # Path(r"results/eval_results/20260713_181256_sacBaseOldWL"),
    # Path(r"results/eval_results/20260714_190137_sacPriorOldWL"),
    
    # Path(r"results/rule_mpc_sacPrior_oldEnv_wrongLanePen/eval_results/20260713_220335"),    # MLC old
    # Path(r"results/rule_mpc_sacPrior_oldEnv_wrongLanePen/eval_results/20260713_220107"),    # MOBIL old
    # Path(r"results/rule_mpc_sacPrior_newEnv_by_lane/eval_results/20260713_220018"),         # MLC new
    # Path(r"results/rule_mpc_sacPrior_newEnv_by_lane/eval_results/20260713_220054"),         # MOBIL new
    # Path(r"results/rule_mpc_sacPrior_oldEnv_wrongLanePen/eval_results/20260713_113433"),
]

# Cross-experiment comparison table. Set to None to disable it. Other batch
# outputs are written beside this file, under the repository's results folder.
BATCH_SUMMARY_PATH: Path | None = RESULTS_OUTPUT_ROOT / "acceleration_experiment_summary.csv"

# Shared acceleration-magnitude bin width for every experiment in the batch.
ACCELERATION_BIN_WIDTH_MPS2 = 0.25
# Distribution plots discard samples whose magnitude is above this limit.
ACCELERATION_DISTRIBUTION_MAX_MPS2 = 4.0
# Coarser bins used by the two multi-experiment comparison charts.
COMPARISON_BIN_WIDTH_MPS2 = 0.5
COMPARISON_GROUP_SIZE = 4

# When enabled, metric CSVs exclude every |acceleration| >= 5 m/s^2 sample.
# This only affects per-episode and all-episode statistics; distribution plots
# keep their own independent display cutoff above.
# EXCLUDE_ABS_ACCELERATION_AT_OR_ABOVE_5_MPS2_FROM_METRICS = False
EXCLUDE_ABS_ACCELERATION_AT_OR_ABOVE_5_MPS2_FROM_METRICS = True
METRICS_ACCELERATION_EXCLUSION_THRESHOLD_MPS2 = 5.0

# Ignore acceleration samples recorded while the ego is effectively stopped.
EXCLUDE_ACCELERATION_BELOW_EGO_SPEED_THRESHOLD = True
MIN_EGO_SPEED_FOR_ACCELERATION_MPS = 1.0


ACCELERATION_COLUMNS = (
    "ego_acceleration_mps2",
    "ego_acceleration",
    "actual_acceleration",
    "acceleration_phys",
    "physical_acceleration",
    "acc_phys",
    "acceleration",
)
SPEED_COLUMNS = ("ego_speed_mps", "ego_speed", "speed", "vx")
NORMALIZED_ACTION_COLUMNS = ("action_post_safety_1", "action_1")
EPISODE_FILE_PATTERN = re.compile(r"(?:^|[_-])ep[_-]?(\d+)(?:[_-]|$)", re.IGNORECASE)


def _experiment_output_dir(input_path: Path) -> Path:
    """Keep experiment outputs under ``results``, preserving in-tree paths."""
    source_dir = input_path.parent if input_path.is_file() else input_path
    source_dir = source_dir.resolve()
    results_root = RESULTS_OUTPUT_ROOT.resolve()
    try:
        source_dir.relative_to(results_root)
    except ValueError:
        experiment_name = input_path.stem if input_path.is_file() else input_path.name
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", experiment_name).strip("._")
        return results_root / "acceleration_analysis" / (safe_name or "experiment")
    return source_dir


def _trajectory_csvs(input_path: Path, output_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".csv":
            raise ValueError(f"Expected a CSV file, got: {input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    csvs = [
        path
        for path in input_path.rglob("*.csv")
        if path.resolve() != output_path.resolve()
        and "trajectory" in path.name.lower()
    ]
    return sorted(csvs)


def _episode_from_filename(path: Path) -> str:
    match = EPISODE_FILE_PATTERN.search(path.stem)
    return match.group(1) if match else path.stem


def _as_finite_float(value: object) -> float | None:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _load_trajectory_metadata(
    input_path: Path,
) -> tuple[float | None, tuple[str, ...], tuple[float, float]]:
    """Read dt, candidate ego-speed columns, and physical action range."""
    root = input_path if input_path.is_dir() else input_path.parent
    config_path = root / "effective_eval_config.json"
    if not config_path.is_file():
        return None, (), (-5.0, 5.0)

    try:
        import json

        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        environment = config.get("environment") or config.get("base_environment") or {}
        frequency = _as_finite_float(environment.get("policy_frequency"))
        dt = 1.0 / frequency if frequency is not None and frequency > 0.0 else None

        observation = environment.get("observation") or {}
        features = list(observation.get("features") or [])
        if "vx" in features:
            vx_index = features.index("vx")
            # HIRO low observations begin with the runner's t_rel, followed
            # directly by the ego vehicle feature vector. Raw SAC observations
            # retain the environment's optional leading time feature.
            sac_time_offset = 1 if bool(observation.get("include_time", False)) else 0
            speed_columns = (
                f"low_obs_{1 + vx_index}",
                f"obs_{sac_time_offset + vx_index}",
            )
        else:
            speed_columns = ()
        acceleration_range = (environment.get("action") or {}).get("acceleration_range", [-5.0, 5.0])
        acc_min, acc_max = float(acceleration_range[0]), float(acceleration_range[1])
        return dt, speed_columns, (acc_min, acc_max)
    except (OSError, ValueError, TypeError, KeyError):
        return None, (), (-5.0, 5.0)


def _read_samples(
    csv_path: Path,
    *,
    dt: float | None,
    metadata_speed_columns: tuple[str, ...],
    acceleration_range: tuple[float, float],
) -> tuple[str | None, list[tuple[str, float]]]:
    """Return ``(source, [(episode, signed_acceleration), ...])``."""
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        normalized = {name.lower(): name for name in fieldnames}
        acceleration_column = next(
            (normalized[name] for name in ACCELERATION_COLUMNS if name in normalized),
            None,
        )
        fallback_episode = _episode_from_filename(csv_path)
        samples: list[tuple[str, float]] = []
        speed_column = next(
            (normalized[name] for name in SPEED_COLUMNS if name in normalized),
            None,
        )
        if speed_column is None:
            speed_column = next(
                (column for column in metadata_speed_columns if column in fieldnames),
                None,
            )

        def speed_is_included(row: dict[str, str], current_speed: float | None = None) -> bool:
            if not EXCLUDE_ACCELERATION_BELOW_EGO_SPEED_THRESHOLD:
                return True
            speed = current_speed
            if speed is None and speed_column is not None:
                speed = _as_finite_float(row.get(speed_column))
            return speed is None or speed >= float(MIN_EGO_SPEED_FOR_ACCELERATION_MPS)

        if acceleration_column is not None:
            for row in reader:
                acceleration = _as_finite_float(row.get(acceleration_column))
                if acceleration is None or not speed_is_included(row):
                    continue
                episode = str(row.get("episode", fallback_episode)).strip() or fallback_episode
                samples.append((episode, acceleration))
            return acceleration_column, samples

        if speed_column is not None and dt is not None:
            previous_by_episode: dict[str, tuple[int | None, float]] = {}
            for row in reader:
                speed = _as_finite_float(row.get(speed_column))
                if speed is None:
                    continue
                episode = str(row.get("episode", fallback_episode)).strip() or fallback_episode
                step = _as_finite_float(row.get("step"))
                step_int = int(step) if step is not None else None
                previous = previous_by_episode.get(episode)
                if previous is not None:
                    previous_step, previous_speed = previous
                    if (
                        (previous_step is None or step_int is None or step_int == previous_step + 1)
                        and speed_is_included(row, speed)
                    ):
                        samples.append((episode, (speed - previous_speed) / dt))
                previous_by_episode[episode] = (step_int, speed)
            return f"finite_difference({speed_column}, dt={dt:g}s)", samples

        action_column = next(
            (normalized[name] for name in NORMALIZED_ACTION_COLUMNS if name in normalized),
            None,
        )
        if action_column is not None:
            acc_min, acc_max = acceleration_range
            for row in reader:
                action_norm = _as_finite_float(row.get(action_column))
                if action_norm is None or not speed_is_included(row):
                    continue
                acceleration = acc_min + (action_norm + 1.0) * 0.5 * (acc_max - acc_min)
                episode = str(row.get("episode", fallback_episode)).strip() or fallback_episode
                samples.append((episode, acceleration))
            return f"mapped_command({action_column})", samples
    return None, []


def _metrics(episode: str, values: Iterable[float]) -> dict[str, float | int | str]:
    samples = np.abs(np.asarray(list(values), dtype=np.float64))
    return {
        "episode": episode,
        "sample_count": int(samples.size),
        "abs_acc_mean_mps2": float(np.mean(samples)),
        "abs_acc_p75_mps2": float(np.percentile(samples, 75)),
        "abs_acc_p90_mps2": float(np.percentile(samples, 90)),
        "abs_acc_p95_mps2": float(np.percentile(samples, 95)),
    }


def _all_episode_rows(rows: list[dict[str, float | int | str]], samples_by_episode: dict[str, list[float]]) -> list[dict[str, float | int | str]]:
    """Return episode-equal and pooled-step summaries for all episodes."""
    metric_columns = (
        "abs_acc_mean_mps2",
        "abs_acc_p75_mps2",
        "abs_acc_p90_mps2",
        "abs_acc_p95_mps2",
    )
    total_sample_count = int(sum(len(values) for values in samples_by_episode.values()))
    episode_mean = {
        "aggregation": "macro_episode_mean",
        "episode_count": len(rows),
        "sample_count": total_sample_count,
        **{
            column: float(np.mean([float(row[column]) for row in rows]))
            for column in metric_columns
        },
    }
    pooled_values = [value for values in samples_by_episode.values() for value in values]
    pooled = _metrics("all", pooled_values)
    pooled_row = {
        "aggregation": "pooled_all_steps",
        "episode_count": len(rows),
        "sample_count": int(pooled["sample_count"]),
        **{column: float(pooled[column]) for column in metric_columns},
    }
    return [episode_mean, pooled_row]


def _write_batch_distribution(samples_by_experiment: dict[str, list[float]]) -> tuple[Path, Path]:
    """Write shared-bin data and separate positive/negative charts per experiment."""
    all_values = [value for values in samples_by_experiment.values() for value in values]
    width = float(ACCELERATION_BIN_WIDTH_MPS2)
    if not all_values or not np.isfinite(width) or width <= 0.0:
        raise ValueError("Cannot plot acceleration distribution without samples and a positive bin width")

    display_max = float(ACCELERATION_DISTRIBUTION_MAX_MPS2)
    if not np.isfinite(display_max) or display_max <= 0.0:
        display_max = max(abs(float(value)) for value in all_values)
    finite_edges = np.arange(0.0, display_max + width * 0.01, width, dtype=np.float64)
    if finite_edges.size < 2:
        finite_edges = np.array([0.0, width], dtype=np.float64)
        display_max = width
    output_dir = (
        BATCH_SUMMARY_PATH.resolve().parent
        if BATCH_SUMMARY_PATH is not None
        else RESULTS_OUTPUT_ROOT.resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "acceleration_distribution_bins.csv"
    plot_dir = output_dir / "acceleration_distribution_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    distribution_rows: list[dict[str, float | int | str]] = []
    for experiment, values in samples_by_experiment.items():
        values_array = np.asarray(values, dtype=np.float64)
        retained_values = values_array[np.abs(values_array) <= display_max]
        discarded_count = int(values_array.size - retained_values.size)
        zero_count = int(np.count_nonzero(retained_values == 0.0))
        direction_data = (
            ("positive", retained_values[retained_values > 0.0]),
            ("negative", -retained_values[retained_values < 0.0]),
        )

        safe_experiment_name = re.sub(r"[^A-Za-z0-9._-]+", "_", experiment).strip("._") or "experiment"
        # Avoid leaving the obsolete combined chart beside the two new charts.
        (plot_dir / f"{safe_experiment_name}_acceleration_distribution.png").unlink(
            missing_ok=True
        )
        for direction, magnitudes in direction_data:
            counts, _ = np.histogram(magnitudes, bins=finite_edges)
            share_percent = counts / max(retained_values.size, 1) * 100.0
            direction_share_percent = counts / max(magnitudes.size, 1) * 100.0
            for index, count in enumerate(counts):
                distribution_rows.append(
                    {
                        "experiment": experiment,
                        "direction": direction,
                        "bin_left_mps2": float(finite_edges[index]),
                        "bin_right_mps2": float(finite_edges[index + 1]),
                        "sample_count": int(count),
                        "direction_sample_count": int(magnitudes.size),
                        "retained_sample_count": int(retained_values.size),
                        "zero_sample_count": zero_count,
                        "discarded_sample_count_gt_4mps2": discarded_count,
                        "share_percent": float(share_percent[index]),
                        "direction_share_percent": float(direction_share_percent[index]),
                    }
                )

            figure, axis = plt.subplots(figsize=(9, 4.8))
            axis.bar(
                finite_edges[:-1],
                direction_share_percent,
                width=np.diff(finite_edges),
                align="edge",
                color="tab:blue" if direction == "positive" else "tab:orange",
                edgecolor="white",
                linewidth=0.5,
            )
            axis.set_xticks(np.arange(0.0, display_max + 1e-9, 1.0))
            axis.set_title(
                f"{direction.capitalize()} acceleration distribution: {experiment} "
                f"(|a| > {display_max:g} m/s^2 discarded; n={magnitudes.size:,})"
            )
            axis.set_xlabel("Acceleration magnitude [m/s^2]")
            axis.set_ylabel(f"Share within {direction} samples [%]")
            axis.set_xlim(left=0.0, right=float(finite_edges[-1]))
            axis.grid(True, axis="y", alpha=0.3)
            figure.tight_layout()
            figure.savefig(
                plot_dir / f"{safe_experiment_name}_{direction}_acceleration_distribution.png",
                dpi=180,
            )
            plt.close(figure)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(distribution_rows[0].keys()))
        writer.writeheader()
        writer.writerows(distribution_rows)
    return csv_path, plot_dir


def _write_group_comparison_charts(samples_by_experiment: dict[str, list[float]]) -> tuple[Path, Path]:
    """Write separate positive/negative comparisons for two experiment groups."""
    width = float(COMPARISON_BIN_WIDTH_MPS2)
    display_max = float(ACCELERATION_DISTRIBUTION_MAX_MPS2)
    edges = np.arange(0.0, display_max + width * 0.01, width, dtype=np.float64)
    if edges.size < 2:
        raise ValueError("Comparison bin settings must define at least one bin")

    output_dir = (
        BATCH_SUMMARY_PATH.resolve().parent
        if BATCH_SUMMARY_PATH is not None
        else RESULTS_OUTPUT_ROOT.resolve()
    )
    plot_dir = output_dir / "acceleration_group_comparison_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "acceleration_group_comparison_bins.csv"
    rows: list[dict[str, float | int | str]] = []

    experiments = list(samples_by_experiment.items())
    groups = [
        experiments[:COMPARISON_GROUP_SIZE],
        experiments[COMPARISON_GROUP_SIZE : 2 * COMPARISON_GROUP_SIZE],
    ]
    group_titles = ("first_four_experiments", "second_four_experiments")
    bin_positions = np.arange(edges.size - 1, dtype=np.float64)
    bin_labels = [f"{edges[i]:g}-{edges[i + 1]:g}" for i in range(edges.size - 1)]
    colors = plt.get_cmap("tab10").colors

    for group_index, (title, group) in enumerate(zip(group_titles, groups), start=1):
        if not group:
            continue
        # Avoid leaving the obsolete two-panel chart beside the split charts.
        (plot_dir / f"acceleration_probability_{title}.png").unlink(missing_ok=True)
        bar_width = 0.8 / len(group)
        for direction in ("positive", "negative"):
            figure, axis = plt.subplots(figsize=(10, 5.2))
            for experiment_index, (experiment, values) in enumerate(group):
                value_array = np.asarray(values, dtype=np.float64)
                retained_values = value_array[np.abs(value_array) <= display_max]
                discarded_count = int(value_array.size - retained_values.size)
                zero_count = int(np.count_nonzero(retained_values == 0.0))
                direction_values = (
                    retained_values[retained_values > 0.0]
                    if direction == "positive"
                    else -retained_values[retained_values < 0.0]
                )
                counts, _ = np.histogram(direction_values, bins=edges)
                probabilities = counts / max(retained_values.size, 1) * 100.0
                direction_probabilities = counts / max(direction_values.size, 1) * 100.0
                offset = (experiment_index - (len(group) - 1) / 2.0) * bar_width
                axis.bar(
                    bin_positions + offset,
                    direction_probabilities,
                    width=bar_width,
                    label=experiment,
                    color=colors[experiment_index % len(colors)],
                    edgecolor="white",
                    linewidth=0.4,
                )
                for bin_index, count in enumerate(counts):
                    rows.append(
                        {
                            "comparison_group": group_index,
                            "experiment": experiment,
                            "direction": direction,
                            "bin_left_mps2": float(edges[bin_index]),
                            "bin_right_mps2": float(edges[bin_index + 1]),
                            "sample_count": int(count),
                            "direction_sample_count": int(direction_values.size),
                            "retained_sample_count": int(retained_values.size),
                            "zero_sample_count": zero_count,
                            "discarded_sample_count_gt_4mps2": discarded_count,
                            "share_percent": float(probabilities[bin_index]),
                            "direction_share_percent": float(direction_probabilities[bin_index]),
                        }
                    )

            axis.set_title(
                f"{direction.capitalize()} acceleration-bin probability comparison: {title}"
            )
            axis.set_xlabel("Acceleration magnitude bin [m/s^2]")
            axis.set_ylabel(f"Share within {direction} samples [%]")
            axis.set_xticks(bin_positions)
            axis.set_xticklabels(bin_labels)
            axis.grid(True, axis="y", alpha=0.3)
            axis.legend()
            figure.tight_layout()
            figure.savefig(
                plot_dir / f"{direction}_acceleration_probability_{title}.png",
                dpi=180,
            )
            plt.close(figure)

    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return csv_path, plot_dir


def _analyze_result_path(
    input_path: Path,
) -> tuple[list[dict[str, float | int | str]], list[float]] | None:
    """Analyze one experiment and write its per-episode and aggregate CSVs."""
    input_path = input_path.resolve()
    output_dir = _experiment_output_dir(input_path)
    output_path = (output_dir / "acceleration_episode_metrics.csv").resolve()
    dt, metadata_speed_columns, acceleration_range = _load_trajectory_metadata(input_path)
    try:
        csv_paths = _trajectory_csvs(input_path, output_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[{input_path}] {exc}", file=sys.stderr)
        return None
    if not csv_paths:
        print(
            f"[{input_path}] No trajectory CSV files found. This analysis needs per-step records; "
            "an eval text summary or PNG acceleration curve is not sufficient.",
            file=sys.stderr,
        )
        return None

    samples_by_episode: dict[str, list[float]] = defaultdict(list)
    source_counts: dict[str, int] = defaultdict(int)
    skipped_files: list[Path] = []
    for csv_path in csv_paths:
        source, samples = _read_samples(
            csv_path,
            dt=dt,
            metadata_speed_columns=metadata_speed_columns,
            acceleration_range=acceleration_range,
        )
        if source is None:
            skipped_files.append(csv_path)
            continue
        source_counts[source] += 1
        for episode, value in samples:
            samples_by_episode[episode].append(value)

    if not samples_by_episode:
        print(
            f"[{input_path}] Trajectory CSV files were found, but no usable acceleration source was found. "
            f"Accepted explicit columns: {', '.join(ACCELERATION_COLUMNS)}. "
            "For HIRO/SAC trajectories, effective_eval_config.json must contain observation.features with 'vx' "
            "and policy_frequency.",
            file=sys.stderr,
        )
        return None

    def episode_sort_key(episode: str) -> tuple[int, int | str]:
        try:
            return (0, int(episode))
        except ValueError:
            return (1, episode)

    metric_samples_by_episode: dict[str, list[float]] = samples_by_episode
    excluded_metric_sample_count = 0
    excluded_metric_episode_count = 0
    if EXCLUDE_ABS_ACCELERATION_AT_OR_ABOVE_5_MPS2_FROM_METRICS:
        threshold = float(METRICS_ACCELERATION_EXCLUSION_THRESHOLD_MPS2)
        if not np.isfinite(threshold) or threshold <= 0.0:
            raise ValueError(
                "METRICS_ACCELERATION_EXCLUSION_THRESHOLD_MPS2 must be a positive finite number"
            )
        metric_samples_by_episode = {}
        for episode, values in samples_by_episode.items():
            retained_values = [value for value in values if abs(value) < threshold]
            excluded_metric_sample_count += len(values) - len(retained_values)
            if retained_values:
                metric_samples_by_episode[episode] = retained_values
            else:
                excluded_metric_episode_count += 1

    rows = [
        _metrics(episode, metric_samples_by_episode[episode])
        for episode in sorted(metric_samples_by_episode, key=episode_sort_key)
    ]
    if not rows:
        print(
            f"[{input_path}] No samples remain after applying the metric exclusion filter.",
            file=sys.stderr,
        )
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    overall_output_path = output_path.with_name("acceleration_all_episodes_metrics.csv")
    overall_rows = _all_episode_rows(rows, metric_samples_by_episode)
    with overall_output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(overall_rows[0].keys()))
        writer.writeheader()
        writer.writerows(overall_rows)

    print(f"[{input_path.name}] Wrote {len(rows)} episode metrics to: {output_path}")
    print(f"[{input_path.name}] Wrote all-episode metrics to: {overall_output_path}")
    if EXCLUDE_ABS_ACCELERATION_AT_OR_ABOVE_5_MPS2_FROM_METRICS:
        print(
            f"[{input_path.name}] Metric filter: excluded {excluded_metric_sample_count} samples "
            f"with |acceleration| >= {METRICS_ACCELERATION_EXCLUSION_THRESHOLD_MPS2:g} m/s^2"
            + (
                f"; omitted {excluded_metric_episode_count} episodes with no retained samples"
                if excluded_metric_episode_count
                else ""
            )
        )
    print(f"[{input_path.name}] All-episode summary (m/s^2):")
    for row in overall_rows:
        print(
            f"  {str(row['aggregation']):<20} "
            f"episodes={int(row['episode_count']):>4d}  samples={int(row['sample_count']):>6d}  "
            f"{float(row['abs_acc_mean_mps2']):>8.4f}  "
            f"{float(row['abs_acc_p75_mps2']):>7.4f}  "
            f"{float(row['abs_acc_p90_mps2']):>7.4f}  "
            f"{float(row['abs_acc_p95_mps2']):>7.4f}"
        )
    if skipped_files:
        print(f"[{input_path.name}] Skipped CSVs without an acceleration column:")
        for path in skipped_files:
            print(f"  {path}")
    print(f"[{input_path.name}] Acceleration sources:")
    for source, count in sorted(source_counts.items()):
        print(f"  {count} CSV files: {source}")
    if EXCLUDE_ACCELERATION_BELOW_EGO_SPEED_THRESHOLD:
        print(
            f"[{input_path.name}] Speed filter: excluded acceleration samples recorded at "
            f"ego speed < {MIN_EGO_SPEED_FOR_ACCELERATION_MPS:g} m/s"
        )
    pooled_values = [value for values in samples_by_episode.values() for value in values]
    return overall_rows, pooled_values


def main() -> int:
    if not RESULT_PATHS:
        print("RESULT_PATHS is empty. Add at least one experiment path.", file=sys.stderr)
        return 2

    batch_rows: list[dict[str, float | int | str]] = []
    samples_by_experiment: dict[str, list[float]] = {}
    success_count = 0
    for configured_path in RESULT_PATHS:
        analysis_result = _analyze_result_path(configured_path)
        if analysis_result is None:
            continue
        overall_rows, pooled_values = analysis_result
        success_count += 1
        resolved_path = configured_path.resolve()
        experiment_name = resolved_path.name
        if experiment_name in samples_by_experiment:
            experiment_name = str(resolved_path)
        samples_by_experiment[experiment_name] = pooled_values
        for row in overall_rows:
            batch_rows.append(
                {
                    "experiment": experiment_name,
                    "experiment_path": str(resolved_path),
                    **row,
                }
            )

    if batch_rows and BATCH_SUMMARY_PATH is not None:
        batch_output_path = BATCH_SUMMARY_PATH.resolve()
        batch_output_path.parent.mkdir(parents=True, exist_ok=True)
        with batch_output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(batch_rows[0].keys()))
            writer.writeheader()
            writer.writerows(batch_rows)
        print(f"Wrote cross-experiment summary to: {batch_output_path}")
    if samples_by_experiment:
        distribution_csv_path, distribution_plot_dir = _write_batch_distribution(samples_by_experiment)
        print(f"Wrote binned acceleration distribution to: {distribution_csv_path}")
        print(f"Wrote per-experiment acceleration bar charts to: {distribution_plot_dir}")
        comparison_csv_path, comparison_plot_dir = _write_group_comparison_charts(samples_by_experiment)
        print(f"Wrote group-comparison bin probabilities to: {comparison_csv_path}")
        print(f"Wrote first/second-four comparison charts to: {comparison_plot_dir}")
    return 0 if success_count == len(RESULT_PATHS) else 2


if __name__ == "__main__":
    raise SystemExit(main())
