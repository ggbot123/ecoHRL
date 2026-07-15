import os
from typing import Dict, List, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ScalarSeries = List[Tuple[int, float]]
LanePathGroup = Sequence[str]
CachedSeriesResult = Union[Tuple[str, ScalarSeries], str]
SeriesCache = Dict[str, Dict[str, CachedSeriesResult]]


# ===== User Config =====
# Each inner list represents one curve and must contain exactly three TensorBoard
# paths in this order: lane2to0, lane2to1, lane2to2. A path can be an event file
# or a directory. The three series are aligned by step and averaged before
# smoothing and plotting.
EVENT_PATH_GROUPS: List[LanePathGroup] = [
    [
        r"D:\workspace\python\ecoHRL\logs\current\sac_260709_base_wronglanePen_newEnv_2to0\SAC_1",
        r"D:\workspace\python\ecoHRL\logs\current\sac_260709_base_wronglanePen_newEnv_2to1\SAC_1",
        r"D:\workspace\python\ecoHRL\logs\current\sac_260709_base_wronglanePen_newEnv_2to2\SAC_1",
    ],
    [
        r"D:\workspace\python\ecoHRL\logs\current\sac_260624_withPrior_2to0\SAC_1",
        r"D:\workspace\python\ecoHRL\logs\current\sac_260704_withPrior_2to1\SAC_1",
        r"D:\workspace\python\ecoHRL\logs\current\sac_260622_withPrior_2to2_noGoalReshape\SAC_1",
    ],
    [
        r"D:\workspace\python\ecoHRL\logs\current\sac_260709_withPrior_wronglanePen_newEnv_2to0\SAC_1",
        r"D:\workspace\python\ecoHRL\logs\current\sac_260704_withPrior_2to1\SAC_1",
        r"D:\workspace\python\ecoHRL\logs\current\sac_260622_withPrior_2to2_noGoalReshape\SAC_1",
    ],
    [
        r"D:\workspace\python\ecoHRL\logs\current\hiro_260708_highonly_reUni_oldLow_newEnv_2to0\hiro_high\hiro_high_1",
        r"D:\workspace\python\ecoHRL\logs\current\hiro_260708_highonly_reUni_oldLow_newEnv_2to1\hiro_high\hiro_high_1",
        r"D:\workspace\python\ecoHRL\logs\current\hiro_260708_highonly_reUni_oldLow_newEnv_2to2\hiro_high\hiro_high_1",
    ],
]
# EVENT_PATH_GROUPS: List[LanePathGroup] = [
#     [
#         r"D:\workspace\python\ecoHRL\logs\current\hiro_260703_highonly_uniOld_fixedHER_newEnv_2to0\hiro_high\hiro_high_1",
#         r"D:\workspace\python\ecoHRL\logs\current\hiro_260706_highonly_uniOld_fixedHER_newEnv_2to1\hiro_high\hiro_high_1",
#         r"D:\workspace\python\ecoHRL\logs\current\hiro_260630_highonly_pretrained_uniOld_fixedHER_newEnv_2to2\hiro_high\hiro_high_1",
#     ],
#     [
#         r"D:\workspace\python\ecoHRL\logs\current\hiro_260703_highonly_uniOld_noHER_newEnv_2to0\hiro_high\hiro_high_1",
#         r"D:\workspace\python\ecoHRL\logs\current\hiro_260703_highonly_uniOld_noHER_newEnv_2to1\hiro_high\hiro_high_1",
#         r"D:\workspace\python\ecoHRL\logs\current\hiro_260703_highonly_uniOld_noHER_newEnv_2to2\hiro_high\hiro_high_1",
#     ],
#     [
#         r"D:\workspace\python\ecoHRL\logs\current\hiro_260702_highonly_ruleReUni_newEnv_SLmpc_noaugObs_2to0\hiro_high\hiro_high_1",
#         r"D:\workspace\python\ecoHRL\logs\current\hiro_260702_highonly_ruleReUni_newEnv_SLmpc_noaugObs_2to1\hiro_high\hiro_high_1",
#         r"D:\workspace\python\ecoHRL\logs\current\hiro_260702_highonly_ruleReUni_newEnv_SLmpc_noaugObs_2to2\hiro_high\hiro_high_1",
#     ],
#     [
#         r"D:\workspace\python\ecoHRL\logs\current\hiro_260708_highonly_reUni_oldLow_newEnv_2to0\hiro_high\hiro_high_1",
#         r"D:\workspace\python\ecoHRL\logs\current\hiro_260708_highonly_reUni_oldLow_newEnv_2to1\hiro_high\hiro_high_1",
#         r"D:\workspace\python\ecoHRL\logs\current\hiro_260708_highonly_reUni_oldLow_newEnv_2to2\hiro_high\hiro_high_1",
#     ],
# ]

LANE_NAMES = ("lane2to0", "lane2to1", "lane2to2")

# Tag(s) to plot. Exact matching is attempted first, followed by suffix matching.
TAG_TO_PLOT: Union[str, List[str]] = [
    "rollout/ep_rew",
    "rollout/punctual_reward",
    "rollout/comfort_reward",
]

# Optional labels for each path group. Keep the same length as EVENT_PATH_GROUPS;
# use "" to derive a label from the first path automatically.
RUN_LABELS = [
    "SAC","SAC+SG","SAC+SG+LP","HRL",
    # "w/o. reGoal","w/o. FH_HER","w/o. lowerRL","HRL",
]

# Output directory for generated figures. Use a directory separate from the
# single-path script so figures from the two scripts cannot overwrite each other.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "tb_reward_curves_lane_mean")

# Whether to remove the reward components listed below from each lane's ep_rew
# before taking the three-lane mean.
ADJUST_EP_REW = True
EP_REW_COMPONENTS_TO_EXCLUDE = [
    "speed_ref_aux_reward",
    "goal_lane_dense_reward",
    "wrong_lane_terminal_penalty",
]

# Alignment mode for ep_rew adjustment:
# "exact": only subtract when steps are exactly the same.
# "previous": subtract the latest speed_ref_aux_reward at or before ep_rew step.
ALIGN_MODE = "exact"

# Alignment mode used when taking the mean of the three lane series:
# "reference": use lane2to0's original logged steps within the range covered by
#              all three runs. Interpolate lane2to1/lane2to2 at those steps. This
#              preserves the same sampling density as plot_tb_reward_curves.py.
# "interpolate": use the union of all three runs' logged steps within their
#                 shared range and linearly interpolate missing values.
# "exact": only average steps that occur in all three logs.
LANE_STEP_ALIGN_MODE = "reference"

# Optional y-axis limits per tag: {"tag": (ymin, ymax)}
Y_LIMITS = {
    "rollout/ep_rew": (0, 18),
    "rollout/punctual_reward": (0, 10),
    "rollout/comfort_reward": (-10, 0),
}

# Figure size config (inches). Full and short tag keys are both supported.
FIGSIZE_DEFAULT = (11, 6)
FIGSIZE_BY_TAG = {
    "rollout/ep_rew": (11, 11),
}

# The three-lane mean curve is smoothed with a trailing rolling window. The
# shadow band shows the rolling [min, max] of that mean curve.
SMOOTH_WINDOW_MODE = "ratio"
SMOOTH_WINDOW = 50
SMOOTH_WINDOW_RATIO = 0.01
SMOOTH_WINDOW_MIN = 10
SMOOTH_WINDOW_MAX = 200
WINDOW_BAND_ALPHA = 0.18
SMOOTH_LINE_WIDTH = 2.0
PLOT_RAW_CURVE = False
RAW_CURVE_ALPHA = 0.2

# Whether to pop up an interactive plot window.
SHOW_PLOT = False


def find_tag(available_tags: Sequence[str], target_tag: str) -> str:
    available = set(available_tags)
    if target_tag in available:
        return target_tag

    suffix = "/" + target_tag if "/" not in target_tag else target_tag
    matched = [tag for tag in available_tags if tag.endswith(suffix)]
    if len(matched) == 1:
        return matched[0]
    if len(matched) > 1:
        raise ValueError(
            f"Tag '{target_tag}' matches multiple scalar tags by suffix: {matched}. "
            "Please use a more specific TAG_TO_PLOT."
        )

    raise ValueError(
        f"Tag '{target_tag}' not found. Available tags: {list(available_tags)}"
    )


def load_series(event_acc: EventAccumulator, tag: str) -> ScalarSeries:
    events = event_acc.Scalars(tag)
    return [(int(e.step), float(e.value)) for e in events]


def auto_label(path_group: LanePathGroup) -> str:
    normalized = os.path.normpath(path_group[0])
    return os.path.basename(normalized) or normalized


def normalize_tags(tag_cfg: Union[str, List[str]]) -> List[str]:
    if isinstance(tag_cfg, str):
        tag = tag_cfg.strip()
        if not tag:
            raise ValueError("TAG_TO_PLOT string is empty.")
        return [tag]

    tags = [str(t).strip() for t in tag_cfg if str(t).strip()]
    if not tags:
        raise ValueError("TAG_TO_PLOT list is empty.")
    return tags


def subtract_component_exact(
    ep_rew: ScalarSeries,
    component: ScalarSeries,
) -> ScalarSeries:
    component_by_step = {step: value for step, value in component}
    return [
        (step, rew - component_by_step[step])
        for step, rew in ep_rew
        if step in component_by_step
    ]


def subtract_component_previous(
    ep_rew: ScalarSeries,
    component: ScalarSeries,
) -> ScalarSeries:
    if not component:
        return []

    result: ScalarSeries = []
    j = 0
    last_aux = None

    for step, rew in ep_rew:
        while j < len(component) and component[j][0] <= step:
            last_aux = component[j][1]
            j += 1
        if last_aux is not None:
            result.append((step, rew - last_aux))
    return result


def _sorted_unique_series(series: ScalarSeries) -> Tuple[np.ndarray, np.ndarray]:
    values_by_step = dict(series)
    sorted_points = sorted(values_by_step.items())
    x = np.asarray([point[0] for point in sorted_points], dtype=np.int64)
    y = np.asarray([point[1] for point in sorted_points], dtype=np.float64)
    return x, y


def mean_lane_series(lane_series: Sequence[ScalarSeries]) -> ScalarSeries:
    """Align three lane series by step and return their pointwise mean."""
    if len(lane_series) != len(LANE_NAMES):
        raise ValueError(f"Expected {len(LANE_NAMES)} lane series, got {len(lane_series)}.")

    mode = str(LANE_STEP_ALIGN_MODE).strip().lower()
    if mode == "exact":
        values_by_lane = [dict(series) for series in lane_series]
        common_steps = set(values_by_lane[0])
        for values in values_by_lane[1:]:
            common_steps.intersection_update(values)

        return [
            (
                step,
                float(np.mean([values[step] for values in values_by_lane])),
            )
            for step in sorted(common_steps)
        ]

    if mode in {"reference", "interpolate"}:
        arrays_by_lane = [_sorted_unique_series(series) for series in lane_series]
        if any(x.size == 0 for x, _ in arrays_by_lane):
            return []

        overlap_start = max(int(x[0]) for x, _ in arrays_by_lane)
        overlap_end = min(int(x[-1]) for x, _ in arrays_by_lane)
        if overlap_start > overlap_end:
            return []

        if mode == "reference":
            reference_steps = arrays_by_lane[0][0]
            target_steps = reference_steps[
                (reference_steps >= overlap_start) & (reference_steps <= overlap_end)
            ]
        else:
            target_steps = np.unique(
                np.concatenate(
                    [
                        x[(x >= overlap_start) & (x <= overlap_end)]
                        for x, _ in arrays_by_lane
                    ]
                )
            )
        if target_steps.size == 0:
            return []

        interpolated = np.vstack(
            [np.interp(target_steps, x, y) for x, y in arrays_by_lane]
        )
        mean_values = np.mean(interpolated, axis=0)
        return [
            (int(step), float(value))
            for step, value in zip(target_steps, mean_values)
        ]

    raise ValueError(
        "LANE_STEP_ALIGN_MODE must be 'reference', 'interpolate', or 'exact'."
    )


def is_ep_rew_tag(tag_name: str) -> bool:
    return tag_name == "ep_rew" or tag_name.endswith("/ep_rew")


def prepare_series_for_tag(
    event_acc: EventAccumulator,
    scalar_tags: Sequence[str],
    tag_cfg: str,
    event_path: str,
) -> Tuple[ScalarSeries, str]:
    resolved_tag = find_tag(scalar_tags, tag_cfg)
    series = load_series(event_acc, resolved_tag)
    if not series:
        raise ValueError(f"No scalar points for tag '{resolved_tag}' in: {event_path}")

    if ADJUST_EP_REW and is_ep_rew_tag(resolved_tag):
        for component_name in EP_REW_COMPONENTS_TO_EXCLUDE:
            component_tag = None
            for candidate in (f"rollout/{component_name}", component_name):
                try:
                    component_tag = find_tag(scalar_tags, candidate)
                    break
                except ValueError:
                    continue

            if component_tag is None:
                print(
                    f"[warn] {event_path}: {component_name} not found; "
                    "continuing without excluding it."
                )
                continue

            component_series = load_series(event_acc, component_tag)
            if ALIGN_MODE == "exact":
                adjusted_series = subtract_component_exact(series, component_series)
            else:
                adjusted_series = subtract_component_previous(series, component_series)

            if adjusted_series:
                series = adjusted_series
            else:
                print(
                    f"[warn] Excluding {component_name} produced no aligned "
                    f"ep_rew points in {event_path}; keeping the current ep_rew "
                    "series. You may try ALIGN_MODE='previous'."
                )

    return series, resolved_tag


def sanitize_tag_for_filename(tag: str) -> str:
    return tag.replace("/", "_").replace("\\", "_").replace(":", "_")


def get_y_limits(tag_name: str) -> Union[Tuple[float, float], None]:
    key = tag_name if tag_name in Y_LIMITS else tag_name.split("/")[-1]
    if key not in Y_LIMITS:
        return None

    y_lim = Y_LIMITS[key]
    if not isinstance(y_lim, (tuple, list)) or len(y_lim) != 2:
        raise ValueError(f"Y_LIMITS['{key}'] must be a tuple/list of length 2.")
    return float(y_lim[0]), float(y_lim[1])


def _normalize_figsize(
    value: Union[Tuple[float, float], List[float]],
    key_name: str,
) -> Tuple[float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{key_name} must be a tuple/list of length 2.")
    width = float(value[0])
    height = float(value[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"{key_name} values must be > 0.")
    return width, height


def get_figsize(tag_name: str) -> Tuple[float, float]:
    if tag_name in FIGSIZE_BY_TAG:
        return _normalize_figsize(
            FIGSIZE_BY_TAG[tag_name],
            f"FIGSIZE_BY_TAG['{tag_name}']",
        )

    short_name = tag_name.split("/")[-1]
    if short_name in FIGSIZE_BY_TAG:
        return _normalize_figsize(
            FIGSIZE_BY_TAG[short_name],
            f"FIGSIZE_BY_TAG['{short_name}']",
        )

    return _normalize_figsize(FIGSIZE_DEFAULT, "FIGSIZE_DEFAULT")


def smooth_with_window_bounds(
    series: ScalarSeries,
    window: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray([point[0] for point in series], dtype=np.int64)
    y = np.asarray([point[1] for point in series], dtype=np.float64)

    if y.size == 0:
        return x, y, y, y

    window = max(int(window), 1)
    if window == 1:
        return x, y, y, y

    indices = np.arange(y.size, dtype=np.int64)
    starts = np.maximum(indices - window + 1, 0)
    cumulative = np.empty(y.size + 1, dtype=np.float64)
    cumulative[0] = 0.0
    np.cumsum(y, out=cumulative[1:])
    counts = indices - starts + 1
    smooth = (cumulative[indices + 1] - cumulative[starts]) / counts

    # The first window-1 points use expanding windows, matching the original code.
    lower = np.minimum.accumulate(y)
    upper = np.maximum.accumulate(y)
    if y.size >= window:
        windows = np.lib.stride_tricks.sliding_window_view(y, window)
        lower[window - 1 :] = np.min(windows, axis=1)
        upper[window - 1 :] = np.max(windows, axis=1)

    return x, smooth, lower, upper


def resolve_smooth_window(num_points: int) -> int:
    num_points = max(int(num_points), 1)
    mode = str(SMOOTH_WINDOW_MODE).strip().lower()

    if mode == "fixed":
        return max(int(SMOOTH_WINDOW), 1)

    if mode == "ratio":
        window = int(round(num_points * float(SMOOTH_WINDOW_RATIO)))
        window = max(window, int(SMOOTH_WINDOW_MIN))
        window = min(window, int(SMOOTH_WINDOW_MAX))
        return max(window, 1)

    raise ValueError("SMOOTH_WINDOW_MODE must be 'fixed' or 'ratio'.")


def validate_config() -> None:
    if not EVENT_PATH_GROUPS:
        raise ValueError("EVENT_PATH_GROUPS is empty.")
    if RUN_LABELS and len(RUN_LABELS) != len(EVENT_PATH_GROUPS):
        raise ValueError("RUN_LABELS length must match EVENT_PATH_GROUPS length.")

    for group_idx, path_group in enumerate(EVENT_PATH_GROUPS):
        if len(path_group) != len(LANE_NAMES):
            raise ValueError(
                f"EVENT_PATH_GROUPS[{group_idx}] must contain exactly "
                f"{len(LANE_NAMES)} paths in lane2to0/lane2to1/lane2to2 order."
            )
        if any(not str(path).strip() for path in path_group):
            raise ValueError(f"EVENT_PATH_GROUPS[{group_idx}] contains an empty path.")

    if ALIGN_MODE not in {"exact", "previous"}:
        raise ValueError("ALIGN_MODE must be 'exact' or 'previous'.")
    if str(LANE_STEP_ALIGN_MODE).strip().lower() not in {
        "reference",
        "interpolate",
        "exact",
    }:
        raise ValueError(
            "LANE_STEP_ALIGN_MODE must be 'reference', 'interpolate', or 'exact'."
        )
    mode = str(SMOOTH_WINDOW_MODE).strip().lower()
    if mode == "fixed" and SMOOTH_WINDOW < 1:
        raise ValueError("SMOOTH_WINDOW must be >= 1 in fixed mode.")
    if mode == "ratio" and SMOOTH_WINDOW_RATIO <= 0:
        raise ValueError("SMOOTH_WINDOW_RATIO must be > 0 in ratio mode.")


def load_lane_series(event_path_raw: str, tag_cfg: str) -> Tuple[ScalarSeries, str]:
    event_path = os.path.abspath(event_path_raw)
    event_acc = EventAccumulator(event_path, size_guidance={"scalars": 0})
    event_acc.Reload()

    scalar_tags = event_acc.Tags().get("scalars", [])
    if not scalar_tags:
        raise ValueError(f"No scalar tags found in: {event_path}")

    return prepare_series_for_tag(event_acc, scalar_tags, tag_cfg, event_path)


def build_series_cache(
    event_paths: Sequence[str],
    tags_to_plot: Sequence[str],
) -> SeriesCache:
    cache: SeriesCache = {}
    for event_path_raw in event_paths:
        event_path = os.path.abspath(event_path_raw)
        if event_path in cache:
            continue

        print(f"[load] TensorBoard events: {event_path}")
        event_acc = EventAccumulator(event_path, size_guidance={"scalars": 0})
        event_acc.Reload()
        scalar_tags = event_acc.Tags().get("scalars", [])

        tag_cache: Dict[str, CachedSeriesResult] = {}
        if not scalar_tags:
            error = f"No scalar tags found in: {event_path}"
            for tag_cfg in tags_to_plot:
                tag_cache[tag_cfg] = error
        else:
            for tag_cfg in tags_to_plot:
                try:
                    series, resolved_tag = prepare_series_for_tag(
                        event_acc,
                        scalar_tags,
                        tag_cfg,
                        event_path,
                    )
                    tag_cache[tag_cfg] = (resolved_tag, series)
                except ValueError as exc:
                    tag_cache[tag_cfg] = str(exc)

        cache[event_path] = tag_cache
        print(f"[loaded] TensorBoard events: {event_path}")

    return cache


def main() -> None:
    validate_config()
    tags_to_plot = normalize_tags(TAG_TO_PLOT)
    all_event_paths = [
        event_path
        for path_group in EVENT_PATH_GROUPS
        for event_path in path_group
    ]
    series_cache = build_series_cache(all_event_paths, tags_to_plot)

    output_dir = os.path.abspath(OUTPUT_DIR or os.getcwd())
    os.makedirs(output_dir, exist_ok=True)
    total_saved = 0

    for tag_cfg in tags_to_plot:
        fig, ax = plt.subplots(figsize=get_figsize(tag_cfg))
        plotted_count = 0

        for group_idx, path_group in enumerate(EVENT_PATH_GROUPS):
            label = RUN_LABELS[group_idx].strip() if RUN_LABELS else ""
            if not label:
                label = auto_label(path_group)

            lane_series: List[ScalarSeries] = []
            group_valid = True

            for lane_name, event_path in zip(LANE_NAMES, path_group):
                event_path_abs = os.path.abspath(event_path)
                cached_result = series_cache[event_path_abs][tag_cfg]
                if isinstance(cached_result, str):
                    print(f"[skip] {label}/{lane_name}: {cached_result}")
                    group_valid = False
                    break
                resolved_tag, series = cached_result

                lane_series.append(series)
                print(
                    f"[ok] {label}/{lane_name}: using tag '{resolved_tag}', "
                    f"points={len(series)}"
                )

            if not group_valid:
                print(f"[skip] Curve '{label}' requires all three lane paths.")
                continue

            mean_series = mean_lane_series(lane_series)
            if not mean_series:
                print(
                    f"[skip] Curve '{label}' cannot align all three lanes for "
                    f"tag '{tag_cfg}' with LANE_STEP_ALIGN_MODE="
                    f"'{LANE_STEP_ALIGN_MODE}'."
                )
                continue

            smooth_window = resolve_smooth_window(len(mean_series))
            x, y_smooth, y_low, y_high = smooth_with_window_bounds(
                mean_series,
                smooth_window,
            )

            if PLOT_RAW_CURVE:
                ax.plot(
                    [point[0] for point in mean_series],
                    [point[1] for point in mean_series],
                    alpha=RAW_CURVE_ALPHA,
                    linewidth=1.0,
                )

            line = ax.plot(
                x,
                y_smooth,
                label=label,
                linewidth=SMOOTH_LINE_WIDTH,
            )[0]
            ax.fill_between(
                x,
                y_low,
                y_high,
                color=line.get_color(),
                alpha=WINDOW_BAND_ALPHA,
            )

            lane_counts = ", ".join(
                f"{name}={len(series)}"
                for name, series in zip(LANE_NAMES, lane_series)
            )
            print(
                f"[mean] {label}: aligned_points={len(mean_series)}, "
                f"lane_align={LANE_STEP_ALIGN_MODE}, "
                f"smooth_window={smooth_window}, source_points=({lane_counts})"
            )
            plotted_count += 1

        if plotted_count == 0:
            print(f"[skip] No valid curve for tag: {tag_cfg}")
            plt.close(fig)
            continue

        title = f"TensorBoard Tag Compare (3-Lane Mean): {tag_cfg}"
        if ADJUST_EP_REW and is_ep_rew_tag(tag_cfg):
            excluded = ", ".join(EP_REW_COMPONENTS_TO_EXCLUDE)
            title += f" (excluded: {excluded})"
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.set_ylabel("value")

        y_lim = get_y_limits(tag_cfg)
        if y_lim is not None:
            ax.set_ylim(y_lim)

        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()

        output_stem = os.path.join(
            output_dir,
            f"tb_tag_compare_lane_mean_{sanitize_tag_for_filename(tag_cfg)}",
        )
        png_path = output_stem + ".png"
        svg_path = output_stem + ".svg"
        fig.savefig(png_path, dpi=160)
        fig.savefig(svg_path, format="svg")
        print(f"Saved PNG figure to: {png_path}")
        print(f"Saved SVG figure to: {svg_path}")
        total_saved += 1

        if SHOW_PLOT:
            plt.show()
        else:
            plt.close(fig)

    if total_saved == 0:
        raise RuntimeError(
            "No figure was generated. Please check EVENT_PATH_GROUPS and TAG_TO_PLOT."
        )


if __name__ == "__main__":
    main()
