import os
from typing import Dict, List, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ScalarSeries = List[Tuple[int, float]]
CachedSeriesResult = Union[Tuple[str, ScalarSeries], str]
SeriesCache = Dict[str, Dict[str, CachedSeriesResult]]


# ===== User Config =====
# Multiple TensorBoard event paths (each item can be an event file or a directory).
EVENT_PATHS = [
    r"D:\workspace\python\ecoHRL\logs\current\sac_260709_base_wronglanePen_oldEnv\SAC_1",
    r"D:\workspace\python\ecoHRL\logs\current\sac_260403_withPrior_SLv2_randomlane\SAC_1",
    r"D:\workspace\python\ecoHRL\logs\current\sac_260613_withPrior_oldEnv_randomto2_wronglanePen_1e7\SAC_1",
    r"D:\workspace\python\ecoHRL\logs\current\hiro_260331_highonly_reachableUniformLane1_Rainbow_amax3_dmin15_10_randomlane\hiro_high\hiro_high_1",
]

# EVENT_PATHS = [
#     r"D:\workspace\python\ecoHRL\logs\current\hiro_260628_highonly_pretrained_uni_oldEnv_fixedHER_SLmpc_noaugObs\hiro_high\hiro_high_1",
#     r"D:\workspace\python\ecoHRL\logs\current\hiro_260628_highonly_pretrained_uni_oldEnv_noHER_SLmpc_noaugObs\hiro_high\hiro_high_1",
#     r"D:\workspace\python\ecoHRL\logs\current\hiro_260706_highonly_ruleReUni_oldEnv\hiro_high\hiro_high_1",
#     # r"D:\workspace\python\ecoHRL\logs\current\hiro_260331_highonly_reachableUniformLane1_Rainbow_amax3_dmin15_10_randomlane\hiro_high\hiro_high_1",
#     r"D:\workspace\python\ecoHRL\logs\current\hiro_260709_highonly_reUni_oldEnv\hiro_high\hiro_high_1",
# ]

# Tag to plot on one figure for all EVENT_PATHS.
# Supports exact match first, then suffix fallback, e.g. "ep_rew" can match "rollout/ep_rew".
TAG_TO_PLOT: Union[str, List[str]] = [
    "rollout/ep_rew",
    "rollout/punctual_reward",
    "rollout/comfort_reward",
]

# Optional labels for each path. Keep same length as EVENT_PATHS; use "" to auto-name.
RUN_LABELS = [
    "SAC","SAC+SG","SAC+SG+LP","HRL",
    # "w/o. reGoal","w/o. FH_HER","w/o. lowerRL","HRL",
]

# Output directory for generated figures. Resolve from the project root so the
# result does not depend on the working directory used to launch this script.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "tb_reward_curves")

# Whether to remove the reward components listed below from ep_rew.
ADJUST_EP_REW = True
EP_REW_COMPONENTS_TO_EXCLUDE = [
    "speed_ref_aux_reward",
    "goal_lane_dense_reward",
    "wrong_lane_terminal_penalty",
]

# Alignment mode for ep_rew adjustment:
# "exact": only subtract when steps are exactly the same.
# "previous": subtract latest available speed_ref_aux_reward at or before ep_rew step.
ALIGN_MODE = "exact"

# Optional y-axis limits per tag: {"tag": (ymin, ymax)}
Y_LIMITS = {
    "rollout/ep_rew":(0, 18),
    "rollout/punctual_reward":(0, 10),
    "rollout/comfort_reward":(-10, 0),
}

# Figure size config (inches).
# Use FIGSIZE_BY_TAG to override size for specific tags.
# Supports full tag key (e.g. "rollout/ep_rew") or short key (e.g. "ep_rew").
FIGSIZE_DEFAULT = (11, 6)
FIGSIZE_BY_TAG = {
    "rollout/ep_rew": (11, 11),
}

# Moving-window smoothing and shadow band options.
# For each curve, the plotted line is rolling-mean; shadow band is rolling [min, max].
# SMOOTH_WINDOW_MODE:
# - "fixed": use SMOOTH_WINDOW for every curve.
# - "ratio": window = clip(round(num_points * SMOOTH_WINDOW_RATIO), SMOOTH_WINDOW_MIN, SMOOTH_WINDOW_MAX)
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


def auto_label(path: str) -> str:
    normalized = os.path.normpath(path)
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


def subtract_component_exact(ep_rew: ScalarSeries, component: ScalarSeries) -> ScalarSeries:
    component_by_step = {step: value for step, value in component}
    result: ScalarSeries = []
    for step, rew in ep_rew:
        if step in component_by_step:
            result.append((step, rew - component_by_step[step]))
    return result


def subtract_component_previous(ep_rew: ScalarSeries, component: ScalarSeries) -> ScalarSeries:
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


def is_ep_rew_tag(tag_name: str) -> bool:
    return tag_name == "ep_rew" or tag_name.endswith("/ep_rew")


def prepare_series_for_tag(
    event_acc: EventAccumulator,
    scalar_tags: Sequence[str],
    tag_cfg: str,
    event_path: str,
) -> Tuple[str, ScalarSeries]:
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
                    f"ep_rew points in {event_path}; keeping the current "
                    "ep_rew series. You may try ALIGN_MODE='previous'."
                )

    return resolved_tag, series


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
                    tag_cache[tag_cfg] = prepare_series_for_tag(
                        event_acc,
                        scalar_tags,
                        tag_cfg,
                        event_path,
                    )
                except ValueError as exc:
                    tag_cache[tag_cfg] = str(exc)

        cache[event_path] = tag_cache
        print(f"[loaded] TensorBoard events: {event_path}")

    return cache


def sanitize_tag_for_filename(tag: str) -> str:
    return tag.replace("/", "_").replace("\\", "_").replace(":", "_")


def get_y_limits(tag_name: str) -> Union[Tuple[float, float], None]:
    if tag_name in Y_LIMITS:
        y_lim = Y_LIMITS[tag_name]
        if isinstance(y_lim, (tuple, list)) and len(y_lim) == 2:
            return float(y_lim[0]), float(y_lim[1])
        raise ValueError(f"Y_LIMITS['{tag_name}'] must be a tuple/list of length 2.")

    short_name = tag_name.split("/")[-1]
    if short_name in Y_LIMITS:
        y_lim = Y_LIMITS[short_name]
        if isinstance(y_lim, (tuple, list)) and len(y_lim) == 2:
            return float(y_lim[0]), float(y_lim[1])
        raise ValueError(f"Y_LIMITS['{short_name}'] must be a tuple/list of length 2.")

    return None


def _normalize_figsize(value: Union[Tuple[float, float], List[float]], key_name: str) -> Tuple[float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{key_name} must be a tuple/list of length 2.")
    w = float(value[0])
    h = float(value[1])
    if w <= 0 or h <= 0:
        raise ValueError(f"{key_name} values must be > 0.")
    return w, h


def get_figsize(tag_name: str) -> Tuple[float, float]:
    if tag_name in FIGSIZE_BY_TAG:
        return _normalize_figsize(FIGSIZE_BY_TAG[tag_name], f"FIGSIZE_BY_TAG['{tag_name}']")

    short_name = tag_name.split("/")[-1]
    if short_name in FIGSIZE_BY_TAG:
        return _normalize_figsize(FIGSIZE_BY_TAG[short_name], f"FIGSIZE_BY_TAG['{short_name}']")

    return _normalize_figsize(FIGSIZE_DEFAULT, "FIGSIZE_DEFAULT")


def smooth_with_window_bounds(series: ScalarSeries, window: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray([p[0] for p in series], dtype=np.int64)
    y = np.asarray([p[1] for p in series], dtype=np.float64)

    if y.size == 0:
        return x, y, y, y

    w = max(int(window), 1)
    if w == 1:
        return x, y, y, y

    indices = np.arange(y.size, dtype=np.int64)
    starts = np.maximum(indices - w + 1, 0)
    cumulative = np.empty(y.size + 1, dtype=np.float64)
    cumulative[0] = 0.0
    np.cumsum(y, out=cumulative[1:])
    counts = indices - starts + 1
    smooth = (cumulative[indices + 1] - cumulative[starts]) / counts

    # The first w-1 points use expanding windows, matching the original code.
    lower = np.minimum.accumulate(y)
    upper = np.maximum.accumulate(y)
    if y.size >= w:
        windows = np.lib.stride_tricks.sliding_window_view(y, w)
        lower[w - 1 :] = np.min(windows, axis=1)
        upper[w - 1 :] = np.max(windows, axis=1)

    return x, smooth, lower, upper


def resolve_smooth_window(num_points: int) -> int:
    n = max(int(num_points), 1)
    mode = str(SMOOTH_WINDOW_MODE).strip().lower()

    if mode == "fixed":
        return max(int(SMOOTH_WINDOW), 1)

    if mode == "ratio":
        ratio = float(SMOOTH_WINDOW_RATIO)
        w = int(round(n * ratio))
        w = max(w, int(SMOOTH_WINDOW_MIN))
        w = min(w, int(SMOOTH_WINDOW_MAX))
        return max(w, 1)

    raise ValueError("SMOOTH_WINDOW_MODE must be 'fixed' or 'ratio'.")


def main() -> None:
    if not EVENT_PATHS:
        raise ValueError("EVENT_PATHS is empty.")
    if RUN_LABELS and len(RUN_LABELS) != len(EVENT_PATHS):
        raise ValueError("RUN_LABELS length must match EVENT_PATHS length.")
    if ALIGN_MODE not in {"exact", "previous"}:
        raise ValueError("ALIGN_MODE must be 'exact' or 'previous'.")
    if str(SMOOTH_WINDOW_MODE).strip().lower() == "fixed" and SMOOTH_WINDOW < 1:
        raise ValueError("SMOOTH_WINDOW must be >= 1 when SMOOTH_WINDOW_MODE='fixed'.")
    if str(SMOOTH_WINDOW_MODE).strip().lower() == "ratio" and SMOOTH_WINDOW_RATIO <= 0:
        raise ValueError("SMOOTH_WINDOW_RATIO must be > 0 when SMOOTH_WINDOW_MODE='ratio'.")
    tags_to_plot = normalize_tags(TAG_TO_PLOT)
    series_cache = build_series_cache(EVENT_PATHS, tags_to_plot)

    output_dir = OUTPUT_DIR
    if not output_dir:
        output_dir = os.getcwd()
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    total_saved = 0

    for tag_cfg in tags_to_plot:
        fig, ax = plt.subplots(figsize=get_figsize(tag_cfg))
        plotted_count = 0

        for idx, event_path_raw in enumerate(EVENT_PATHS):
            event_path = os.path.abspath(event_path_raw)

            label = ""
            if RUN_LABELS:
                label = RUN_LABELS[idx].strip()
            if not label:
                label = auto_label(event_path)

            cached_result = series_cache[event_path][tag_cfg]
            if isinstance(cached_result, str):
                print(f"[skip] {event_path}: {cached_result}")
                continue
            resolved_tag, series = cached_result

            smooth_window = resolve_smooth_window(len(series))
            x, y_smooth, y_low, y_high = smooth_with_window_bounds(series, smooth_window)

            if PLOT_RAW_CURVE:
                ax.plot(
                    [p[0] for p in series],
                    [p[1] for p in series],
                    alpha=RAW_CURVE_ALPHA,
                    linewidth=1.0,
                )

            line = ax.plot(x, y_smooth, label=label, linewidth=SMOOTH_LINE_WIDTH)[0]
            ax.fill_between(
                x,
                y_low,
                y_high,
                color=line.get_color(),
                alpha=WINDOW_BAND_ALPHA,
            )

            print(
                f"[ok] {event_path}: using tag '{resolved_tag}', "
                f"points={len(series)}, smooth_window={smooth_window}"
            )
            plotted_count += 1

        if plotted_count == 0:
            print(f"[skip] No valid curve for tag: {tag_cfg}")
            plt.close(fig)
            continue

        title = f"TensorBoard Tag Compare: {tag_cfg}"
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
            f"tb_tag_compare_{sanitize_tag_for_filename(tag_cfg)}",
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
        raise RuntimeError("No figure was generated. Please check EVENT_PATHS and TAG_TO_PLOT.")


if __name__ == "__main__":
    main()
