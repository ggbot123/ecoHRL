import csv
import re
from pathlib import Path

import numpy as np


_STATE_COL_PATTERN = re.compile(r"^state_(\d+)$")

# ===== Edit parameters here =====
CSV_PATH = Path(r"d:\workspace\python\ecoHRL\models\eval_results\20260317_165130\hiro_ep_0006_trajectory.csv")
ROW_N = 177  # 1-based data row index (excluding header)
# Feature order used in state_* flattening.
FEATURE_NAMES = ["presence", "x", "y", "vx", "vy"]
# goal_rel in ego subspace [dx, dy, dvx, dvy], used to convert to absolute goal_phys.
# If None, goal_phys will be printed as None.
GOAL_REL = np.array([20.0, 0.0, 0.0, 0.0], dtype=np.float32)


def _load_row(csv_path: Path, row_n: int) -> dict:
    """Load the n-th data row (1-based, excluding header) from csv_path."""
    if row_n <= 0:
        raise ValueError(f"row_n must be >= 1, got {row_n}")

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            if i == row_n:
                return row

    raise IndexError(f"row_n={row_n} is out of range for file: {csv_path}")


def _extract_state_vector_from_row(row: dict) -> np.ndarray:
    state_items = []
    for k, v in row.items():
        m = _STATE_COL_PATTERN.match(k)
        if m is None:
            continue
        state_items.append((int(m.group(1)), float(v)))

    if not state_items:
        raise KeyError("No state_* columns found in CSV row.")

    state_items.sort(key=lambda x: x[0])
    return np.asarray([x[1] for x in state_items], dtype=np.float32)


def _to_abs_states(state_vec: np.ndarray, feature_names: list[str]):
    if state_vec.ndim != 1 or state_vec.size < 2:
        raise ValueError("state vector is invalid.")

    feat_dim = int(len(feature_names))
    kin_flat = state_vec[1:]
    if kin_flat.size % feat_dim != 0:
        raise ValueError(
            f"state length mismatch: kin_flat size={kin_flat.size}, feat_dim={feat_dim}."
        )

    kin = kin_flat.reshape(-1, feat_dim)
    if kin.shape[0] == 0:
        raise ValueError("No vehicle kinematics found in state vector.")

    idx_presence = int(feature_names.index("presence"))
    idx_x = int(feature_names.index("x"))
    idx_y = int(feature_names.index("y"))
    idx_vx = int(feature_names.index("vx"))
    idx_vy = int(feature_names.index("vy"))

    ego = kin[0]
    ego_state = np.array(
        [ego[idx_x], ego[idx_y], ego[idx_vx], ego[idx_vy]],
        dtype=np.float32,
    )

    neighbors_abs: list[np.ndarray] = []
    for i in range(1, kin.shape[0]):
        item = kin[i]
        if float(item[idx_presence]) <= 0.0:
            continue

        # Observation for non-ego vehicles is relative to ego when absolute=False (default).
        abs_state = np.array(
            [
                ego_state[0] + item[idx_x],
                ego_state[1] + item[idx_y],
                ego_state[2] + item[idx_vx],
                ego_state[3] + item[idx_vy],
            ],
            dtype=np.float32,
        )
        neighbors_abs.append(abs_state)

    return ego_state, neighbors_abs


def _format_vec4(vec: np.ndarray) -> str:
    vals = [float(vec[0]), float(vec[1]), float(vec[2]), float(vec[3])]
    return f"[{vals[0]:.6f}, {vals[1]:.6f}, {vals[2]:.6f}, {vals[3]:.6f}]"


def _build_goal_phys(ego_state: np.ndarray, goal_rel: np.ndarray | None) -> np.ndarray | None:
    if goal_rel is None:
        return None

    g = np.asarray(goal_rel, dtype=np.float32).reshape(-1)
    if g.size < 4:
        raise ValueError("GOAL_REL must contain 4 values: [dx, dy, dvx, dvy].")
    return (ego_state + g[:4]).astype(np.float32)



def main() -> None:
    row = _load_row(CSV_PATH, ROW_N)
    state_vec = _extract_state_vector_from_row(row)
    feature_names = list(FEATURE_NAMES)
    ego_state, neighbors_state = _to_abs_states(state_vec, feature_names)
    goal_phys = _build_goal_phys(ego_state, GOAL_REL)

    print(f"csv_path: {CSV_PATH}")
    print(f"row_n: {ROW_N}")
    print(f"state_dim: {state_vec.shape[0]}")
    print(f"features: {feature_names}")
    print(f"neighbors_present: {len(neighbors_state)}")

    print("\n# Copy to test_hiro_low.py")
    print(f"ego_state={_format_vec4(ego_state)},")

    print("neighbors_state=[")
    for n in neighbors_state:
        print(f"    {_format_vec4(n)},")
    print("],")

    if goal_phys is None:
        print("# GOAL_REL is None, set goal_phys manually")
        print("goal_phys=[0.0, 4.0, 0.0, 0.0],")
    else:
        print(f"goal_phys={_format_vec4(goal_phys)},")


if __name__ == "__main__":
    main()
