from __future__ import annotations

import argparse
from pathlib import Path

from util.plot_result import batch_render_high_interval_debug_csv


def _find_latest_csv(workspace: Path, explicit_csv: Path | None) -> Path:
    if explicit_csv is not None:
        if not explicit_csv.exists():
            raise FileNotFoundError(f"CSV file not found: {explicit_csv}")
        return explicit_csv

    candidates = list(workspace.rglob("high_interval_debug.csv"))
    if not candidates:
        raise FileNotFoundError(
            "No high_interval_debug.csv found under workspace. "
            "Please pass --csv with an explicit path."
        )
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the last N high-interval debug samples from high_interval_debug.csv and save images "
            "under debug/<datetime>/"
        )
    )
    parser.add_argument("--csv", type=str, default="./logs/current/hiro_260322_highonly_reachableUniform_newSLv2_vio03_HER_reDim_lc10_amax3_dmin10_8/high_interval_debug.csv")
    # parser.add_argument("--csv", type=str, default="./logs/current/hiro_260322_highonly_reachableUniform_newSLv2_vio03_HER_reDim_lc10_amax3_dmin0/high_interval_debug.csv")
    # parser.add_argument("--csv", type=str, default="./logs/current/hiro_260320_debug/high_interval_debug.csv", help="Path to high_interval_debug.csv")
    parser.add_argument("--last", type=int, default=100, help="Number of latest rows to plot")
    parser.add_argument("--workspace", type=str, default=".", help="Workspace root used for auto CSV search")
    parser.add_argument("--debug-root", type=str, default="./debug", help="Root debug output directory")
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    explicit_csv = Path(args.csv).resolve() if args.csv else None
    csv_path = _find_latest_csv(workspace, explicit_csv)

    out_dir = batch_render_high_interval_debug_csv(
        csv_path=str(csv_path),
        debug_root=str(Path(args.debug_root).resolve()),
        n_last=int(args.last),
    )
    print(f"CSV used: {csv_path}")
    print(f"Saved recent high-interval debug figures to: {out_dir}")


if __name__ == "__main__":
    main()
