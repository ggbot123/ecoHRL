import argparse
import csv
from pathlib import Path


def sample_csv_rows(
    input_path: Path,
    output_path: Path,
    every_n: int | None = None,
    max_rows: int | None = None,
) -> tuple[int, int]:
    """Keep header and sample data rows into a new CSV.

    If max_rows is provided, stop after writing at most that many data rows.
    If every_n is provided, keep one row every N rows from the source.
    When both are provided, max_rows is applied after sampling.
    """
    if every_n <= 0:
        raise ValueError("--every-n must be a positive integer")
    if max_rows is not None and max_rows <= 0:
        raise ValueError("--max-rows must be a positive integer")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    kept_rows = 0

    with input_path.open("r", encoding="utf-8", newline="") as src, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as dst:
        reader = csv.reader(src)
        writer = csv.writer(dst)

        header = next(reader, None)
        if header is None:
            return total_rows, kept_rows

        writer.writerow(header)
        for idx, row in enumerate(reader):
            total_rows += 1
            should_keep = True
            if every_n is not None:
                should_keep = (idx % every_n == 0)

            if should_keep:
                writer.writerow(row)
                kept_rows += 1

                if max_rows is not None and kept_rows >= max_rows:
                    break

    return total_rows, kept_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Keep CSV header and sample data rows into a smaller CSV."
    )
    parser.add_argument(
        "--input",
        default=r"D:\workspace\python\ecoHRL\logs\current\hiro_260412_lowonly_reUni_fixedHERsimp_amax3_dmin15_10\her_relabel_debug.csv",
        help="Input CSV path.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path. Default: <input stem>_sampled.csv in the same folder.",
    )
    parser.add_argument(
        "--every-n",
        type=int,
        default=100,
        help="Keep one row every N rows (default: 100). Use 1 to keep all rows.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional maximum number of data rows to write after sampling.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_name(f"{input_path.stem}_sampled.csv")

    total_rows, kept_rows = sample_csv_rows(
        input_path,
        output_path,
        every_n=args.every_n,
        max_rows=args.max_rows,
    )
    print(f"Input rows (excluding header): {total_rows}")
    print(f"Sampled rows written: {kept_rows}")
    print(f"Output file: {output_path}")


if __name__ == "__main__":
    main()
