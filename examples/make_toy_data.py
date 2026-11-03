"""Generate a small, purely synthetic merged household-individual CSV.

The file follows the layout that ``hm_popsyn.io.prepare_from_merged_table`` expects
with its default arguments (and therefore the layout ``hm-popsyn-run`` reads):

    col 0      household id
    col 1      individual id within the household
    cols 2-5   household attributes; col 5 is the household size (number of rows
               that share the household id), used by the generator as a count
    cols 6-8   individual attributes
    col 9      not used by the default column mapping (filled with 0)
    cols 10-11 individual attributes

All values are random integer codes starting at 1. They carry no meaning and are
not derived from any real data. The first line is a header, which ``load_csv`` skips.

Usage:
    python examples/make_toy_data.py [--output PATH] [--n-households N] [--seed S]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

HEADER = [
    "hh_id",
    "person_id",
    "hh_attr1",
    "hh_attr2",
    "hh_attr3",
    "hh_size",
    "person_attr1",
    "person_attr2",
    "person_attr3",
    "unused",
    "person_attr4",
    "person_attr5",
]

# Number of categories for each random attribute (codes are drawn from 1..k).
HH_CATEGORIES = (3, 4, 2)
PERSON_CATEGORIES = (6, 4, 3, 2, 5)
# Household sizes 1..5 and the probabilities used to draw them.
SIZE_PROBS = (0.25, 0.30, 0.20, 0.15, 0.10)


def make_rows(n_households: int, seed: int) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    sizes = rng.choice(np.arange(1, len(SIZE_PROBS) + 1), size=n_households, p=SIZE_PROBS)
    sizes[0] = 1  # guarantee size 1 is present so the min-shift in io.py keeps sizes unchanged

    rows: list[list[int]] = []
    for hh in range(n_households):
        hh_attrs = [int(rng.integers(1, k + 1)) for k in HH_CATEGORIES]
        size = int(sizes[hh])
        for person in range(size):
            p = [int(rng.integers(1, k + 1)) for k in PERSON_CATEGORIES]
            rows.append([hh + 1, person + 1, *hh_attrs, size, p[0], p[1], p[2], 0, p[3], p[4]])
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "toy_merged.csv"),
        help="Output CSV path (default: examples/toy_merged.csv next to this script)",
    )
    parser.add_argument("--n-households", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2018)
    args = parser.parse_args(argv)

    rows = make_rows(args.n_households, args.seed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(HEADER)
        writer.writerows(rows)

    try:
        shown = out.resolve().relative_to(Path.cwd())
    except ValueError:
        shown = out
    print(f"Wrote {len(rows)} individuals in {args.n_households} households to {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
