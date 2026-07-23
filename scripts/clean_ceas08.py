from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import pandas as pd

SPACES_RE = re.compile(r"[ \t]+")
SENDER_MISSING_RE = re.compile(r"^.*<\s*>$")
NA_VALUE = "NaN"


def normalize_whitespace(value: str) -> str:
    "we trim edges and collapse repeated the spaces or tabs while preserving line breaks"
    normalized_lines = [SPACES_RE.sub(" ", line).strip() for line in value.splitlines()]
    return "\n".join(normalized_lines).strip()


def normalize_field(value: str, *, is_sender: bool = False) -> str:
    normalized_value = normalize_whitespace(value)
    if not normalized_value:
        return NA_VALUE

    if is_sender and SENDER_MISSING_RE.match(normalized_value):
        return NA_VALUE

    return normalized_value


def parse_fields(input_path: Path) -> pd.DataFrame:
    "loaded only CEAS_08.csv to normalize every field except sender parsing"
    frame = pd.read_csv(input_path, dtype=str, keep_default_na=False)

    for column in frame.columns:
        frame[column] = frame[column].map(
            lambda value, current_column=column: (
                normalize_field(value, is_sender=current_column == "sender")
                if isinstance(value, str)
                else NA_VALUE
            )
        )

    if "date" in frame.columns:
        parsed_dates = pd.to_datetime(frame["date"], errors="coerce")
        frame["date"] = parsed_dates.dt.strftime("%Y-%m-%d %H:%M:%S").fillna(NA_VALUE)
    return frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Parsing CEAS_08 email logs except for sender parsing"
        )
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help="Path to the source CSV file.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional path for a parsed CSV output file.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    parsed_frame = parse_fields(args.input_csv)
    output_path = args.output_csv or args.input_csv.with_name("clean.csv")
    parsed_frame.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)


if __name__ == "__main__":
    main()
