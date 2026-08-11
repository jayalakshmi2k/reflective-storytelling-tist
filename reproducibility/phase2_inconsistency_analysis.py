from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd


ISSUE_LABEL = (
    "Did you notice any issues or inconsistencies compared to the persona "
    "that you selected?"
)
BOTHER_LABEL = "How much did the issues bother you?"
EXPECTED_PAIRS = 8


def normalise_text(value: object) -> str:
    """Normalise whitespace and case for reliable label comparisons."""
    return re.sub(r"\s+", " ", str(value).strip()).casefold()


def parse_value_codes(value: object) -> dict[float, str]:
    """Parse codebook entries such as '1 = Yes\n2 = No\n3 = Not sure'."""
    mapping: dict[float, str] = {}
    if pd.isna(value):
        return mapping

    for line in str(value).splitlines():
        match = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*=\s*(.*?)\s*$", line)
        if match:
            mapping[float(match.group(1))] = match.group(2)
    return mapping


def read_inputs(codebook_path: Path, data_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    codebook = pd.read_csv(codebook_path)
    required = {"Variable Name", "Label", "Value Codes", "Missing Code"}
    missing = required.difference(codebook.columns)
    if missing:
        raise ValueError(
            "Codebook is missing required column(s): " + ", ".join(sorted(missing))
        )

    workbook = pd.ExcelFile(data_path)
    if "Data" not in workbook.sheet_names:
        raise ValueError(
            f"Excel file has no 'Data' sheet. Available sheets: {workbook.sheet_names}"
        )
    data = pd.read_excel(workbook, sheet_name="Data")
    return codebook, data


def variables_with_label(codebook: pd.DataFrame, label: str) -> list[str]:
    target = normalise_text(label)
    selected = codebook.loc[
        codebook["Label"].map(normalise_text).eq(target), "Variable Name"
    ]
    return selected.astype(str).tolist()


def missing_codes_for(codebook_indexed: pd.DataFrame, variables: list[str]) -> set[float]:
    codes: set[float] = set()
    for variable in variables:
        raw = codebook_indexed.at[variable, "Missing Code"]
        if not pd.isna(raw):
            try:
                codes.add(float(raw))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Non-numeric missing code for {variable}: {raw!r}"
                ) from exc
    return codes


def numeric_without_missing(series: pd.Series, missing_codes: set[float]) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if missing_codes:
        values = values.mask(values.isin(missing_codes))
    return values


def analyse(codebook: pd.DataFrame, data: pd.DataFrame) -> None:
    codebook_indexed = codebook.set_index("Variable Name", drop=False)
    if codebook_indexed.index.has_duplicates:
        duplicates = codebook_indexed.index[codebook_indexed.index.duplicated()].unique()
        raise ValueError(f"Duplicate variable names in codebook: {list(duplicates)}")

    issue_vars = variables_with_label(codebook, ISSUE_LABEL)
    bother_vars = variables_with_label(codebook, BOTHER_LABEL)

    if len(issue_vars) != EXPECTED_PAIRS or len(bother_vars) != EXPECTED_PAIRS:
        raise ValueError(
            "Expected exactly 8 inconsistency variables and 8 bother variables; "
            f"found {len(issue_vars)} and {len(bother_vars)}."
        )

    required_vars = issue_vars + bother_vars
    absent = [variable for variable in required_vars if variable not in data.columns]
    if absent:
        raise ValueError("Variables missing from the Data sheet: " + ", ".join(absent))

    issue_missing = missing_codes_for(codebook_indexed, issue_vars)
    bother_missing = missing_codes_for(codebook_indexed, bother_vars)

    # Confirm that every repeated issue item uses the same response coding.
    mappings = [parse_value_codes(codebook_indexed.at[v, "Value Codes"]) for v in issue_vars]
    if not mappings[0] or any(mapping != mappings[0] for mapping in mappings[1:]):
        raise ValueError("The eight inconsistency variables do not share one value-code mapping.")
    response_mapping = mappings[0]

    expected_labels = {"yes", "no", "not sure"}
    labels_found = {normalise_text(label) for label in response_mapping.values()}
    if labels_found != expected_labels:
        raise ValueError(
            "Expected response labels Yes, No, and Not sure; found "
            f"{sorted(response_mapping.values())}."
        )

    long_parts: list[pd.DataFrame] = []
    for narrative_number, (issue_var, bother_var) in enumerate(
        zip(issue_vars, bother_vars), start=1
    ):
        issue = numeric_without_missing(data[issue_var], issue_missing)
        bother = numeric_without_missing(data[bother_var], bother_missing)
        long_parts.append(
            pd.DataFrame(
                {
                    "row": data.index,
                    "narrative": narrative_number,
                    "issue_code": issue,
                    "bother_rating": bother,
                }
            )
        )

    long_data = pd.concat(long_parts, ignore_index=True)
    valid_issues = long_data.dropna(subset=["issue_code"]).copy()
    unknown_codes = sorted(set(valid_issues["issue_code"]) - set(response_mapping))
    if unknown_codes:
        raise ValueError(f"Unexpected inconsistency response code(s): {unknown_codes}")

    valid_issues["response"] = valid_issues["issue_code"].map(response_mapping)
    preferred_order = ["Yes", "No", "Not sure"]
    canonical = {normalise_text(v): v for v in response_mapping.values()}
    output_labels = [canonical[normalise_text(v)] for v in preferred_order]
    counts = valid_issues["response"].value_counts().reindex(output_labels, fill_value=0)
    total = int(counts.sum())

    yes_or_unsure_codes = {
        code
        for code, label in response_mapping.items()
        if normalise_text(label) in {"yes", "not sure"}
    }
    conditional = valid_issues.loc[
        valid_issues["issue_code"].isin(yes_or_unsure_codes), "bother_rating"
    ].dropna()

    print("Phase II narrative inconsistency analysis")
    print(f"Participants (rows): {len(data)}")
    print(f"Repeated narrative assessments per participant: {len(issue_vars)}")
    print(f"Expected assessments: {len(data) * len(issue_vars)}")
    print(f"Valid inconsistency assessments: {total}")
    print()
    print("Inconsistency responses")
    for label in output_labels:
        count = int(counts[label])
        percentage = 100.0 * count / total if total else float("nan")
        print(f"  {label}: {count}/{total} ({percentage:.1f}%)")

    yes_or_unsure_count = int(
        valid_issues["issue_code"].isin(yes_or_unsure_codes).sum()
    )
    yes_or_unsure_pct = 100.0 * yes_or_unsure_count / total if total else float("nan")
    print(f"  Yes or Not sure: {yes_or_unsure_count}/{total} ({yes_or_unsure_pct:.1f}%)")
    print()
    print("Bother ratings among Yes or Not sure assessments")
    print(f"  Eligible assessments: {yes_or_unsure_count}")
    print(f"  Valid bother ratings: {len(conditional)}")
    if conditional.empty:
        print("  No valid bother ratings available.")
    else:
        q1 = conditional.quantile(0.25)
        median = conditional.median()
        q3 = conditional.quantile(0.75)
        print(f"  Median: {median:g}")
        print(f"  IQR: {q1:g}-{q3:g}")
        print(f"  Range: {conditional.min():g}-{conditional.max():g}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce Phase II inconsistency and bother-rating results."
    )
    parser.add_argument("codebook", type=Path, help="Path to phase2_codebook.csv")
    parser.add_argument("data", type=Path, help="Path to phase2_data.xlsx")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        codebook, data = read_inputs(args.codebook, args.data)
        analyse(codebook, data)
    except (FileNotFoundError, OSError, ValueError, pd.errors.ParserError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
