from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


# In the questionnaire, each dialogue prompt was presented first in its Low
# version and then in its High version. The eight repeated item blocks retain
# that order in the codebook, so positions 1, 3, 5, 7 are Low and 2, 4, 6, 8
# are High.
CONDITIONS = ["Low", "High"] * 4


def read_table(path: Path, preferred_sheet: str | None = None) -> pd.DataFrame:
    """Read CSV or Excel input without changing the source file."""
    suffix = path.suffix.lower()

    if suffix in {".xlsx", ".xls"}:
        workbook = pd.ExcelFile(path)
        sheet = (
            preferred_sheet
            if preferred_sheet in workbook.sheet_names
            else workbook.sheet_names[0]
        )
        return pd.read_excel(path, sheet_name=sheet)

    if suffix in {".csv", ".txt", ".tsv"}:
        separator = "\t" if suffix == ".tsv" else ","
        for encoding in ("utf-8-sig", "cp1252", "latin1"):
            try:
                return pd.read_csv(path, sep=separator, encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Could not decode {path}")

    raise ValueError(f"Unsupported file type: {path.suffix}")


def variables_with_label(codebook: pd.DataFrame, label: str) -> list[str]:
    """Return variable names matching a codebook label, in questionnaire order."""
    required = {"Variable Name", "Label"}

    if not required.issubset(codebook.columns):
        raise ValueError(f"Codebook must contain columns: {sorted(required)}")

    labels = codebook["Label"].astype(str).str.strip()
    variables = (
        codebook.loc[labels.eq(label), "Variable Name"]
        .astype(str)
        .tolist()
    )

    return variables


def require_variable_count(
    name: str,
    variables: list[str],
    expected: int
) -> None:
    """Check questionnaire structure defined by the codebook."""
    if len(variables) != expected:
        raise ValueError(
            f"Expected {expected} codebook variables for '{name}', "
            f"found {len(variables)}: {variables}"
        )


def add_row(
    rows: list[dict],
    analysis: str,
    outcome: str,
    condition: str = "",
    n: int | float | None = None,
    value: int | float | str | None = None,
    statistic: str = "",
    p_value: float | None = None,
    notes: str = "",
) -> None:
    rows.append(
        {
            "analysis": analysis,
            "outcome": outcome,
            "condition": condition,
            "n": n,
            "value": value,
            "statistic": statistic,
            "p_value": p_value,
            "notes": notes,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "data_file",
        type=Path,
        help="Final Phase II data file (CSV or XLSX)",
    )
    parser.add_argument(
        "codebook_file",
        type=Path,
        help="Phase II variable codebook (CSV or XLSX)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase2"),
    )
    parser.add_argument(
        "--participant-id",
        default="ID",
        help="Participant identifier column (default: ID)",
    )

    args = parser.parse_args()

    data = read_table(args.data_file, preferred_sheet="Data")
    codebook = read_table(args.codebook_file, preferred_sheet="VariableView")

    if args.participant_id not in data.columns:
        raise ValueError(
            f"Participant ID column '{args.participant_id}' was not found"
        )

    if data[args.participant_id].isna().any():
        raise ValueError("Participant IDs must be complete")

    if data[args.participant_id].duplicated().any():
        raise ValueError("Participant IDs must be unique")

    # ------------------------------------------------------------------
    # Map questionnaire variables through the codebook
    # ------------------------------------------------------------------

    relevance_vars = variables_with_label(
        codebook,
        "How relevant were these narratives to you?",
    )

    appreciation_vars = variables_with_label(
        codebook,
        "How much did you like the narrative?",
    )

    creativity_vars = variables_with_label(
        codebook,
        "How creative was the narrative?",
    )

    category_vars = variables_with_label(
        codebook,
        "Was the level of creativity:",
    )

    inconsistency_vars = variables_with_label(
        codebook,
        "Did you notice any issues or inconsistencies compared to the persona that you selected?",
    )

    disturbance_vars = variables_with_label(
        codebook,
        "How much did the issues bother you?",
    )

    cultural_vars = variables_with_label(
        codebook,
        "Did the narratives feel culturally relatable to you?",
    )

    future_vars = variables_with_label(
        codebook,
        "Would you like to use a such a functionality in an app in the future?",
    )

    # These checks concern questionnaire structure, not expected study results.
    for label, variables, expected in (
        ("overall relevance", relevance_vars, 1),
        ("narrative appreciation", appreciation_vars, 8),
        ("perceived creativity", creativity_vars, 8),
        ("categorical creativity", category_vars, 8),
        ("participant-perceived inconsistencies", inconsistency_vars, 8),
        ("disturbance", disturbance_vars, 8),
        ("cultural relatability", cultural_vars, 1),
        ("future-use intention", future_vars, 1),
    ):
        require_variable_count(label, variables, expected)

        missing_columns = [
            column for column in variables
            if column not in data.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Mapped variables absent from data for '{label}': "
                f"{missing_columns}"
            )

    # ------------------------------------------------------------------
    # Derive dataset size from input data
    # ------------------------------------------------------------------

    participant_n = data[args.participant_id].nunique()
    evaluations_per_participant = len(appreciation_vars)
    evaluation_n = participant_n * evaluations_per_participant

    if evaluations_per_participant != len(CONDITIONS):
        raise ValueError(
            f"Expected {len(CONDITIONS)} repeated narrative evaluations "
            f"from the questionnaire structure, found "
            f"{evaluations_per_participant}"
        )

    # ------------------------------------------------------------------
    # Cultural relatability and future-use intention
    # ------------------------------------------------------------------

    yes_maybe_no = {
        1: "Yes",
        2: "Maybe",
        3: "No",
    }

    cultural_raw = pd.to_numeric(
        data[cultural_vars[0]],
        errors="coerce",
    )

    future_raw = pd.to_numeric(
        data[future_vars[0]],
        errors="coerce",
    )

    if not cultural_raw.dropna().isin(yes_maybe_no.keys()).all():
        raise ValueError(
            "Cultural-relatability responses contain unexpected values"
        )

    if not future_raw.dropna().isin(yes_maybe_no.keys()).all():
        raise ValueError(
            "Future-use responses contain unexpected values"
        )

    cultural_counts = (
        cultural_raw
        .map(yes_maybe_no)
        .value_counts()
        .to_dict()
    )

    future_counts = (
        future_raw
        .map(yes_maybe_no)
        .value_counts()
        .to_dict()
    )

    rows: list[dict] = []

    add_row(
        rows,
        "dataset_validation",
        "participants",
        n=participant_n,
        value=participant_n,
    )

    add_row(
        rows,
        "dataset_validation",
        "evaluations_per_participant",
        n=participant_n,
        value=evaluations_per_participant,
    )

    add_row(
        rows,
        "dataset_validation",
        "narrative_evaluations",
        n=participant_n,
        value=evaluation_n,
    )

    for outcome, counts in (
        ("cultural_relatability", cultural_counts),
        ("future_use", future_counts),
    ):
        for response in ("Yes", "Maybe", "No"):
            add_row(
                rows,
                "dataset_validation",
                outcome,
                response,
                participant_n,
                counts.get(response, 0),
            )

    # ------------------------------------------------------------------
    # Overall relevance
    # ------------------------------------------------------------------

    relevance = pd.to_numeric(
        data[relevance_vars[0]],
        errors="coerce",
    ).dropna()

    add_row(
        rows,
        "overall_relevance",
        "valid_responses",
        n=len(relevance),
        value=len(relevance),
        notes="One post-study item per participant",
    )

    add_row(
        rows,
        "overall_relevance",
        "mean",
        n=len(relevance),
        value=relevance.mean(),
    )

    add_row(
        rows,
        "overall_relevance",
        "median",
        n=len(relevance),
        value=relevance.median(),
    )

    add_row(
        rows,
        "overall_relevance",
        "minimum",
        n=len(relevance),
        value=relevance.min(),
    )

    add_row(
        rows,
        "overall_relevance",
        "maximum",
        n=len(relevance),
        value=relevance.max(),
    )

    # ------------------------------------------------------------------
    # Paired Low-vs-High analyses
    # ------------------------------------------------------------------

    def paired_condition_analysis(
        variables: list[str],
        analysis_name: str,
    ) -> tuple[pd.Series, pd.Series, object]:

        numeric = data[variables].apply(
            pd.to_numeric,
            errors="coerce",
        )

        numeric.columns = CONDITIONS

        # Each participant contributes four Low and four High ratings.
        # Statistical testing uses participant-level condition means,
        # not 440 independent observations.
        low = numeric.loc[:, numeric.columns == "Low"].mean(axis=1)
        high = numeric.loc[:, numeric.columns == "High"].mean(axis=1)

        paired = pd.DataFrame(
            {
                "Low": low,
                "High": high,
            }
        ).dropna()

        test = wilcoxon(
            paired["Low"],
            paired["High"],
            alternative="two-sided",
            method="auto",
        )

        add_row(
            rows,
            analysis_name,
            "condition_mean",
            "Low",
            len(paired),
            paired["Low"].mean(),
        )

        add_row(
            rows,
            analysis_name,
            "condition_mean",
            "High",
            len(paired),
            paired["High"].mean(),
        )

        add_row(
            rows,
            analysis_name,
            "paired_difference_mean",
            "Low minus High",
            len(paired),
            (paired["Low"] - paired["High"]).mean(),
        )

        add_row(
            rows,
            analysis_name,
            "paired_difference_median",
            "Low minus High",
            len(paired),
            (paired["Low"] - paired["High"]).median(),
        )

        add_row(
            rows,
            analysis_name,
            "paired_wilcoxon",
            "Low vs High",
            len(paired),
            statistic="W",
            value=float(test.statistic),
            p_value=float(test.pvalue),
            notes="Two-sided; scipy method='auto'",
        )

        return paired["Low"], paired["High"], test

    creativity_low, creativity_high, creativity_test = (
        paired_condition_analysis(
            creativity_vars,
            "perceived_creativity",
        )
    )

    appreciation_low, appreciation_high, appreciation_test = (
        paired_condition_analysis(
            appreciation_vars,
            "narrative_appreciation",
        )
    )

    # ------------------------------------------------------------------
    # Categorical creativity
    # ------------------------------------------------------------------

    category_long = data[category_vars].copy()
    category_long.columns = CONDITIONS

    category_long = (
        category_long
        .reset_index(names="row")
        .melt(
            id_vars="row",
            var_name="condition",
            value_name="code",
        )
    )

    category_long["code"] = pd.to_numeric(
        category_long["code"],
        errors="coerce",
    )

    invalid_999 = int(
        (category_long["code"] == 999).sum()
    )

    category_valid = category_long[
        category_long["code"].isin([1, 2, 3, 4])
    ].copy()

    accounted_category_responses = (
        len(category_valid) + invalid_999
    )

    if accounted_category_responses != evaluation_n:
        raise ValueError(
            "Categorical creativity responses do not account for "
            f"all evaluations: expected {evaluation_n}, found "
            f"{accounted_category_responses}"
        )

    category_labels = {
        1: "Appealing",
        2: "Appropriate",
        3: "Too much",
        4: "Too little",
    }

    add_row(
        rows,
        "categorical_creativity",
        "excluded_999",
        n=evaluation_n,
        value=invalid_999,
    )

    add_row(
        rows,
        "categorical_creativity",
        "valid_evaluations",
        n=len(category_valid),
        value=len(category_valid),
    )

    for code, label in category_labels.items():
        count = int(
            (category_valid["code"] == code).sum()
        )

        add_row(
            rows,
            "categorical_creativity",
            label,
            "Overall",
            len(category_valid),
            count,
            notes=f"{100 * count / len(category_valid):.1f}%",
        )

        for condition in ("Low", "High"):
            condition_data = category_valid[
                category_valid["condition"] == condition
            ]

            condition_count = int(
                (condition_data["code"] == code).sum()
            )

            add_row(
                rows,
                "categorical_creativity",
                label,
                condition,
                len(condition_data),
                condition_count,
                notes=(
                    f"{100 * condition_count / len(condition_data):.1f}%"
                ),
            )

    # ------------------------------------------------------------------
    # Participant-perceived inconsistencies
    # ------------------------------------------------------------------

    inconsistency_long = pd.concat(
        [
            pd.to_numeric(
                data[column],
                errors="coerce",
            )
            for column in inconsistency_vars
        ],
        ignore_index=True,
    )

    inconsistency_labels = {
        1: "Yes",
        2: "No",
        3: "Not sure",
    }

    valid_inconsistency = inconsistency_long[
        inconsistency_long.isin(inconsistency_labels)
    ]

    if len(valid_inconsistency) != evaluation_n:
        raise ValueError(
            f"Expected {evaluation_n} valid inconsistency responses, "
            f"found {len(valid_inconsistency)}"
        )

    for code, label in inconsistency_labels.items():
        count = int(
            (valid_inconsistency == code).sum()
        )

        add_row(
            rows,
            "participant_perceived_inconsistencies",
            label,
            "Overall",
            len(valid_inconsistency),
            count,
            notes=(
                f"{100 * count / len(valid_inconsistency):.1f}%"
            ),
        )

    # ------------------------------------------------------------------
    # Disturbance ratings
    # ------------------------------------------------------------------

    disturbance_values: list[float] = []
    missing_disturbance = 0

    for inconsistency_column, disturbance_column in zip(
        inconsistency_vars,
        disturbance_vars,
    ):
        yes_mask = pd.to_numeric(
            data[inconsistency_column],
            errors="coerce",
        ).eq(1)

        scores = pd.to_numeric(
            data.loc[yes_mask, disturbance_column],
            errors="coerce",
        )

        valid_scores = scores[
            scores.between(1, 5)
        ]

        disturbance_values.extend(
            valid_scores.tolist()
        )

        missing_disturbance += int(
            len(scores) - len(valid_scores)
        )

    disturbance = pd.Series(
        disturbance_values,
        dtype=float,
    )

    add_row(
        rows,
        "disturbance",
        "valid_ratings",
        n=len(disturbance),
        value=len(disturbance),
        notes=(
            "Restricted to Yes inconsistency responses; "
            f"{missing_disturbance} missing/invalid ratings excluded"
        ),
    )

    add_row(
        rows,
        "disturbance",
        "mean",
        n=len(disturbance),
        value=disturbance.mean(),
    )

    add_row(
        rows,
        "disturbance",
        "median",
        n=len(disturbance),
        value=disturbance.median(),
    )

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        args.output_dir / "phase2_summary.csv"
    )

    pd.DataFrame(rows).to_csv(
        output_path,
        index=False,
    )

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------

    print("Phase II reproducibility summary")

    print(
        f"Participants: {participant_n}"
    )

    print(
        f"Narrative evaluations: {evaluation_n} "
        f"({evaluations_per_participant} per participant)"
    )

    print(
        f"Cultural relatability: {cultural_counts}"
    )

    print(
        f"Future-use intention: {future_counts}"
    )

    print(
        f"Overall relevance: N={len(relevance)}, "
        f"mean={relevance.mean():.2f}, "
        f"median={relevance.median():.0f}, "
        f"range={relevance.min():.0f}-{relevance.max():.0f}"
    )

    print(
        f"Perceived creativity: "
        f"N={len(creativity_low)}, "
        f"Low={creativity_low.mean():.2f}, "
        f"High={creativity_high.mean():.2f}, "
        f"W={creativity_test.statistic:.0f}, "
        f"p={creativity_test.pvalue:.6f}"
    )

    print(
        f"Narrative appreciation: "
        f"N={len(appreciation_low)}, "
        f"Low={appreciation_low.mean():.2f}, "
        f"High={appreciation_high.mean():.2f}, "
        f"W={appreciation_test.statistic:.0f}, "
        f"p={appreciation_test.pvalue:.6f}"
    )

    overall_categories = {
        label: int(
            (category_valid["code"] == code).sum()
        )
        for code, label in category_labels.items()
    }

    print(
        f"Categorical creativity "
        f"(valid N={len(category_valid)}, "
        f"excluded 999={invalid_999}): "
        f"{overall_categories}"
    )

    overall_inconsistencies = {
        label: int(
            (valid_inconsistency == code).sum()
        )
        for code, label in inconsistency_labels.items()
    }

    print(
        "Participant-perceived inconsistencies: "
        f"{overall_inconsistencies}"
    )

    print(
        f"Disturbance ratings: "
        f"N={len(disturbance)}, "
        f"mean={disturbance.mean():.2f}, "
        f"median={disturbance.median():.0f}; "
        f"excluded missing/invalid={missing_disturbance}"
    )

    print(
        f"Saved: {output_path}"
    )


if __name__ == "__main__":
    main()