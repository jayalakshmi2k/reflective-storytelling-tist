from pathlib import Path
import argparse
import math

import pandas as pd
from scipy.stats import fisher_exact


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input",
        help="CSV file containing the human-reviewed grounding sample"
    )

    parser.add_argument(
        "--output-dir",
        default="results/grounding",
        help="Folder where result files will be saved"
    )

    return parser.parse_args()


def wilson_interval(successes, total):
    if total == 0:
        return 0.0, 0.0

    z = 1.959963984540054

    p = successes / total

    denominator = 1 + z * z / total

    centre = (
        p + z * z / (2 * total)
    ) / denominator

    margin = (
        z
        * math.sqrt(
            (
                p * (1 - p)
                + z * z / (4 * total)
            ) / total
        )
        / denominator
    )

    low = max(0.0, centre - margin)
    high = min(1.0, centre + margin)

    return low, high


def clean_data(df):
    required_columns = [
        "Review ID",
        "Unit ID",
        "Condition",
        "Persona",
        "Story",
        "Atomic text",
        "Human label"
    ]

    missing = []

    for column in required_columns:
        if column not in df.columns:
            missing.append(column)

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    df = df.copy()

    df["Condition"] = (
        df["Condition"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["Human label"] = (
        df["Human label"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(" ", "_")
    )

    accepted_labels = [
        "GROUNDED",
        "UNSUPPORTED",
        "CONTRADICTED",
        "NOT_APPLICABLE",
        "UNCLEAR"
    ]

    unexpected = sorted(
        set(df["Human label"])
        - set(accepted_labels)
    )

    if unexpected:
        raise ValueError(
            "Unexpected Human label values: "
            + str(unexpected)
        )

    unexpected_conditions = sorted(
        set(df["Condition"])
        - {"A", "B", "C"}
    )

    if unexpected_conditions:
        raise ValueError(
            "Unexpected conditions: "
            + str(unexpected_conditions)
        )

    return df


def validate_sample(df):
    print("\nInput validation")

    print("Total statements:", len(df))

    condition_counts = (
        df["Condition"]
        .value_counts()
        .sort_index()
    )

    print("\nStatements per condition")
    print(condition_counts.to_string())

    if len(df) != 120:
        print(
            "\nWARNING: Expected 120 statements, "
            "but found",
            len(df)
        )

    for condition in ["A", "B", "C"]:
        count = (
            df["Condition"] == condition
        ).sum()

        if count != 40:
            print(
                "WARNING:",
                condition,
                "contains",
                count,
                "statements instead of 40"
            )


def make_summary(df):
    rows = []

    for condition in ["A", "B", "C"]:
        x = df[
            df["Condition"] == condition
        ]

        grounded = (
            x["Human label"] == "GROUNDED"
        ).sum()

        unsupported = (
            x["Human label"] == "UNSUPPORTED"
        ).sum()

        contradicted = (
            x["Human label"] == "CONTRADICTED"
        ).sum()

        not_applicable = (
            x["Human label"] == "NOT_APPLICABLE"
        ).sum()

        unclear = (
            x["Human label"] == "UNCLEAR"
        ).sum()

        applicable = (
            grounded
            + unsupported
            + contradicted
        )

        if applicable > 0:
            grounded_rate = grounded / applicable
            unsupported_rate = unsupported / applicable
            contradicted_rate = contradicted / applicable
        else:
            grounded_rate = 0.0
            unsupported_rate = 0.0
            contradicted_rate = 0.0

        ci_low, ci_high = wilson_interval(
            grounded,
            applicable
        )

        rows.append({
            "condition": condition,
            "sampled": len(x),
            "applicable": applicable,
            "grounded": grounded,
            "unsupported": unsupported,
            "contradicted": contradicted,
            "not_applicable": not_applicable,
            "unclear": unclear,
            "grounded_rate": grounded_rate,
            "grounded_percent": grounded_rate * 100,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "ci_low_percent": ci_low * 100,
            "ci_high_percent": ci_high * 100,
            "unsupported_rate": unsupported_rate,
            "contradicted_rate": contradicted_rate
        })

    return pd.DataFrame(rows)


def make_label_counts(df):
    labels = [
        "GROUNDED",
        "UNSUPPORTED",
        "CONTRADICTED",
        "NOT_APPLICABLE",
        "UNCLEAR"
    ]

    table = pd.crosstab(
        df["Condition"],
        df["Human label"]
    )

    for label in labels:
        if label not in table.columns:
            table[label] = 0

    table = table[labels]

    table = table.reset_index()

    return table


def fisher_comparison(df, condition1, condition2):
    x = df[
        (df["Condition"] == condition1)
        & df["Human label"].isin(
            [
                "GROUNDED",
                "UNSUPPORTED",
                "CONTRADICTED"
            ]
        )
    ]

    y = df[
        (df["Condition"] == condition2)
        & df["Human label"].isin(
            [
                "GROUNDED",
                "UNSUPPORTED",
                "CONTRADICTED"
            ]
        )
    ]

    grounded1 = (
        x["Human label"] == "GROUNDED"
    ).sum()

    non_grounded1 = len(x) - grounded1

    grounded2 = (
        y["Human label"] == "GROUNDED"
    ).sum()

    non_grounded2 = len(y) - grounded2

    table = [
        [grounded1, non_grounded1],
        [grounded2, non_grounded2]
    ]

    odds_ratio, p_value = fisher_exact(
        table,
        alternative="two-sided"
    )

    rate1 = (
        grounded1 / len(x)
        if len(x)
        else 0
    )

    rate2 = (
        grounded2 / len(y)
        if len(y)
        else 0
    )

    return {
        "comparison":
            condition1 + " vs " + condition2,
        "applicable_" + condition1:
            len(x),
        "grounded_" + condition1:
            grounded1,
        "rate_" + condition1:
            rate1,
        "applicable_" + condition2:
            len(y),
        "grounded_" + condition2:
            grounded2,
        "rate_" + condition2:
            rate2,
        "percentage_point_difference":
            (rate1 - rate2) * 100,
        "odds_ratio":
            odds_ratio,
        "p_value":
            p_value
    }


def make_comparisons(df):
    comparisons = []

    comparisons.append(
        fisher_comparison(
            df,
            "A",
            "B"
        )
    )

    comparisons.append(
        fisher_comparison(
            df,
            "A",
            "C"
        )
    )

    comparisons.append(
        fisher_comparison(
            df,
            "B",
            "C"
        )
    )

    return pd.DataFrame(comparisons)


def print_summary(summary):
    display = summary[
        [
            "condition",
            "sampled",
            "applicable",
            "grounded",
            "unsupported",
            "contradicted",
            "not_applicable",
            "unclear",
            "grounded_percent",
            "ci_low_percent",
            "ci_high_percent"
        ]
    ].copy()

    display[
        "grounded_percent"
    ] = display[
        "grounded_percent"
    ].round(1)

    display[
        "ci_low_percent"
    ] = display[
        "ci_low_percent"
    ].round(1)

    display[
        "ci_high_percent"
    ] = display[
        "ci_high_percent"
    ].round(1)

    print("\nGrounding summary")
    print(
        display.to_string(index=False)
    )


def main():
    args = get_args()

    input_file = Path(args.input)

    if not input_file.exists():
        raise FileNotFoundError(
            "Input file not found: "
            + str(input_file)
        )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    df = pd.read_csv(
        input_file
    )

    df = clean_data(df)

    validate_sample(df)

    summary = make_summary(df)

    label_counts = make_label_counts(
        df
    )

    comparisons = make_comparisons(
        df
    )

    print_summary(summary)

    print("\nLabel counts")
    print(
        label_counts.to_string(
            index=False
        )
    )

    comparison_display = (
        comparisons.copy()
    )

    for column in comparison_display.columns:
        if column.startswith("rate_"):
            comparison_display[column] = (
                comparison_display[column]
                * 100
            ).round(1)

    comparison_display[
        "percentage_point_difference"
    ] = comparison_display[
        "percentage_point_difference"
    ].round(1)

    comparison_display[
        "p_value"
    ] = comparison_display[
        "p_value"
    ].round(6)

    print(
        "\nPairwise Fisher exact tests "
        "(applicable statements only)"
    )

    print(
        comparison_display.to_string(
            index=False
        )
    )

    summary.to_csv(
        output_dir
        / "grounding_summary.csv",
        index=False
    )

    label_counts.to_csv(
        output_dir
        / "grounding_label_counts.csv",
        index=False
    )

    comparisons.to_csv(
        output_dir
        / "grounding_comparisons.csv",
        index=False
    )

    print(
        "\nOutputs saved in:",
        output_dir
    )


if __name__ == "__main__":
    main()