import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


LABELS = ["CLAIM", "PREMISE", "NONE"]


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pred-a",
        type=Path,
        default=Path("argument_mining/predictions_A.csv"),
    )

    parser.add_argument(
        "--pred-b",
        type=Path,
        default=Path("argument_mining/predictions_B.csv"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/argument_mining_evaluation"),
    )

    return parser.parse_args()


def load_predictions(filename, dataset):
    if not filename.is_file():
        raise FileNotFoundError(f"File not found: {filename}")

    df = pd.read_csv(filename)

    required = [
        "persona",
        "story",
        "segment",
        "text",
        "gold_label",
        "predicted_label",
    ]

    missing = [x for x in required if x not in df.columns]

    if missing:
        raise ValueError(f"{filename} missing columns: {missing}")

    df = df.copy()

    df["gold_label"] = (
        df["gold_label"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["predicted_label"] = (
        df["predicted_label"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    invalid_gold = sorted(
        set(df["gold_label"]) - set(LABELS)
    )

    invalid_pred = sorted(
        set(df["predicted_label"]) - set(LABELS)
    )

    if invalid_gold:
        raise ValueError(
            f"{filename} contains invalid gold labels: {invalid_gold}"
        )

    if invalid_pred:
        raise ValueError(
            f"{filename} contains invalid predictions: {invalid_pred}"
        )

    return df




def evaluate(df, dataset, output_dir):
    y_true = df["gold_label"]
    y_pred = df["predicted_label"]

    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    rows = []

    for label in LABELS:
        rows.append({
            "dataset": dataset,
            "class": label,
            "precision": report[label]["precision"],
            "recall": report[label]["recall"],
            "f1": report[label]["f1-score"],
            "support": int(report[label]["support"]),
        })

    pd.DataFrame(rows).to_csv(
        output_dir / f"metrics_per_class_{dataset}.csv",
        index=False,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=LABELS,
    )

    cm_df = pd.DataFrame(
        cm,
        index=[f"gold_{x}" for x in LABELS],
        columns=[f"pred_{x}" for x in LABELS],
    )

    cm_df.to_csv(
        output_dir / f"confusion_matrix_{dataset}.csv",
        index_label="gold_label",
    )

    errors = df[
        df["gold_label"] != df["predicted_label"]
    ].copy()

    errors.to_csv(
        output_dir / f"errors_{dataset}.csv",
        index=False,
    )

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    return {
        "dataset": dataset,
        "n_adus": len(df),
        "n_errors": len(errors),
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "accuracy": accuracy_score(y_true, y_pred),
    }


def main():
    args = get_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    data_a = load_predictions(
        args.pred_a,
        "A",
    )

    data_b = load_predictions(
        args.pred_b,
        "B",
    )

    print("\nInput validation passed")

    print(
        "Dataset A:",
        len(data_a),
        data_a["gold_label"].value_counts().to_dict(),
    )

    print(
        "Dataset B:",
        len(data_b),
        data_b["gold_label"].value_counts().to_dict(),
    )

    result_a = evaluate(
        data_a,
        "A",
        args.output_dir,
    )

    result_b = evaluate(
        data_b,
        "B",
        args.output_dir,
    )

    summary = pd.DataFrame([
        result_a,
        result_b,
    ])

    summary.to_csv(
        args.output_dir / "metrics_summary.csv",
        index=False,
    )

    print("\nReproduced results")

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print("\nOutputs saved in:")
    print(args.output_dir)


if __name__ == "__main__":
    main()