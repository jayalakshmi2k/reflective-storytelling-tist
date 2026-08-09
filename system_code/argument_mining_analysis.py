import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from openai import OpenAI
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


LABELS = ["CLAIM", "PREMISE", "NONE"]

ANNOTATION_INSTRUCTIONS = """You are an argument-mining annotator.

You will receive one complete narrative divided into argument discourse units
(ADUs). Exactly one ADU is marked TARGET. Classify only the TARGET ADU using
exactly one label: CLAIM, PREMISE, or NONE. Use the other ADUs only as context
for determining the argumentative function of the TARGET ADU.

CLAIM:
A recommendation, advice, conclusion, interpretation, judgment, proposal,
position, or main argumentative point.

PREMISE:
A reason, justification, explanation, or evidence offered to support a claim.
It must primarily function as support and not as a recommendation or conclusion.

NONE:
Narrative description, event description, greeting, scene-setting, background
information, personal observation, or emotional encouragement that does not
perform an argumentative function.

Mixed ADU rule:
If one ADU contains both a claim and its supporting reason, assign CLAIM.

Return only one label: CLAIM, PREMISE, or NONE.
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-a", type=Path, required=True)
    parser.add_argument("--gold-b", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/argument_mining_gpt5mini_context"),
    )
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--request-delay", type=float, default=0.2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def load_gold(filename, dataset):
    if not filename.is_file():
        raise FileNotFoundError(f"File not found: {filename}")

    df = pd.read_csv(filename)

    required = ["persona", "story", "segment", "text", "label"]
    missing = [x for x in required if x not in df.columns]
    if missing:
        raise ValueError(f"{filename} is missing columns: {missing}")

    df = df[required].copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.upper()

    if df["text"].eq("").any():
        raise ValueError(f"{filename} contains empty ADU text")

    invalid = sorted(set(df["label"]) - set(LABELS))
    if invalid:
        raise ValueError(f"{filename} contains invalid labels: {invalid}")

    if df[["persona", "story", "segment"]].isna().any().any():
        raise ValueError(f"{filename} contains missing identifiers")

    df.insert(0, "dataset", dataset)
    df.insert(1, "source_row", range(2, len(df) + 2))
    df.insert(2, "adu_id", [f"{dataset}_{i + 1:04d}" for i in range(len(df))])

    return df


def add_story_context(df):
    contexts = []

    for _, row in df.iterrows():
        story = df[
            (df["persona"] == row["persona"])
            & (df["story"] == row["story"])
        ]

        lines = []
        for i, (_, story_row) in enumerate(story.iterrows(), start=1):
            target = " [TARGET]" if story_row["adu_id"] == row["adu_id"] else ""
            lines.append(f"ADU {i}{target}: {story_row['text']}")

        contexts.append("\n".join(lines))

    df = df.copy()
    df["story_context"] = contexts
    return df


def get_label(client, context, model, max_attempts):
    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            response = client.responses.create(
                model=model,
                store=False,
                instructions=ANNOTATION_INSTRUCTIONS,
                input=f"NARRATIVE:\n{context}",
            )

            answer = response.output_text.strip().upper()

            if answer not in LABELS:
                raise ValueError(f"Unexpected model output: {answer}")

            return {
                "llm_pred": answer,
                "requested_model": model,
                "returned_model": response.model,
                "response_id": response.id,
                "retry_count": attempt - 1,
                "classified_at_utc": utc_now(),
            }

        except Exception as e:
            last_error = e
            if attempt < max_attempts:
                time.sleep(min(2 ** (attempt - 1), 30))

    raise RuntimeError(
        f"Classification failed after {max_attempts} attempts: {last_error}"
    )


def load_old_predictions(filename, model):
    if not filename.is_file():
        return {}

    old = pd.read_csv(filename, keep_default_na=False)

    needed = ["adu_id", "llm_pred", "requested_model"]
    if not all(x in old.columns for x in needed):
        return {}

    old = old[
        (old["requested_model"] == model)
        & (old["llm_pred"].isin(LABELS))
    ]

    return {str(row["adu_id"]): row.to_dict() for _, row in old.iterrows()}


def run_dataset(client, df, dataset, output_dir, model,
                max_attempts, request_delay, overwrite):

    prediction_file = output_dir / f"predictions_{dataset}.csv"

    if overwrite:
        old_predictions = {}
    else:
        old_predictions = load_old_predictions(prediction_file, model)

    records = []

    print(f"\nDataset {dataset}: {len(df)} ADUs")

    for number, (_, row) in enumerate(df.iterrows(), start=1):
        adu_id = row["adu_id"]

        if adu_id in old_predictions:
            old = old_predictions[adu_id]
            prediction = {
                "llm_pred": old["llm_pred"],
                "requested_model": old["requested_model"],
                "returned_model": old.get("returned_model", ""),
                "response_id": old.get("response_id", ""),
                "retry_count": old.get("retry_count", 0),
                "classified_at_utc": old.get("classified_at_utc", ""),
            }
        else:
            print(f"[{dataset} {number}/{len(df)}] {adu_id}")
            prediction = get_label(
                client,
                row["story_context"],
                model,
                max_attempts,
            )
            time.sleep(max(request_delay, 0))

        record = {
            "dataset": row["dataset"],
            "persona": row["persona"],
            "story": row["story"],
            "segment": row["segment"],
            "adu_id": row["adu_id"],
            "text": row["text"],
            "label": row["label"],
            **prediction,
        }

        records.append(record)

        if adu_id not in old_predictions:
            pd.DataFrame(records).to_csv(prediction_file, index=False)

    result = pd.DataFrame(records)
    result["is_correct"] = result["label"] == result["llm_pred"]
    result.to_csv(prediction_file, index=False)

    return result


def evaluate(df, dataset, output_dir):
    y_true = df["label"]
    y_pred = df["llm_pred"]

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

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    pd.DataFrame(
        cm,
        index=[f"gold_{x}" for x in LABELS],
        columns=[f"pred_{x}" for x in LABELS],
    ).to_csv(
        output_dir / f"confusion_matrix_{dataset}.csv",
        index_label="gold_label",
    )

    errors = df[df["label"] != df["llm_pred"]].copy()
    errors.to_csv(output_dir / f"errors_{dataset}.csv", index=False)

    p, r, f1, _ = precision_recall_fscore_support(
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
        "macro_precision": p,
        "macro_recall": r,
        "macro_f1": f1,
        "accuracy": accuracy_score(y_true, y_pred),
    }


def main():
    args = get_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set")

    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be at least 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    a = add_story_context(load_gold(args.gold_a, "A"))
    b = add_story_context(load_gold(args.gold_b, "B"))

    client = OpenAI()
    started = utc_now()

    pred_a = run_dataset(
        client, a, "A", args.output_dir, args.model,
        args.max_attempts, args.request_delay, args.overwrite
    )

    pred_b = run_dataset(
        client, b, "B", args.output_dir, args.model,
        args.max_attempts, args.request_delay, args.overwrite
    )

    combined = pd.concat([pred_a, pred_b], ignore_index=True)
    combined.to_csv(
        args.output_dir / "predictions_A_plus_B.csv",
        index=False,
    )

    summaries = [
        evaluate(pred_a, "A", args.output_dir),
        evaluate(pred_b, "B", args.output_dir),
        evaluate(combined, "A_plus_B", args.output_dir),
    ]

    pd.DataFrame(summaries).to_csv(
        args.output_dir / "metrics_summary.csv",
        index=False,
    )

    metadata = {
        "run_started_utc": started,
        "run_completed_utc": utc_now(),
        "requested_model": args.model,
        "returned_models": sorted(
            combined["returned_model"].dropna().unique().tolist()
        ),
        "sampling_configuration": "model default",
        "evaluation_mode":
            "full narrative context; classify marked target ADU only",
        "store": False,
        "gold_a": str(args.gold_a),
        "gold_b": str(args.gold_b),
        "n_a": len(pred_a),
        "n_b": len(pred_b),
        "labels": LABELS,
        "annotation_instructions": ANNOTATION_INSTRUCTIONS,
    }

    with open(args.output_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\nCompleted")
    print(
        pd.DataFrame(summaries).to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )
    print(f"\nOutputs saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
