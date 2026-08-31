"""比较 TextCNN 与预训练 Transformer 在同一验证集上的预测。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_recall_fscore_support,
)
from transformers import AutoTokenizer

from inspect_pretrained_tokenizer import MODEL_NAME, encode_with_head_tail


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
MAX_LENGTH = 128
SAMPLES_PER_GROUP = 6


def metrics(true_stars: pd.Series, predicted_stars: pd.Series) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(true_stars, predicted_stars)),
        "macro_f1": float(
            f1_score(true_stars, predicted_stars, average="macro", zero_division=0)
        ),
        "mean_absolute_error": float(mean_absolute_error(true_stars, predicted_stars)),
    }


def per_star_metrics(
    true_stars: pd.Series, predicted_stars: pd.Series, prefix: str
) -> pd.DataFrame:
    precision, recall, f1, support = precision_recall_fscore_support(
        true_stars,
        predicted_stars,
        labels=[1, 2, 3, 4, 5],
        zero_division=0,
    )
    return pd.DataFrame(
        {
            "true_star": [1, 2, 3, 4, 5],
            "examples": support,
            f"{prefix}_precision": precision,
            f"{prefix}_recall": recall,
            f"{prefix}_f1": f1,
        }
    )


def transformer_input_tokens(tokenizer, review: str) -> str:
    """返回该评论实际送入 Transformer 的非 padding token。"""
    encoded = encode_with_head_tail(tokenizer, review, MAX_LENGTH)
    kept_count = sum(encoded["attention_mask"])
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][:kept_count])
    return " ".join(tokens)


def take_sample(frame: pd.DataFrame, model_name: str, random_state: int) -> pd.DataFrame:
    return (
        frame.sample(n=min(SAMPLES_PER_GROUP, len(frame)), random_state=random_state)
        .assign(only_correct_model=model_name)
        .copy()
    )


def main() -> None:
    textcnn = pd.read_csv(
        RESULTS_DIR / "textcnn_dev_predictions.csv",
        usecols=["id", "review", "true_star", "predicted_star"],
    ).rename(columns={"predicted_star": "textcnn_prediction"})
    pretrained = pd.read_csv(
        RESULTS_DIR / "pretrained_mini_full_dev_predictions.csv",
        usecols=["id", "true_star", "predicted_star"],
    ).rename(
        columns={
            "true_star": "pretrained_true_star",
            "predicted_star": "pretrained_prediction",
        }
    )

    comparison = textcnn.merge(pretrained, on="id", validate="one_to_one")
    if len(comparison) != len(textcnn) or not (
        comparison["true_star"] == comparison["pretrained_true_star"]
    ).all():
        raise ValueError("两份预测没有对齐到同一批验证样本和真实星级")

    comparison["textcnn_correct"] = (
        comparison["textcnn_prediction"] == comparison["true_star"]
    )
    comparison["pretrained_correct"] = (
        comparison["pretrained_prediction"] == comparison["true_star"]
    )

    textcnn_only = comparison["textcnn_correct"] & ~comparison["pretrained_correct"]
    pretrained_only = ~comparison["textcnn_correct"] & comparison["pretrained_correct"]
    both_correct = comparison["textcnn_correct"] & comparison["pretrained_correct"]
    both_wrong = ~comparison["textcnn_correct"] & ~comparison["pretrained_correct"]

    model_metrics = {
        "textcnn": metrics(comparison["true_star"], comparison["textcnn_prediction"]),
        "pretrained": metrics(
            comparison["true_star"], comparison["pretrained_prediction"]
        ),
    }
    outcome_counts = {
        "both_correct": int(both_correct.sum()),
        "textcnn_only_correct": int(textcnn_only.sum()),
        "pretrained_only_correct": int(pretrained_only.sum()),
        "both_wrong": int(both_wrong.sum()),
    }
    textcnn_per_star = per_star_metrics(
        comparison["true_star"], comparison["textcnn_prediction"], "textcnn"
    )
    pretrained_per_star = per_star_metrics(
        comparison["true_star"], comparison["pretrained_prediction"], "pretrained"
    ).drop(columns="examples")
    per_star = textcnn_per_star.merge(
        pretrained_per_star, on="true_star", validate="one_to_one"
    )

    sample = pd.concat(
        [
            take_sample(comparison.loc[pretrained_only], "pretrained", 42),
            take_sample(comparison.loc[textcnn_only], "textcnn", 43),
        ],
        ignore_index=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    sample["transformer_input_tokens"] = sample["review"].map(
        lambda review: transformer_input_tokens(tokenizer, str(review))
    )
    sample_path = RESULTS_DIR / "pretrained_textcnn_disagreement_sample.csv"
    sample.to_csv(sample_path, index=False, encoding="utf-8-sig")

    summary = {
        "comparison_split": "dev.csv only",
        "model_metrics": model_metrics,
        "outcome_counts": outcome_counts,
        "per_star": per_star.to_dict(orient="records"),
        "sample_file": str(sample_path.relative_to(PROJECT_ROOT)),
        "note": "The sample records the Transformer's retained input tokens, not a claim about semantic understanding.",
    }
    summary_path = RESULTS_DIR / "pretrained_textcnn_dev_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("整体指标：")
    print(pd.DataFrame(model_metrics).T.round(4))
    print("\n逐条预测关系：")
    print(pd.Series(outcome_counts))
    print("\n各真实星级 Precision / Recall / F1：")
    print(per_star.to_string(index=False))
    print(f"\n已保存：{sample_path.relative_to(PROJECT_ROOT)}")
    print(f"已保存：{summary_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
