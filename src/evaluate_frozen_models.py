"""用冻结配置统一评价 TF-IDF 与 TextCNN；测试集只在配置确定后运行一次。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader, TensorDataset

from train_textcnn import Config, TextCNN, build_vocabulary, encode_texts


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"
STAR_LABELS = [1, 2, 3, 4, 5]


def read_split(filename: str) -> pd.DataFrame:
    frame = pd.read_csv(
        DATA_DIR / filename, usecols=["id", "review", "star"]
    )
    if frame[["review", "star"]].isna().any().any():
        raise ValueError(f"{filename} 中存在空评论或空星级")
    return frame


def make_tfidf_model() -> Pipeline:
    """返回已经在验证集上选定、此后不再调整的传统基线。"""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(2, 2),
                    min_df=3,
                    max_features=50_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(C=2.0, max_iter=400, random_state=42),
            ),
        ]
    )


def predict_with_textcnn(train: pd.DataFrame, evaluation: pd.DataFrame) -> np.ndarray:
    """重建训练词表并加载验证集 Macro-F1 最佳的 TextCNN 参数。"""
    config = Config()
    character_to_id = build_vocabulary(train["review"], config)
    evaluation_ids = encode_texts(
        evaluation["review"], character_to_id, config.max_length
    )

    model_path = RESULTS_DIR / "textcnn_full_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            "缺少 results/textcnn_full_model.pt，请先完成正式 TextCNN 训练"
        )
    model = TextCNN(len(character_to_id), config)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    batches = DataLoader(
        TensorDataset(evaluation_ids), batch_size=config.batch_size, shuffle=False
    )
    predictions: list[np.ndarray] = []
    with torch.inference_mode():
        for (character_ids,) in batches:
            predictions.append(model(character_ids).argmax(dim=1).numpy() + 1)
    return np.concatenate(predictions)


def compute_metrics(true_stars: pd.Series, predicted_stars: np.ndarray) -> dict:
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        true_stars,
        predicted_stars,
        labels=STAR_LABELS,
        average="macro",
        zero_division=0,
    )
    per_class_precision, per_class_recall, per_class_f1, support = (
        precision_recall_fscore_support(
            true_stars,
            predicted_stars,
            labels=STAR_LABELS,
            average=None,
            zero_division=0,
        )
    )
    return {
        "accuracy": float(accuracy_score(true_stars, predicted_stars)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "mean_absolute_error": float(
            mean_absolute_error(true_stars, predicted_stars)
        ),
        "confusion_matrix": confusion_matrix(
            true_stars, predicted_stars, labels=STAR_LABELS
        ).tolist(),
        "per_class": {
            str(star): {
                "precision": float(per_class_precision[index]),
                "recall": float(per_class_recall[index]),
                "f1": float(per_class_f1[index]),
                "support": int(support[index]),
            }
            for index, star in enumerate(STAR_LABELS)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        required=True,
        choices=["dev", "test"],
        help="先用 dev 检查脚本；配置冻结后只运行一次 test。",
    )
    args = parser.parse_args()

    train = read_split("train.csv")
    evaluation = read_split(f"{args.split}.csv")

    tfidf_model = make_tfidf_model()
    tfidf_started = time.perf_counter()
    tfidf_model.fit(train["review"], train["star"])
    tfidf_predictions = tfidf_model.predict(evaluation["review"])
    tfidf_seconds = time.perf_counter() - tfidf_started

    textcnn_started = time.perf_counter()
    textcnn_predictions = predict_with_textcnn(train, evaluation)
    textcnn_seconds = time.perf_counter() - textcnn_started

    metrics = {
        "evaluation_split": args.split,
        "protocol": {
            "fit_split": "train",
            "model_selection_split": "dev",
            "test_used_for_model_selection": False,
            "random_seed": 42,
        },
        "rows": {"train": len(train), args.split: len(evaluation)},
        "tfidf": compute_metrics(evaluation["star"], tfidf_predictions),
        "textcnn": compute_metrics(evaluation["star"], textcnn_predictions),
        "runtime_seconds_this_command": {
            "tfidf_fit_and_predict": tfidf_seconds,
            "textcnn_load_and_predict": textcnn_seconds,
        },
    }

    summary = pd.DataFrame(
        {
            model_name: {
                key: model_metrics[key]
                for key in [
                    "accuracy",
                    "macro_precision",
                    "macro_recall",
                    "macro_f1",
                    "mean_absolute_error",
                ]
            }
            for model_name, model_metrics in [
                ("tfidf", metrics["tfidf"]),
                ("textcnn", metrics["textcnn"]),
            ]
        }
    ).T

    RESULTS_DIR.mkdir(exist_ok=True)
    metrics_path = RESULTS_DIR / f"frozen_{args.split}_metrics.json"
    predictions_path = RESULTS_DIR / f"frozen_{args.split}_predictions.csv"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        {
            "id": evaluation["id"],
            "review": evaluation["review"],
            "true_star": evaluation["star"].astype(int),
            "tfidf_prediction": tfidf_predictions.astype(int),
            "textcnn_prediction": textcnn_predictions.astype(int),
        }
    ).to_csv(predictions_path, index=False, encoding="utf-8-sig")

    print(f"冻结模型统一评价：{args.split}")
    print(summary.round(4))
    print(f"已保存 {metrics_path.relative_to(PROJECT_ROOT)}")
    print(f"已保存 {predictions_path.relative_to(PROJECT_ROOT)}")
    if args.split == "test":
        print("测试集结果只用于最终报告；不得再依据这些结果修改配置。")


if __name__ == "__main__":
    main()
