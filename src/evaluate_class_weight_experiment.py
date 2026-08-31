"""对已冻结的类别加权实验模型做一次测试集评价，不重新训练或调参。"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from evaluate_frozen_models import compute_metrics, read_split
from train_textcnn import Config, TextCNN, build_vocabulary, encode_texts


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
SUMMARY_METRICS = [
    "accuracy",
    "macro_f1",
    "mean_absolute_error",
    "star_2_precision",
    "star_2_recall",
    "star_2_f1",
    "star_3_precision",
    "star_3_recall",
    "star_3_f1",
]


def config_from_saved_metrics(saved_config: dict) -> Config:
    """把 JSON 中的列表恢复为 Config 需要的卷积核元组。"""
    config_values = dict(saved_config)
    config_values["kernel_sizes"] = tuple(config_values["kernel_sizes"])
    return Config(**config_values)


def predict(
    model_path: Path, config: Config, vocabulary: dict[str, int], texts: pd.Series
) -> np.ndarray:
    encoded = encode_texts(texts, vocabulary, config.max_length)
    model = TextCNN(len(vocabulary), config)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    loader = DataLoader(
        TensorDataset(encoded), batch_size=config.batch_size, shuffle=False
    )
    predictions: list[np.ndarray] = []
    with torch.inference_mode():
        for (character_ids,) in loader:
            predictions.append(model(character_ids).argmax(dim=1).numpy() + 1)
    return np.concatenate(predictions)


def collect_test_runs(train: pd.DataFrame, test: pd.DataFrame) -> list[dict]:
    runs: list[dict] = []
    for metrics_path in sorted(
        RESULTS_DIR.glob("textcnn_class_weight_*_metrics.json")
    ):
        dev_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if dev_metrics.get("mode") != "full":
            continue

        model_path = metrics_path.with_name(
            metrics_path.name.replace("_metrics.json", "_model.pt")
        )
        if not model_path.exists():
            raise FileNotFoundError(f"缺少已训练模型：{model_path.name}")

        config = config_from_saved_metrics(dev_metrics["config"])
        vocabulary = build_vocabulary(train["review"], config)
        predictions = predict(model_path, config, vocabulary, test["review"])
        test_metrics = compute_metrics(test["star"], predictions)
        per_class = test_metrics["per_class"]
        runs.append(
            {
                "file": metrics_path.name,
                "condition": dev_metrics["class_weighting"],
                "seed": config.random_seed,
                "best_dev_epoch": dev_metrics["best_epoch"],
                "accuracy": test_metrics["accuracy"],
                "macro_f1": test_metrics["macro_f1"],
                "mean_absolute_error": test_metrics["mean_absolute_error"],
                "star_2_precision": per_class["2"]["precision"],
                "star_2_recall": per_class["2"]["recall"],
                "star_2_f1": per_class["2"]["f1"],
                "star_3_precision": per_class["3"]["precision"],
                "star_3_recall": per_class["3"]["recall"],
                "star_3_f1": per_class["3"]["f1"],
            }
        )
    return runs


def summarize(runs: list[dict]) -> list[dict]:
    summary: list[dict] = []
    for condition in ["none", "balanced"]:
        condition_runs = [run for run in runs if run["condition"] == condition]
        if not condition_runs:
            raise ValueError(f"缺少 {condition} 条件的结果")
        row: dict[str, float | int | str] = {
            "condition": condition,
            "runs": len(condition_runs),
        }
        for metric_name in SUMMARY_METRICS:
            values = [float(run[metric_name]) for run in condition_runs]
            row[f"{metric_name}_mean"] = float(statistics.mean(values))
            row[f"{metric_name}_std"] = float(
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        summary.append(row)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="对已冻结的类别加权实验模型进行一次测试集评价。"
    )
    parser.add_argument(
        "--confirm-test-evaluation",
        action="store_true",
        help="明确确认本次命令会读取测试集并写入最终比较结果。",
    )
    args = parser.parse_args()
    if not args.confirm_test_evaluation:
        parser.error("请显式加入 --confirm-test-evaluation 后再读取测试集")

    train = read_split("train.csv")
    test = read_split("test.csv")
    runs = collect_test_runs(train, test)
    if len(runs) != 6:
        raise ValueError(
            f"预期 6 个完整模型（3 个种子 × 2 个条件），实际找到 {len(runs)} 个"
        )
    summary = summarize(runs)

    run_table = pd.DataFrame(runs).sort_values(["condition", "seed"])
    summary_table = pd.DataFrame(summary).sort_values("condition")
    print("测试集逐次结果：")
    print(run_table.drop(columns=["file"]).round(4).to_string(index=False))
    print("\n测试集按条件汇总（mean ± sample std）：")
    display_columns = [
        "condition",
        "runs",
        "accuracy_mean",
        "accuracy_std",
        "macro_f1_mean",
        "macro_f1_std",
        "mean_absolute_error_mean",
        "mean_absolute_error_std",
        "star_2_recall_mean",
        "star_2_recall_std",
        "star_3_recall_mean",
        "star_3_recall_std",
    ]
    print(summary_table[display_columns].round(4).to_string(index=False))

    output_path = RESULTS_DIR / "class_weight_test_metrics.json"
    output_path.write_text(
        json.dumps(
            {
                "protocol": {
                    "fit_split": "train",
                    "selection_split": "dev",
                    "evaluation_split": "test",
                    "conditions": ["none", "balanced"],
                    "seeds": [42, 43, 44],
                    "test_result_used_to_change_config": False,
                    "note": (
                        "第一阶段已查看过原始模型的测试集结果；本文件是验证集"
                        "协议冻结后的预先规定扩展评价。"
                    ),
                },
                "runs": runs,
                "summary": summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nsaved: {output_path.relative_to(PROJECT_ROOT)}")
    print("测试集输出只用于记录最终比较，不得据此继续修改类别权重或训练设置。")


if __name__ == "__main__":
    main()
