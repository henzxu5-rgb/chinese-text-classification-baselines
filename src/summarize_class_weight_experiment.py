"""汇总类别加权 TextCNN 的多随机种子验证集结果。"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
METRIC_NAMES = [
    "best_dev_accuracy",
    "best_dev_macro_f1",
    "best_dev_mae",
    "star_2_precision",
    "star_2_recall",
    "star_2_f1",
    "star_3_precision",
    "star_3_recall",
    "star_3_f1",
]


def load_runs() -> list[dict[str, float | int | str]]:
    runs: list[dict[str, float | int | str]] = []
    for path in sorted(RESULTS_DIR.glob("textcnn_class_weight_*_metrics.json")):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        condition = str(metrics["class_weighting"])
        if metrics.get("mode") != "full" or condition not in {"none", "balanced"}:
            continue
        per_class = metrics["per_class"]
        runs.append(
            {
                "file": path.name,
                "condition": condition,
                "seed": int(metrics["config"]["random_seed"]),
                "best_epoch": int(metrics["best_epoch"]),
                "best_dev_accuracy": float(metrics["best_dev_accuracy"]),
                "best_dev_macro_f1": float(metrics["best_dev_macro_f1"]),
                "best_dev_mae": float(metrics["best_dev_mae"]),
                "star_2_precision": float(per_class["2"]["precision"]),
                "star_2_recall": float(per_class["2"]["recall"]),
                "star_2_f1": float(per_class["2"]["f1"]),
                "star_3_precision": float(per_class["3"]["precision"]),
                "star_3_recall": float(per_class["3"]["recall"]),
                "star_3_f1": float(per_class["3"]["f1"]),
            }
        )
    return runs


def main() -> None:
    runs = load_runs()
    if not runs:
        raise FileNotFoundError(
            "没有找到 textcnn_class_weight_*_metrics.json；请先运行对照实验。"
        )

    run_table = pd.DataFrame(runs).sort_values(["condition", "seed"])
    print("逐次运行结果：")
    print(run_table.drop(columns=["file"]).round(4).to_string(index=False))

    rows: list[dict[str, float | int | str]] = []
    for condition, group in run_table.groupby("condition", sort=True):
        row: dict[str, float | int | str] = {"condition": condition, "runs": len(group)}
        for metric_name in METRIC_NAMES:
            values = group[metric_name].tolist()
            row[f"{metric_name}_mean"] = float(statistics.mean(values))
            row[f"{metric_name}_std"] = float(
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("condition")
    print("\n按条件汇总（mean ± sample std）：")
    display_columns = [
        "condition",
        "runs",
        "best_dev_accuracy_mean",
        "best_dev_accuracy_std",
        "best_dev_macro_f1_mean",
        "best_dev_macro_f1_std",
        "best_dev_mae_mean",
        "best_dev_mae_std",
        "star_2_recall_mean",
        "star_2_recall_std",
        "star_3_recall_mean",
        "star_3_recall_std",
    ]
    print(summary[display_columns].round(4).to_string(index=False))

    output_path = RESULTS_DIR / "class_weight_experiment_summary.json"
    output_path.write_text(
        json.dumps(
            {"runs": runs, "summary": rows}, ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )
    print(f"\nsaved: {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
