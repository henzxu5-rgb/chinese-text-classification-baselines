"""逐条比较 TF-IDF 与 TextCNN 在同一验证集上的预测。"""

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

tfidf = pd.read_csv(
    RESULTS_DIR / "dev_predictions.csv",
    usecols=["id", "review", "true_star", "predicted_star"],
).rename(columns={"predicted_star": "tfidf_prediction"})

textcnn = pd.read_csv(
    RESULTS_DIR / "textcnn_dev_predictions.csv",
    usecols=["id", "true_star", "predicted_star"],
).rename(
    columns={
        "true_star": "textcnn_true_star",
        "predicted_star": "textcnn_prediction",
    }
)

# validate="one_to_one" 确保每个 id 在两份文件中都只出现一次。
comparison = tfidf.merge(textcnn, on="id", validate="one_to_one")
if len(comparison) != len(tfidf) or not (
    comparison["true_star"] == comparison["textcnn_true_star"]
).all():
    raise ValueError("两份预测没有对齐到相同的验证集样本和真实星级")

comparison["tfidf_correct"] = (
    comparison["tfidf_prediction"] == comparison["true_star"]
)
comparison["textcnn_correct"] = (
    comparison["textcnn_prediction"] == comparison["true_star"]
)

both_correct = comparison["tfidf_correct"] & comparison["textcnn_correct"]
tfidf_only = comparison["tfidf_correct"] & ~comparison["textcnn_correct"]
textcnn_only = ~comparison["tfidf_correct"] & comparison["textcnn_correct"]
both_wrong = ~comparison["tfidf_correct"] & ~comparison["textcnn_correct"]

model_metrics = {
    "tfidf": {
        "accuracy": accuracy_score(
            comparison["true_star"], comparison["tfidf_prediction"]
        ),
        "macro_f1": f1_score(
            comparison["true_star"],
            comparison["tfidf_prediction"],
            average="macro",
        ),
        "mean_absolute_error": mean_absolute_error(
            comparison["true_star"], comparison["tfidf_prediction"]
        ),
    },
    "textcnn": {
        "accuracy": accuracy_score(
            comparison["true_star"], comparison["textcnn_prediction"]
        ),
        "macro_f1": f1_score(
            comparison["true_star"],
            comparison["textcnn_prediction"],
            average="macro",
        ),
        "mean_absolute_error": mean_absolute_error(
            comparison["true_star"], comparison["textcnn_prediction"]
        ),
    },
}

outcome_counts = {
    "both_correct": int(both_correct.sum()),
    "tfidf_only_correct": int(tfidf_only.sum()),
    "textcnn_only_correct": int(textcnn_only.sum()),
    "both_wrong": int(both_wrong.sum()),
}

per_star = (
    comparison.groupby("true_star")
    .agg(
        examples=("id", "size"),
        tfidf_correct=("tfidf_correct", "sum"),
        textcnn_correct=("textcnn_correct", "sum"),
    )
    .reset_index()
)

print("模型整体指标:")
print(pd.DataFrame(model_metrics).T.round(4))
print("\n逐条预测关系:")
print(pd.Series(outcome_counts))
print("\n各真实星级答对数量:")
print(per_star.to_string(index=False))

# 分别抽取两种模型独自答对的样本，供人工检查它们是否真的学到不同模式。
sample = pd.concat(
    [
        comparison.loc[tfidf_only].sample(n=8, random_state=42).assign(
            only_correct_model="tfidf"
        ),
        comparison.loc[textcnn_only].sample(n=8, random_state=42).assign(
            only_correct_model="textcnn"
        ),
    ],
    ignore_index=True,
)
sample.to_csv(
    RESULTS_DIR / "model_disagreement_sample.csv",
    index=False,
    encoding="utf-8-sig",
)

summary = {
    "model_metrics": model_metrics,
    "outcome_counts": outcome_counts,
    "per_star": per_star.to_dict(orient="records"),
}
(RESULTS_DIR / "model_comparison_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
)
print("\n已保存 results/model_disagreement_sample.csv")
print("已保存 results/model_comparison_summary.json")
