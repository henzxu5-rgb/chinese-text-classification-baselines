from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline


project_root = Path(__file__).resolve().parents[1]
data_dir = project_root / "data" / "raw"
model_columns = ["id", "review", "star"]

train = pd.read_csv(data_dir / "train.csv", usecols=model_columns)
dev = pd.read_csv(data_dir / "dev.csv", usecols=model_columns)

# 对照：不读取评论内容，永远预测训练集中数量最多的星级。
majority_star = int(train["star"].mode().iloc[0])
majority_predictions = np.full(len(dev), majority_star)
print("多数类基线预测星级:", majority_star)
print("多数类 Accuracy:", round(accuracy_score(dev["star"], majority_predictions), 4))
print("多数类 Macro-F1:", round(f1_score(dev["star"], majority_predictions, average="macro"), 4))

model = Pipeline(
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

started = time.perf_counter()
model.fit(train["review"], train["star"])
dev_predictions = model.predict(dev["review"])
elapsed_seconds = time.perf_counter() - started

print("TF-IDF + 逻辑回归 Accuracy:", round(accuracy_score(dev["star"], dev_predictions), 4))
print("TF-IDF + 逻辑回归 Macro-F1:", round(f1_score(dev["star"], dev_predictions, average="macro"), 4))
print("训练与预测耗时（秒）:", round(elapsed_seconds, 1))

labels = [1, 2, 3, 4, 5]
confusion = pd.DataFrame(
    confusion_matrix(dev["star"], dev_predictions, labels=labels),
    index=[f"真实{star}星" for star in labels],
    columns=[f"预测{star}星" for star in labels],
)
print("\n混淆矩阵（行是真实星级，列是预测星级）:")
print(confusion)

print("\n逐星级 Precision / Recall / F1:")
print(
    classification_report(
        dev["star"],
        dev_predictions,
        labels=labels,
        digits=3,
        zero_division=0,
    )
)

results_dir = project_root / "results"
results_dir.mkdir(exist_ok=True)
prediction_table = pd.DataFrame(
    {
        "id": dev["id"],
        "review": dev["review"],
        "true_star": dev["star"].astype(int),
        "predicted_star": dev_predictions,
    }
)
prediction_table["is_correct"] = (
    prediction_table["true_star"] == prediction_table["predicted_star"]
)
prediction_table["error_distance"] = (
    prediction_table["true_star"] - prediction_table["predicted_star"]
).abs()
prediction_table.to_csv(
    results_dir / "dev_predictions.csv", index=False, encoding="utf-8-sig"
)
print("验证集逐条预测已保存到 results/dev_predictions.csv")
