from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parents[1]
data_path = project_root / "data" / "raw" / "train.csv"

df = pd.read_csv(data_path)

# 官方 sample 使用 index 和 reviewbody；统一成后续模型使用的名称。
df = df.rename(columns={"index": "id", "reviewbody": "review"})
model_columns = ["id", "review", "star"]
model_data = df[model_columns].copy()
review_lengths = model_data["review"].astype(str).str.len()

print("样本数:", len(model_data))
print("模型相关字段:", model_columns)
print("各字段空值数:")
print(model_data.isna().sum())
print("重复评论数:", model_data["review"].duplicated().sum())
print("各星级样本数:")
print(model_data["star"].value_counts().sort_index())
print("各星级比例 (%):")
print((model_data["star"].value_counts(normalize=True).sort_index() * 100).round(1))
print("评论长度（字符）:")
print(
    {
        "最短": int(review_lengths.min()),
        "中位数": float(review_lengths.median()),
        "95%分位数": float(review_lengths.quantile(0.95)),
        "最长": int(review_lengths.max()),
    }
)
