from pathlib import Path

import pandas as pd


project_root = Path(__file__).resolve().parents[1]
data_dir = project_root / "data" / "raw"
model_columns = ["id", "review", "star"]

train = pd.read_csv(data_dir / "train.csv", usecols=model_columns)
dev = pd.read_csv(data_dir / "dev.csv", usecols=model_columns)
test = pd.read_csv(data_dir / "test.csv", usecols=model_columns)

print("训练集样本数:", len(train))
print("验证集样本数:", len(dev))
print("测试集样本数:", len(test))

train_reviews = set(train["review"])
dev_reviews = set(dev["review"])
test_reviews = set(test["review"])

print("训练集与验证集重复评论:", len(train_reviews & dev_reviews))
print("训练集与测试集重复评论:", len(train_reviews & test_reviews))
print("验证集与测试集重复评论:", len(dev_reviews & test_reviews))
