"""对验证样本 5124 做文本扰动，检查显式负面结尾对两模型的影响。"""

from pathlib import Path

import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from train_textcnn import Config, TextCNN, build_vocabulary, encode_texts


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"

train = pd.read_csv(DATA_DIR / "train.csv", usecols=["review", "star"])
dev = pd.read_csv(DATA_DIR / "dev.csv", usecols=["id", "review", "star"])
case = dev.loc[dev["id"] == 5124].iloc[0]
explicit_ending = "差评！负分！"

variants = pd.DataFrame(
    {
        "variant": ["原文", "删除显式结尾", "只保留显式结尾"],
        "review": [
            case["review"],
            case["review"].replace(explicit_ending, ""),
            explicit_ending,
        ],
    }
)

# 使用与正式传统基线完全相同的设置，只在训练集上拟合。
tfidf_model = Pipeline(
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
tfidf_model.fit(train["review"], train["star"])
tfidf_probabilities = tfidf_model.predict_proba(variants["review"])

# 训练词表的构造是确定性的，因此可以重建词表并加载用户已训练的模型参数。
config = Config()
character_to_id = build_vocabulary(train["review"], config)
textcnn = TextCNN(len(character_to_id), config)
model_path = RESULTS_DIR / "textcnn_full_model.pt"
if not model_path.exists():
    raise FileNotFoundError("请先运行完整 TextCNN 训练，生成 textcnn_full_model.pt")
textcnn.load_state_dict(torch.load(model_path, map_location="cpu"))
textcnn.eval()

variant_ids = encode_texts(variants["review"], character_to_id, config.max_length)
with torch.inference_mode():
    textcnn_probabilities = torch.softmax(textcnn(variant_ids), dim=1).numpy()

output = variants[["variant"]].copy()
output["字符数"] = variants["review"].str.len()
output["TF-IDF预测"] = tfidf_probabilities.argmax(axis=1) + 1
output["TF-IDF一星概率"] = tfidf_probabilities[:, 0]
output["TextCNN预测"] = textcnn_probabilities.argmax(axis=1) + 1
output["TextCNN一星概率"] = textcnn_probabilities[:, 0]

print(f"样本 5124 真实星级：{int(case['star'])}")
print(output.round(4).to_string(index=False))
print("\n这是单样本扰动实验，只能检验这个案例中的模型敏感性，不能直接代表整个数据集。")

output.to_csv(
    RESULTS_DIR / "case_5124_perturbation.csv",
    index=False,
    encoding="utf-8-sig",
)
print("已保存 results/case_5124_perturbation.csv")
