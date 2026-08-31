"""字符级 TextCNN：只根据 ASAP 评论文本预测 1~5 星。"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"
PAD_ID = 0
UNK_ID = 1
STAR_LABELS = [0, 1, 2, 3, 4]


# 1. 实验设置：所有会影响复现和比较的主要选择集中放在这里。
@dataclass(frozen=True)
class Config:
    max_length: int = 512
    max_vocab_size: int = 8_000
    min_char_frequency: int = 2
    embedding_dim: int = 32
    channels_per_kernel: int = 32
    kernel_sizes: tuple[int, ...] = (3, 4, 5)
    batch_size: int = 256
    epochs: int = 8
    learning_rate: float = 0.002
    weight_decay: float = 0.0001
    dropout: float = 0.3
    random_seed: int = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# 2. 文本表示：只用训练集建立字符词表，再把评论变成固定长度的字符 ID。
def build_vocabulary(texts: pd.Series, config: Config) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for text in texts.astype(str):
        counts.update(text)

    characters = [
        character
        for character, frequency in sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        )
        if frequency >= config.min_char_frequency
    ][: config.max_vocab_size - 2]

    return {
        character: character_id
        for character_id, character in enumerate(["<PAD>", "<UNK>", *characters])
    }


def keep_head_and_tail(text: str, max_length: int) -> list[str]:
    characters = list(str(text))
    if len(characters) <= max_length:
        return characters
    head_length = max_length // 2
    return characters[:head_length] + characters[-(max_length - head_length) :]


def encode_texts(
    texts: pd.Series, character_to_id: dict[str, int], max_length: int
) -> torch.Tensor:
    encoded = torch.full(
        (len(texts), max_length), fill_value=PAD_ID, dtype=torch.long
    )
    for row_index, text in enumerate(texts.astype(str)):
        characters = keep_head_and_tail(text, max_length)
        character_ids = [
            character_to_id.get(character, UNK_ID) for character in characters
        ]
        if character_ids:
            encoded[row_index, : len(character_ids)] = torch.tensor(character_ids)
    return encoded


# 3. 模型：Embedding -> 3/4/5 字卷积 -> 最大池化 -> 五分类分数。
class TextCNN(nn.Module):
    def __init__(self, vocab_size: int, config: Config) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, config.embedding_dim, padding_idx=PAD_ID
        )
        self.convolutions = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=config.embedding_dim,
                    out_channels=config.channels_per_kernel,
                    kernel_size=kernel_size,
                    bias=False,
                )
                for kernel_size in config.kernel_sizes
            ]
        )
        feature_count = config.channels_per_kernel * len(config.kernel_sizes)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(feature_count, 5)

    def forward(self, character_ids: torch.Tensor) -> torch.Tensor:
        # [批量, 长度] -> [批量, 向量维度, 长度]
        embedded = self.embedding(character_ids).transpose(1, 2)

        # 每种卷积得到 [批量, 卷积核数, 位置数]，池化后只保留位置最大值。
        pooled_features = [
            torch.relu(convolution(embedded)).amax(dim=2)
            for convolution in self.convolutions
        ]
        review_features = torch.cat(pooled_features, dim=1)
        return self.classifier(self.dropout(review_features))


# 4. 训练与验证：训练集更新参数；验证集只计算指标。
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
) -> float:
    model.train()
    total_loss = 0.0
    total_loss_weight = 0.0

    for character_ids, true_stars in loader:
        optimizer.zero_grad()
        star_scores = model(character_ids)
        loss = loss_function(star_scores, true_stars)
        loss.backward()
        optimizer.step()

        if isinstance(loss_function, nn.CrossEntropyLoss) and loss_function.weight is not None:
            batch_loss_weight = float(loss_function.weight[true_stars].sum())
        else:
            batch_loss_weight = float(len(true_stars))
        total_loss += loss.item() * batch_loss_weight
        total_loss_weight += batch_loss_weight

    return total_loss / total_loss_weight


@torch.inference_mode()
def evaluate(
    model: nn.Module, loader: DataLoader, loss_function: nn.Module
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_loss_weight = 0.0
    all_true_stars: list[np.ndarray] = []
    all_predicted_stars: list[np.ndarray] = []

    for character_ids, true_stars in loader:
        star_scores = model(character_ids)
        loss = loss_function(star_scores, true_stars)
        predicted_stars = star_scores.argmax(dim=1)

        if isinstance(loss_function, nn.CrossEntropyLoss) and loss_function.weight is not None:
            batch_loss_weight = float(loss_function.weight[true_stars].sum())
        else:
            batch_loss_weight = float(len(true_stars))
        total_loss += loss.item() * batch_loss_weight
        total_loss_weight += batch_loss_weight
        all_true_stars.append(true_stars.numpy())
        all_predicted_stars.append(predicted_stars.numpy())

    return (
        total_loss / total_loss_weight,
        np.concatenate(all_true_stars),
        np.concatenate(all_predicted_stars),
    )


def read_split(filename: str) -> pd.DataFrame:
    frame = pd.read_csv(DATA_DIR / filename, usecols=["id", "review", "star"])
    if frame[["review", "star"]].isna().any().any():
        raise ValueError(f"{filename} 中存在空评论或空星级")
    return frame


def make_loader(
    character_ids: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    return DataLoader(
        TensorDataset(character_ids, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
    )


def make_loss_function(
    train_labels: torch.Tensor, class_weighting: str
) -> tuple[nn.Module, torch.Tensor | None]:
    """按训练集标签决定交叉熵是否对少数类别加权。"""
    if class_weighting == "none":
        return nn.CrossEntropyLoss(), None

    # balanced: w_c = N / (类别数 * 该类别训练样本数)。
    # 因此每个星级在一个 epoch 中的总损失权重大致相同。
    class_counts = torch.bincount(train_labels, minlength=len(STAR_LABELS)).float()
    if torch.any(class_counts == 0):
        raise ValueError("训练集中存在没有样本的星级，无法计算平衡类别权重")
    class_weights = len(train_labels) / (len(STAR_LABELS) * class_counts)
    return nn.CrossEntropyLoss(weight=class_weights), class_weights


def per_class_metrics(
    true_stars: np.ndarray, predicted_stars: np.ndarray
) -> dict[str, dict[str, float | int]]:
    precision, recall, f1, support = precision_recall_fscore_support(
        true_stars,
        predicted_stars,
        labels=STAR_LABELS,
        average=None,
        zero_division=0,
    )
    return {
        str(star + 1): {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, star in enumerate(STAR_LABELS)
    }


# 5. 完整实验：每轮验证并保存验证集 Macro-F1 最好的模型。
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="只用少量数据训练一轮，先检查完整流程能否运行。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="控制参数初始化、训练顺序和 Dropout 的随机种子。",
    )
    parser.add_argument(
        "--class-weighting",
        choices=["none", "balanced"],
        default="none",
        help="none 为普通交叉熵；balanced 按训练集类别频率加权。",
    )
    parser.add_argument(
        "--run-name",
        help="实验结果文件名中的自定义部分；省略时保持原来的结果文件名。",
    )
    args = parser.parse_args()

    config = (
        Config(epochs=3, random_seed=args.seed)
        if args.smoke_test
        else Config(random_seed=args.seed)
    )
    set_seed(config.random_seed)

    train = read_split("train.csv")
    dev = read_split("dev.csv")
    if args.smoke_test:
        train = train.sample(n=2_000, random_state=42).reset_index(drop=True)
        dev = dev.sample(n=500, random_state=42).reset_index(drop=True)

    character_to_id = build_vocabulary(train["review"], config)
    train_ids = encode_texts(train["review"], character_to_id, config.max_length)
    dev_ids = encode_texts(dev["review"], character_to_id, config.max_length)
    # 原始星级是 1~5；CrossEntropyLoss 要求类别编号为 0~4。
    train_labels = torch.tensor(train["star"].to_numpy() - 1, dtype=torch.long)
    dev_labels = torch.tensor(dev["star"].to_numpy() - 1, dtype=torch.long)

    train_loader = make_loader(
        train_ids, train_labels, config.batch_size, True, config.random_seed
    )
    dev_loader = make_loader(
        dev_ids, dev_labels, config.batch_size, False, config.random_seed
    )

    model = TextCNN(len(character_to_id), config)
    loss_function, class_weights = make_loss_function(
        train_labels, args.class_weighting
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    print(f"mode={'smoke' if args.smoke_test else 'full'}")
    print(f"train/dev rows={len(train)}/{len(dev)}")
    print(f"vocab size={len(character_to_id)}")
    print(f"train tensor shape={tuple(train_ids.shape)}")
    print(f"model parameters={sum(p.numel() for p in model.parameters())}")
    print(f"class weighting={args.class_weighting}")
    if class_weights is not None:
        print(
            "class weights="
            + str({star + 1: round(float(weight), 4) for star, weight in enumerate(class_weights)})
        )

    history: list[dict[str, float | int]] = []
    best_epoch = 0
    best_macro_f1 = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    started = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_function)
        dev_loss, true_stars, predicted_stars = evaluate(
            model, dev_loader, loss_function
        )
        accuracy = accuracy_score(true_stars, predicted_stars)
        macro_f1 = f1_score(true_stars, predicted_stars, average="macro")
        row = {
            "epoch": epoch,
            "train_loss": round(float(train_loss), 4),
            "dev_loss": round(float(dev_loss), 4),
            "dev_accuracy": round(float(accuracy), 4),
            "dev_macro_f1": round(float(macro_f1), 4),
        }
        history.append(row)
        print(row)

        if macro_f1 > best_macro_f1:
            best_epoch = epoch
            best_macro_f1 = float(macro_f1)
            best_state = {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("训练没有产生可保存的模型")
    model.load_state_dict(best_state)
    _, true_stars, predicted_stars = evaluate(model, dev_loader, loss_function)

    RESULTS_DIR.mkdir(exist_ok=True)
    mode = "smoke" if args.smoke_test else "full"
    if args.run_name:
        result_stem = f"textcnn_{args.run_name}"
    elif args.class_weighting == "none":
        # 保持第一阶段默认文件名，供冻结模型评价脚本继续读取。
        result_stem = f"textcnn_{mode}"
    else:
        result_stem = (
            f"textcnn_{mode}_{args.class_weighting}_seed{config.random_seed}"
        )
    metrics = {
        "mode": mode,
        "config": asdict(config),
        "class_weighting": args.class_weighting,
        "class_weights": (
            None
            if class_weights is None
            else {
                str(star + 1): float(weight)
                for star, weight in enumerate(class_weights)
            }
        ),
        "best_epoch": best_epoch,
        "best_dev_accuracy": float(accuracy_score(true_stars, predicted_stars)),
        "best_dev_macro_f1": float(
            f1_score(true_stars, predicted_stars, average="macro")
        ),
        "best_dev_mae": float(mean_absolute_error(true_stars, predicted_stars)),
        "per_class": per_class_metrics(true_stars, predicted_stars),
        "training_seconds": time.perf_counter() - started,
        "history": history,
    }
    metrics_path = RESULTS_DIR / f"{result_stem}_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    torch.save(best_state, RESULTS_DIR / f"{result_stem}_model.pt")
    if not args.smoke_test and args.run_name is None:
        pd.DataFrame(
            {
                "id": dev["id"],
                "review": dev["review"],
                "true_star": true_stars + 1,
                "predicted_star": predicted_stars + 1,
            }
        ).to_csv(
            RESULTS_DIR / "textcnn_dev_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )

    print(f"best epoch={best_epoch}, best dev Macro-F1={best_macro_f1:.4f}")
    print(f"saved: {metrics_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
