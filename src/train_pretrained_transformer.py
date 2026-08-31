"""轻量中文预训练 Transformer 的微调实验。

本脚本只使用 train.csv 和 dev.csv；测试集在本阶段不读取。
长评论沿用“开头 + 结尾”的 128 token 表示。先运行小规模冒烟实验，
再显式选择全训练集的验证集实验。
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from inspect_pretrained_tokenizer import MODEL_NAME, encode_with_head_tail


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"


@dataclass(frozen=True)
class Config:
    max_length: int = 128
    batch_size: int = 16
    epochs: int = 3
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    random_seed: int = 42
    smoke_train_rows: int = 512
    smoke_dev_rows: int = 256


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_split(filename: str) -> pd.DataFrame:
    frame = pd.read_csv(DATA_DIR / filename, usecols=["id", "review", "star"])
    if frame[["review", "star"]].isna().any().any():
        raise ValueError(f"{filename} 中存在空评论或空星级")
    return frame


def encode_reviews(
    tokenizer, reviews: pd.Series, max_length: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """将每条评论变为 input_ids、attention_mask、token_type_ids 三个张量。"""
    encoded_reviews = [
        encode_with_head_tail(tokenizer, text, max_length)
        for text in reviews.astype(str)
    ]
    input_ids = torch.tensor(
        [item["input_ids"] for item in encoded_reviews], dtype=torch.long
    )
    attention_masks = torch.tensor(
        [item["attention_mask"] for item in encoded_reviews], dtype=torch.long
    )
    token_type_ids = torch.tensor(
        [item["token_type_ids"] for item in encoded_reviews], dtype=torch.long
    )
    truncated_count = sum(item["truncated"] for item in encoded_reviews)
    return input_ids, attention_masks, token_type_ids, truncated_count


def make_loader(
    input_ids: torch.Tensor,
    attention_masks: torch.Tensor,
    token_type_ids: torch.Tensor,
    labels: torch.Tensor,
    config: Config,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        TensorDataset(input_ids, attention_masks, token_type_ids, labels),
        batch_size=config.batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(config.random_seed),
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_rows = 0

    for input_ids, attention_masks, token_type_ids, labels in loader:
        input_ids = input_ids.to(device)
        attention_masks = attention_masks.to(device)
        token_type_ids = token_type_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        output = model(
            input_ids=input_ids,
            attention_mask=attention_masks,
            token_type_ids=token_type_ids,
        )
        loss = loss_function(output.logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        total_rows += len(labels)

    return total_loss / total_rows


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_rows = 0
    all_labels: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []

    for input_ids, attention_masks, token_type_ids, labels in loader:
        input_ids = input_ids.to(device)
        attention_masks = attention_masks.to(device)
        token_type_ids = token_type_ids.to(device)
        labels = labels.to(device)

        output = model(
            input_ids=input_ids,
            attention_mask=attention_masks,
            token_type_ids=token_type_ids,
        )
        loss = loss_function(output.logits, labels)
        predictions = output.logits.argmax(dim=1)

        total_loss += loss.item() * len(labels)
        total_rows += len(labels)
        all_labels.append(labels.cpu().numpy())
        all_predictions.append(predictions.cpu().numpy())

    return (
        total_loss / total_rows,
        np.concatenate(all_labels),
        np.concatenate(all_predictions),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--smoke-test",
        action="store_true",
        help="用固定的小样本训练 3 轮，检查微调流程及按验证集选轮次的逻辑。",
    )
    mode_group.add_argument(
        "--full-experiment",
        action="store_true",
        help="使用完整 train/dev 划分训练 3 轮；仍不读取测试集。",
    )
    args = parser.parse_args()

    config = Config()
    set_seed(config.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train = read_split("train.csv")
    dev = read_split("dev.csv")
    mode = "smoke" if args.smoke_test else "full"
    if args.smoke_test:
        train = train.sample(
            n=config.smoke_train_rows, random_state=config.random_seed
        )
        dev = dev.sample(
            n=config.smoke_dev_rows, random_state=config.random_seed
        )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ids, train_masks, train_types, train_truncated = encode_reviews(
        tokenizer, train["review"], config.max_length
    )
    dev_ids, dev_masks, dev_types, dev_truncated = encode_reviews(
        tokenizer, dev["review"], config.max_length
    )
    # 原始星级是 1~5；交叉熵损失要求分类编号为 0~4。
    train_labels = torch.tensor(train["star"].astype(int).to_numpy() - 1)
    dev_labels = torch.tensor(dev["star"].astype(int).to_numpy() - 1)

    train_loader = make_loader(
        train_ids, train_masks, train_types, train_labels, config, shuffle=True
    )
    dev_loader = make_loader(
        dev_ids, dev_masks, dev_types, dev_labels, config, shuffle=False
    )

    # 预训练 Transformer 参数和新建的五分类层都会参与反向传播。
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=5
    ).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    if args.smoke_test:
        print("mode=smoke (this is a pipeline check, not a model comparison)")
    else:
        print("mode=full (development-set experiment; test split is unused)")
    print(f"device={device}")
    print(f"train/dev rows={len(train)}/{len(dev)}")
    print(f"input tensor shape={tuple(train_ids.shape)}")
    print(f"train/dev truncated={train_truncated}/{dev_truncated}")
    print(f"model parameters={sum(parameter.numel() for parameter in model.parameters())}")

    history: list[dict[str, float | int]] = []
    best_epoch = 0
    best_macro_f1 = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    started = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_function, device
        )
        dev_loss, true_stars, predicted_stars = evaluate(
            model, dev_loader, loss_function, device
        )
        row = {
            "epoch": epoch,
            "train_loss": round(float(train_loss), 4),
            "dev_loss": round(float(dev_loss), 4),
            "dev_accuracy": round(float(accuracy_score(true_stars, predicted_stars)), 4),
            "dev_macro_f1": round(
                float(f1_score(true_stars, predicted_stars, average="macro", zero_division=0)),
                4,
            ),
        }
        history.append(row)
        print(row)

        if row["dev_macro_f1"] > best_macro_f1:
            best_epoch = epoch
            best_macro_f1 = float(row["dev_macro_f1"])
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("训练没有产生可保存的模型")

    model.load_state_dict(best_state)
    _, true_stars, predicted_stars = evaluate(
        model, dev_loader, loss_function, device
    )
    RESULTS_DIR.mkdir(exist_ok=True)
    purpose = (
        "Verify the fine-tuning pipeline only; do not compare this sample result with full-data baselines."
        if args.smoke_test
        else "Train on the complete train split and select an epoch by development Macro-F1; the test split is deliberately unused."
    )
    metrics = {
        "mode": mode,
        "purpose": purpose,
        "model_name": MODEL_NAME,
        "input_strategy": "short: [CLS] text [SEP]; long: [CLS] head [SEP] tail [SEP]",
        "config": asdict(config),
        "device": str(device),
        "train_truncated_reviews": train_truncated,
        "dev_truncated_reviews": dev_truncated,
        "best_epoch": best_epoch,
        "best_dev_accuracy": float(accuracy_score(true_stars, predicted_stars)),
        "best_dev_macro_f1": float(
            f1_score(true_stars, predicted_stars, average="macro", zero_division=0)
        ),
        "best_dev_mae": float(mean_absolute_error(true_stars, predicted_stars)),
        "training_seconds": time.perf_counter() - started,
        "history": history,
    }
    result_stem = f"pretrained_mini_{mode}"
    metrics_path = RESULTS_DIR / f"{result_stem}_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    torch.save(best_state, RESULTS_DIR / f"{result_stem}_model.pt")
    if args.full_experiment:
        pd.DataFrame(
            {
                "id": dev["id"],
                "review": dev["review"],
                "true_star": true_stars + 1,
                "predicted_star": predicted_stars + 1,
            }
        ).to_csv(
            RESULTS_DIR / "pretrained_mini_full_dev_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )
    print(f"best epoch={best_epoch}, best dev Macro-F1={best_macro_f1:.4f}")
    print(f"saved: {metrics_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
