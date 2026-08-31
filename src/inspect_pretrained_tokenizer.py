"""观察预训练中文 Transformer 如何把真实评论转换为模型输入。"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_NAME = "uer/roberta-mini-wwm-chinese-cluecorpussmall"


def choose_examples(frame: pd.DataFrame) -> pd.DataFrame:
    """选择最短、中等、最长三条真实评论，便于观察 padding 与截断。"""
    with_lengths = frame.assign(review_length=frame["review"].astype(str).str.len())
    shortest = with_lengths["review_length"].idxmin()
    median_length = with_lengths["review_length"].median()
    middle = (with_lengths["review_length"] - median_length).abs().idxmin()
    longest = with_lengths["review_length"].idxmax()
    chosen_indices = [shortest, middle, longest]
    return with_lengths.loc[chosen_indices, ["id", "review", "star", "review_length"]]


def encode_with_head_tail(tokenizer, text: str, max_length: int) -> dict:
    """短评论保留完整；长评论以 [CLS] 开头片段 [SEP] 结尾片段 [SEP] 输入。"""
    original_ids = tokenizer(str(text), add_special_tokens=False)["input_ids"]
    single_special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
    pair_special_tokens = tokenizer.num_special_tokens_to_add(pair=True)

    if len(original_ids) + single_special_tokens <= max_length:
        first_ids = original_ids
        second_ids: list[int] | None = None
        truncated = False
    else:
        usable_content_length = max_length - pair_special_tokens
        head_length = usable_content_length // 2
        tail_length = usable_content_length - head_length
        first_ids = original_ids[:head_length]
        second_ids = original_ids[-tail_length:]
        truncated = True

    # transformers 5 中这个分词器不再暴露旧版的
    # build_inputs_with_special_tokens。这里显式拼接，输入结构也更直观：
    # 短文本：[CLS] 全文 [SEP]
    # 长文本：[CLS] 开头片段 [SEP] 结尾片段 [SEP]
    if tokenizer.cls_token_id is None or tokenizer.sep_token_id is None:
        raise ValueError("当前分词器缺少 [CLS] 或 [SEP] 特殊标记")

    if second_ids is None:
        input_ids = [tokenizer.cls_token_id, *first_ids, tokenizer.sep_token_id]
        token_type_ids = [0] * len(input_ids)
    else:
        input_ids = [
            tokenizer.cls_token_id,
            *first_ids,
            tokenizer.sep_token_id,
            *second_ids,
            tokenizer.sep_token_id,
        ]
        # 0 标记开头片段，1 标记结尾片段；模型可据此区分两个片段。
        token_type_ids = [0] * (len(first_ids) + 2) + [1] * (len(second_ids) + 1)
    attention_mask = [1] * len(input_ids)
    padding_length = max_length - len(input_ids)
    if padding_length < 0:
        raise ValueError("tokenizer 输出超过设定的最大长度")

    input_ids += [tokenizer.pad_token_id] * padding_length
    token_type_ids += [0] * padding_length
    attention_mask += [0] * padding_length
    return {
        "original_token_count": len(original_ids),
        "truncated": truncated,
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="显示用长度；默认与后续微调的最大长度一致。",
    )
    args = parser.parse_args()

    dev = pd.read_csv(DATA_DIR / "dev.csv", usecols=["id", "review", "star"])
    examples = choose_examples(dev)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"model={MODEL_NAME}")
    print(f"display max_length={args.max_length}")
    for position, row in enumerate(examples.itertuples(index=False), start=1):
        encoded = encode_with_head_tail(
            tokenizer, str(row.review), args.max_length
        )
        tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
        kept_token_count = sum(encoded["attention_mask"])
        kept_tokens = tokens[:kept_token_count]
        preview = str(row.review)
        if len(preview) > 100:
            preview = preview[:100] + "..."

        print(f"\nexample={position}, id={row.id}, star={int(row.star)}")
        print(f"original length={row.review_length}")
        print(f"review={preview}")
        print(f"tokens={kept_tokens}")
        print(f"input_ids={encoded['input_ids'][:kept_token_count]}")
        print(f"token_type_ids={encoded['token_type_ids'][:kept_token_count]}")
        print(
            "kept tokens="
            f"{kept_token_count}/{args.max_length}; "
            f"padding tokens={args.max_length - kept_token_count}; "
            f"original content tokens={encoded['original_token_count']}; "
            f"truncated={encoded['truncated']}"
        )
        print(
            "attention_mask tail="
            f"{encoded['attention_mask'][max(0, args.max_length - 12):]}"
        )


if __name__ == "__main__":
    main()
