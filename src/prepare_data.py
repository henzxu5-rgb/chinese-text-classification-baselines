"""Prepare the official ASAP train/dev/test CSV files for this project.

The script deliberately keeps only the fields used by this experiment:
sample id, review text, and 1--5 star label.  It also checks the expected
split sizes so that a wrong or incomplete data source fails loudly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"
EXPECTED_ROWS = {"train": 36_850, "dev": 4_940, "test": 4_940}
REQUIRED_COLUMNS = ["id", "review", "star"]
COLUMN_ALIASES = {"index": "id", "reviewbody": "review"}


def normalize_split(source_path: Path, split: str) -> pd.DataFrame:
    data = pd.read_csv(source_path).rename(columns=COLUMN_ALIASES)

    missing_columns = [name for name in REQUIRED_COLUMNS if name not in data]
    if missing_columns:
        raise ValueError(f"{source_path} 缺少列: {missing_columns}")

    data = data[REQUIRED_COLUMNS].copy()
    if len(data) != EXPECTED_ROWS[split]:
        raise ValueError(
            f"{split} 应有 {EXPECTED_ROWS[split]} 行，实际为 {len(data)} 行"
        )
    if data[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError(f"{split} 的 id、review 或 star 中存在空值")
    if data["id"].duplicated().any():
        raise ValueError(f"{split} 内存在重复 id")

    numeric_star = pd.to_numeric(data["star"], errors="raise")
    if not numeric_star.between(1, 5).all():
        raise ValueError(f"{split} 的 star 必须全部位于 1--5")
    if not (numeric_star % 1 == 0).all():
        raise ValueError(f"{split} 的 star 必须是整数")
    data["star"] = numeric_star.astype(int)
    data["review"] = data["review"].astype(str)
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从官方 ASAP 仓库准备本项目使用的 train/dev/test.csv"
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="包含 train.csv、dev.csv、test.csv 的 ASAP data 目录",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"输出目录（默认: {DEFAULT_OUTPUT_DIR}）",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已经存在的输出文件",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prepared: dict[str, pd.DataFrame] = {}
    for split in EXPECTED_ROWS:
        source_path = source_dir / f"{split}.csv"
        output_path = output_dir / f"{split}.csv"
        if not source_path.is_file():
            raise FileNotFoundError(f"找不到源文件: {source_path}")
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"输出已存在: {output_path}；确认后可加 --overwrite 覆盖"
            )
        prepared[split] = normalize_split(source_path, split)

    all_ids = pd.concat(
        [frame[["id"]].assign(split=split) for split, frame in prepared.items()],
        ignore_index=True,
    )
    duplicated_across_splits = all_ids[all_ids["id"].duplicated(keep=False)]
    if not duplicated_across_splits.empty:
        raise ValueError("train/dev/test 之间存在重复 id，可能发生数据泄漏")

    for split, data in prepared.items():
        output_path = output_dir / f"{split}.csv"
        data.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"{split}: {len(data)} rows -> {output_path}")


if __name__ == "__main__":
    main()
