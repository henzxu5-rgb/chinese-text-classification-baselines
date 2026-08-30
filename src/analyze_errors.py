from pathlib import Path

import pandas as pd


project_root = Path(__file__).resolve().parents[1]
results_dir = project_root / "results"
predictions = pd.read_csv(results_dir / "dev_predictions.csv")

distance_counts = predictions["error_distance"].value_counts().sort_index()
within_one = (predictions["error_distance"] <= 1).mean()
mean_error_distance = predictions["error_distance"].mean()

print("星级误差距离数量:")
print(distance_counts)
print("预测与真实星级相差不超过1的比例:", round(within_one, 4))
print("平均绝对星级误差:", round(mean_error_distance, 4))

errors = predictions[predictions["error_distance"] > 0]
adjacent_errors = errors[errors["error_distance"] == 1].sample(
    n=6, random_state=42
)
far_errors = errors[errors["error_distance"] >= 2].sample(
    n=6, random_state=42
)

analysis_sample = pd.concat([adjacent_errors, far_errors], ignore_index=True)
analysis_sample["my_error_type"] = ""
analysis_sample["my_explanation"] = ""
analysis_sample.to_csv(
    results_dir / "error_analysis_candidates.csv", index=False, encoding="utf-8-sig"
)

markdown_lines = [
    "# 人工错误分析样本",
    "",
    "前6条是相邻星级错误，后6条是真实与预测相差至少两星的错误。",
    "请先根据原文、标签和预测独立填写，再归纳共同现象；不必强行套用预设类别。",
]
for number, row in analysis_sample.iterrows():
    review = str(row["review"]).replace("\\n", " ").replace("\\r", " ")
    markdown_lines.extend(
        [
            "",
            f"## 样本 {number + 1}",
            "",
            f"- id：{row['id']}",
            f"- 真实星级：{int(row['true_star'])}",
            f"- 预测星级：{int(row['predicted_star'])}",
            f"- 相差：{int(row['error_distance'])} 星",
            "",
            f"> {review}",
            "",
            "- 我的错误类型：",
            "- 我的解释：",
        ]
    )

(results_dir / "error_analysis_candidates.md").write_text(
    "\n".join(markdown_lines) + "\n", encoding="utf-8"
)
print(
    "待人工分析的12条错误已保存到 "
    "results/error_analysis_candidates.csv 和 results/error_analysis_candidates.md"
)
