# 数据说明

本项目使用美团公开的 [ASAP（A Chinese Review Dataset Towards Aspect Category Sentiment Analysis and Rating Prediction）](https://github.com/Meituan-Dianping/asap) 数据集。数据集原仓库采用 Apache-2.0 许可证；如需使用或再分发，请同时核对原仓库的说明与许可证。

## 本实验使用的字段

| 字段 | 含义 |
| --- | --- |
| `id` | 样本标识 |
| `review` | 中文评论正文，也是唯一模型输入 |
| `star` | 1--5 星分类标签 |

原始数据中的 aspect 标签不进入本实验。这个边界是主动做出的研究取舍：先聚焦评论正文与总体星级的关系，减少十天冲刺阶段的工作量和混杂因素。

## 目录约定

```text
data/
├─ external/asap/       官方仓库的本地克隆，Git 忽略
├─ raw/                 准备后的完整 train/dev/test.csv，Git 忽略
└─ sample/              可提交的小型观察样本
```

完整数据不随本仓库提交。准备步骤如下（在项目根目录运行）：

```powershell
git clone https://github.com/Meituan-Dianping/asap.git .\data\external\asap
.\.venv\Scripts\python.exe .\src\prepare_data.py `
  --source-dir .\data\external\asap\data
```

脚本只保留 `id,review,star`，并检查：

- train/dev/test 行数分别为 36,850 / 4,940 / 4,940；
- 三个必需字段无空值；
- 每个划分内部以及不同划分之间没有重复 `id`；
- `star` 是 1--5 的整数。

若 `data/raw/` 已有文件，脚本默认拒绝覆盖。只有确认目标后才添加 `--overwrite`。

## 小样本

`sample/asap_train_sample.csv` 用于观察字段、评论文本与标签，完整统计和模型训练使用官方完整数据。
