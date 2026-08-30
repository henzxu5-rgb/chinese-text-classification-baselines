# 中文评论星级分类基线

本项目使用中文餐厅评论预测用户给出的 1—5 星评分，并比较两种文本分类方法：

- TF-IDF + 逻辑回归；
- TextCNN 轻量神经网络。

这个项目的重点不是提出新模型，而是走完一次比较完整的实验流程：检查数据、划分训练集和测试集、训练模型、比较指标、阅读错误样本，并记录模型的局限。

## 主要工作

- 检查数据中的空值、重复、类别分布和评论长度；
- 确认只使用评论正文预测星级，不使用数据中的方面情感标签；
- 运行并比较 TF-IDF + 逻辑回归和 TextCNN；
- 将 TF-IDF 的字符片段范围从 `(2,4)` 改为 `(2,2)`，比较修改前后的结果；
- 使用 Accuracy、Macro-F1 和 MAE 评价模型；
- 阅读真实错误样本，比较两个模型会在哪些评论上作出不同判断；
- 在配置确定后只进行一次最终测试集评价。

## 数据

项目使用美团公开的 [ASAP 中文评论数据集](https://github.com/Meituan-Dianping/asap)。官方划分包括：

| 训练集 | 验证集 | 测试集 |
| ---: | ---: | ---: |
| 36,850 | 4,940 | 4,940 |

模型输入是评论正文 `review`，预测目标是星级 `star`。训练集中四星和五星评论明显更多，因此只看 Accuracy 可能会高估模型的实际表现。

## 方法

TF-IDF 将评论切成相邻的二字片段，再把每条评论表示成稀疏向量；逻辑回归根据这些片段学习不同星级的判断边界。

TextCNN 先把字符转换成向量，再用宽度为 3、4、5 的卷积核提取局部特征，最后预测五个星级。它可以学习局部组合，但仍不等于真正理解整句评论。

## 实验结果

最终测试集结果如下：

| 模型 | Accuracy | Macro-F1 | MAE |
| --- | ---: | ---: | ---: |
| TF-IDF + 逻辑回归 | 0.5605 | **0.4577** | 0.5004 |
| TextCNN | **0.5634** | 0.4354 | **0.4988** |

TextCNN 的 Accuracy 略高，但只比 TF-IDF 多预测正确 14 条评论。TF-IDF 的 Macro-F1 更高，说明它在五个星级之间的平均表现更好。

结果表明，只看一个指标可能得到片面的结论。TextCNN 在数量较多的四星评论上表现更好，因此 Accuracy 略高；但它在二星、三星等类别上的表现较差，所以 Macro-F1 反而更低。

## 错误观察

错误样本中主要出现三种情况：

1. 有些评论同时包含表扬和批评，相邻星级本来就不容易区分；
2. 有些评论正文与最终评分并不完全一致，只依靠文字可能无法还原用户的真实评分理由；
3. 模型容易受到“五分”“差评”等局部片段影响，却没有很好地结合转折、否定和完整上下文。

两个模型也会答对不同的样本，这说明它们使用的信息可能存在一定互补性。不过目前的案例数量不足以证明哪种方法真正更理解语义。

更详细的内容见 [实验报告](report/experiment_report.md) 和 [误差分析](results/error_analysis.md)。

## 运行方法

以下命令在项目根目录的 Windows PowerShell 中运行。

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt

git clone https://github.com/Meituan-Dianping/asap.git .\data\external\asap
.\.venv\Scripts\python.exe .\src\prepare_data.py `
  --source-dir .\data\external\asap\data

.\.venv\Scripts\python.exe .\src\data_summary.py
.\.venv\Scripts\python.exe .\src\check_splits.py
.\.venv\Scripts\python.exe .\src\train_baseline.py
.\.venv\Scripts\python.exe .\src\train_textcnn.py --smoke-test
.\.venv\Scripts\python.exe .\src\train_textcnn.py
```

完整数据、模型权重和大型逐条预测文件不会提交到 Git，可以通过代码重新生成。最终测试脚本是 `src/evaluate_frozen_models.py`，不应在调参过程中反复运行。

## 目前的不足和下一步

- TextCNN 目前只使用一个随机种子，结果可能存在偶然性；
- 二星样本很少，两种模型的二星识别效果都较差；
- TF-IDF 和 TextCNN 主要依赖局部片段，对完整语义的处理仍然有限；
- 下一步准备尝试类别加权、多随机种子实验，并继续学习预训练语言模型。

## AI 辅助说明

本项目在环境配置、参考代码和调试中使用了 coding agent；实验运行、参数修改、结果核对和错误分析在参考实现基础上完成。该项目属于学习型复现，不代表已经能够从零独立实现全部模型。
