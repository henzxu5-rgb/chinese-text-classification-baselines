# 结果文件

- `frozen_test_metrics.json`：最终测试集的完整指标、逐类结果和混淆矩阵；
- `textcnn_full_metrics.json`：TextCNN 每轮训练记录和最佳轮次；
- `model_comparison_summary.json`：两个模型在验证集上的正确与错误数量；
- `model_disagreement_sample.csv`：少量模型分歧样本；
- `textcnn_class_weight_*_metrics.json`：三个随机种子下普通/加权 TextCNN 的验证集结果；
- `class_weight_experiment_summary.json`：类别加权验证集结果汇总；
- `class_weight_test_metrics.json`：已固定模型的类别加权测试集比较；
- `experiment_log.md`：主要实验记录；
- `error_analysis.md`：人工错误观察。

模型权重、完整数据和大型逐条预测文件保留在本地，不提交到 Git，可以通过代码重新生成。
