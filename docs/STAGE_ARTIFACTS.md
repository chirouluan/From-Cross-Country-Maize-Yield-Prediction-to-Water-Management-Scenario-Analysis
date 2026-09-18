# 三阶段数据产物协议

三个阶段通过磁盘产物衔接。CSV 便于人工检查，相邻的 `.manifest.json` 记录阶段、产物类型、行列结构、metadata 和 SHA-256；下游会据此验证来源和完整性。

## Stage 1

正式根目录：`cybench/output/stages/stage1_feature_engineering/`。

推荐与选择证据：

- `recommendations/raw/maize_<country>/recommendation_<advisor>.json`：原始 Qwen 或 RAG Qwen 候选；
- `recommendations/raw/maize_<country>/retrieval_qwen_rag.csv`：实际命中的结构化历史气候行及检索角色；
- `recommendations/calibration/maize_<country>/<advisor>/inner_candidate_predictions_<advisor>_<model>.csv`：只使用校准年份的五次候选预测；
- `recommendations/calibration/.../selection_scores.csv`：五次 pooled MSE 的均值和方差；
- `recommendations/selected_configs/maize_<country>/<advisor>/selected_config.json`：最终参数和特征数。

外层评测：

- `results/maize_<country>/<advisor>/feature_dataset.csv`：该配置生成的完整特征数据；
- `results/.../predictions_<model>.csv`：带 `train/test`、`repeat` 和 `seed` 的逐样本预测；
- `results/.../metrics_<model>_repeats.csv`：五次训练集和测试集指标；
- `results/metrics_all.csv`：全部逐次指标；
- `results/metrics_summary.csv`：各指标均值和样本方差；
- `results/nrmse_by_country_model.csv`、`r2_by_country_model.csv`、`macro_summary.csv`：便于查看的汇总。

当前国家为 `CN、ES、PT、ZA、ZM、IT、MX`；参数来源为 `hardcode、qwen_base、qwen_rag`，其中前两者分别作为硬编码基线和无检索消融；模型为 `ridge、xgboost、svr、cnn1d、transformer_flat`。

Stage 2/3 统一读取 `results/maize_<country>/qwen_rag/feature_dataset.csv`，确保两个下游阶段使用同一套 Base-Qwen + 结构化 CSV RAG 特征。

## Stage 2

- `cn_finetuning/maize_CN_train.csv`、`maize_CN_validation.csv`：继续训练输入；
- `evaluation/maize_<country>/input_dataset.csv`：Stage 1 输入快照；
- `evaluation/.../predictions_tabpfn_base.csv`；
- `evaluation/.../predictions_tabpfn_cn_finetuned.csv`；
- `evaluation/.../metrics.csv`。

Stage 2 的预测与指标同样保存 5 次 repeat、seed、train/test，并生成 `metrics_summary.csv`。

预测 manifest 同时记录 Stage 1 数据和 TabPFN checkpoint 的 SHA-256。

## Stage 3

- `model_training_data.csv`：严格早于决策年份的代理模型数据；
- `model_predictions.csv`：ExtraTrees 五次 train/test 预测；
- `model_validation_repeats.csv` 和 `model_validation_summary.csv`：逐次指标及均值/方差；
- `decision_input_snapshot.csv` 和 `stage2_forecast_context.csv`；
- 每个策略分支的 `stress_diagnostics.csv`、`recovery_scenarios.csv`、`accepted_proxy_changes.csv`、`scenario_summary.csv`；
- Qwen 策略、建议及证据校验 JSON；
- `policy_comparison.csv`。


## 续跑

```powershell
conda run -n specllm python -m cybench.stages.stage1_feature_engineering.prepare --resume
conda run -n specllm python -m cybench.stages.stage1_feature_engineering.run --resume
conda run -n specllm python -m cybench.stages.stage2_tabpfn_transfer.run
conda run -n specllm python -m cybench.stages.stage3_advisory.run
```

不要直接修改正式产物；人工分析时请复制到其他目录。修改 CSV 后 manifest 哈希将不再匹配。
