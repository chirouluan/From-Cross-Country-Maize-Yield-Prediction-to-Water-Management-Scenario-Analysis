# Stage 1 当前实验协议

## 问题

检验原始 Qwen 与结构化气候 RAG 增强的原始 Qwen 所给出的地域化特征参数，是否能在多类回归模型上改进固定 CY-Bench 特征。固定特征是必要基线，原始 Qwen 是无检索消融；LoRA 不再属于当前正式流程。

## 防止信息泄漏

每个国家最后 30% 年份只用于最终外层评测。LLM 看到的气候画像、胁迫证据、候选筛选和特征数选择都来自更早的校准年份。RAG 直接查询 `Climate_rag_knowledge_base/processed/climate_rag_metrics.csv`，以 crop、country 和 `year <= calibration_end_year` 硬过滤，再以透明的标量规则抽取典型、热旱、冷湿和长干旱年份；不使用 embedding 或纯向量搜索。外层结果不参与参数选择。

## 参数与模型

每个非固定来源提出 3 套参数。候选可直接验证，也可扩展为 low/mid/high。每个回归模型可在校准期独立选择候选，随后在完全相同的外层年份分别运行 Ridge、XGBoost、SVR、CNN1D 和 TransformerFlat。神经网络默认最多 50 epochs，并使用内部早停。

最终结论必须查看五类模型的完整结果，不能用 Ridge 单独代表 Stage 1。

## 可复现入口

`prepare.py` 管理 CSV 检索、推荐、内部预测和最终配置；`run.py` 生成特征并执行外层实验；`summarize.py` 校验并生成宽表和宏平均。检索命中行另存为 `retrieval_qwen_rag.csv`，原始推荐 JSON 内保留查询条件、CSV SHA-256 和完整检索证据。
