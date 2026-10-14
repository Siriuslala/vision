# 实验产物

## 1. 目录规则

新实验的完整产物统一保存于 WORK_DIR 的 `analysis/` 目录。其下一层对应研究问题，再下一层由实验日期与明确实验名称组成。运行目录与文档一一对应，名称直接说明实验日期与所测因素。

| 研究问题 | 新运行目录 | 对应实验 |
| --- | --- | --- |
| 视觉表征差异性 | `analysis/representation/2026-10-01_layouts_random_targets/` | 规则密集网格、布局与网络层比较，以及高范数排除 |
| 视觉表征差异性 | `analysis/representation/2026-10-01_cross_model_random_targets/` | Qwen3-VL-8B-Instruct 的同图像布局对照 |
| 位置信息提取 | `analysis/position/2026-10-01_refcoco_layers/` | 真实图像的 ViT 与 decoder 逐层坐标 probe |
| 词汇读出 | `analysis/token_semantics/2026-10-01_final_layers/` | 末尾 decoder 层及 top-1/2/3 词图 |
| Grounding 行为 | `analysis/behavior/2026-09-30_selection_formats/` | 已有 73 条查询的图示和历史原始运行索引 |
| 视觉细节表征 | `analysis/representation/2026-10-13_activation_patching/` | 实验 2.5a：五个条件的合格配对筛选、层×token 组恢复与破坏扫描、逐文本 token 替换及 float32 复核 |
| 行为诊断 | `analysis/behavior/2026-10-13_decomposition/` | 实验 1.3 正式运行：主评测 37752 条、显式推理 300 条、thinking on 52 条、BabyVision 对应问题 192 条，以及分析与推理文本编码 |
| 行为诊断 | `analysis/behavior/2026-10-13_decomposition_pilot/` | 实验 1.3 的开发运行（每单元 3 张图），只用于确定规模与格式，不进入正式评分 |
| 共享刺激检查 | `analysis/common/stimuli_check/` | 全部模板与属性变化的样例图 |

2026-10-13 起新增的研究问题使用 `analysis/deepstack/` 与 `analysis/reread/`；行为诊断、表征和位置问题的新实验继续使用上表对应的研究问题目录。各实验的计划目录名称见其设计文档的“运行与产物”一节，运行完成后补入上表。

每个目录至少保留参数与模型来源、数据选择和划分、评分或统计结果、日志及图表。大规模特征可以保存固定投影，但报告必须说明投影维度与随机种子。用于文档的图表复制到对应问题的 figures 目录，完整产物仍由 WORK_DIR 管理。

## 2. 历史结果

2026-09-30 的原始运行仍保存在 `grounding/` 下，旧记录和评分不覆盖。新 `analysis/` 下的历史索引指向这些来源，不复制模型特征或将旧运行伪装成新实验。

| 原运行 | 内容 | 文档 |
| --- | --- | --- |
| `grounding/qwen35_9b_pilot_v1/representation/` | 36 个位置的正常图／缺口图对照 | [固定高分位置](representation/results_2026-09-30_pilot.md) |
| `grounding/qwen35_9b_pilot_v2/behavior/` | 32 条 point 查询 | [规模与查询条件](behavior/results_2026-09-30_point.md) |
| `grounding/distance_register_v1/` | 216 个合成改动与 27 个外部改动 | [距离与高范数](representation/results_2026-09-30.md) |
| `grounding/qwen35_9b_slot_probe_v2/` | 62 张独立合成图的位置 probe | [合成图位置实验](position/results_2026-09-30.md) |
| `grounding/qwen35_9b_raw_lens_v1/` | 12 张图的五阶段 raw lens | [初始词汇读出](token_semantics/results_2026-09-30.md) |
| `grounding/comparison_grounding_v1/` | 73 条目标选择／答案格式查询 | [选择条件与答案格式](behavior/results_2026-09-30.md) |

原 pilot 的较短输出预算运行及位置 probe 的开发运行也保留，但不混入后续正式报告的评分。每份报告说明它实际使用的运行版本。

## 3. 复现

运行参数在 `scripts/analysis/` 对应 Bash 入口顶部配置；本地路径仅放入私有 `.env`。运行使用锁定环境 `uv run --frozen`。重复实验创建新目录；恢复已有缓存时检查数据、模型与处理配置。不得根据结果删除失败样本，不能以新增诊断评分覆盖原严格评分。
