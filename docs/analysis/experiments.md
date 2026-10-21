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
| 位置信息提取 | `analysis/position/2026-10-14_probe_refinement/` | 实验 3.4：3000 张 RefCOCO 与 352 张合成图的 62 个位置特征，全维 ridge、MLP、学习曲线、相对位置与内容对照；大体积特征存放于 DATA_DIR 的 scratch 目录，WORK_DIR 路径为符号链接 |
| 坐标读取 | `analysis/position/2026-10-14_coordinate_readout/` | 实验 3.5：C0 基准、C1 生成时 attention 与梯度归因、C2 与 C3 位置来源交换、C4 head 消融，以及 20×20 与 10×10 网格失败的定向分析；2026-10-15 补充 C4 输出 patching 与冗余结构（`patching/`、`ablation/`）、RefCOCO 自然图像（`refcoco/`）、C5 插行（`insertion/`）与 `results_supp.json` |
| 多层视觉特征注入 | `analysis/deepstack/2026-10-14_end_to_end/` | 实验 4.2：四个模型的 BabyVision 全集、T0–T6 子任务（含修订 T3 问法）、RefCOCO 面积分层定位与推理文本编码 |
| 多层视觉特征注入 | `analysis/deepstack/2026-10-14_end_to_end/diagnostic_scoring/` | 4.2 补充：BabyVision 格式宽松的诊断评分，四个模型的改判题目与逐题核对 |
| 坐标读取 | `analysis/position/2026-10-16_vit_layout_r2/` | 实验 3.7 第二轮：合成至自然图像的刺激阶梯、两模型的 PE 与 RoPE 消融（RefCOCO）、merger 位置子空间的共享与擦除、L20 H11 的 key/value 分解 |
| 坐标读取 | `analysis/position/2026-10-15_vit_layout/` | 实验 3.7：PE 与 RoPE 的消融、缩放与区域互换（Qwen3.5 与 Qwen3-VL），ViT 逐层位置与布局 probe、merger 位置子空间的 LEACE 擦除、局部重读试验与 RefCOCO 对照 |
| 坐标读取 | `analysis/position/2026-10-16_sequence_order_r2/` | 实验 3.8 第二轮：DeltaNet 门控测量与干预、行序数 probe、按距离的 mixer 恢复、ViT 位置与上下文移植、直接作答与先推理后作答对照 |
| 坐标读取 | `analysis/position/2026-10-15_sequence_order/` | 实验 3.8：槽位重排、图形行与单元编辑、mixer 输出恢复、相同 token 序列 probe 与 20×20 列号多计 1 的顺序检验 |
| 坐标读取 | `analysis/position/2026-10-16_coordinate_value_r2/` | 实验 3.6 第二轮：多种答案格式的数位输出、attention/MLP/head 输出的几何、unembedding 子空间分析、L20 H11 的 path patching、MLP 消融与 neuron 调谐 |
| 坐标读取 | `analysis/position/2026-10-16_readout_tool/` | 实验 3.9：20,282 个样本的 hidden state 与 attention 读出器训练与评估、12.5k 次模型自身 point/bbox 生成的配对比较、7,680 次工具上下文下的行列作答与 918 次自由工具调用；特征与图像缓存于 `$DATA_DIR/scratch/` |
| 坐标读取 | `analysis/position/2026-10-20_readout_tool_r2/` | 实验 3.9 第二轮：25,328 个样本（RefCOCO、第一轮合成集与五级真实感合成集）的槽位 attention、内容特征与 hidden state 捕获，含与不含 attention 的读出器比较，1.5 刺激上的工具读数与 200 张网格图像的坐标–序号干预；全部数字汇总于 `tool_r2/results_r2.json` |
| 坐标读取 | `analysis/position/2026-10-20_rowcol_mechanism/` | 实验 3.10：1280 张槽位网格的门控与计数 probe 特征、512 张边界干预及两组各 256 张的后续干预、128 张 8×8 网格的 head 类别门控干预、192 张直接作答与先推理后作答对照、384 张输出顺序交换 |
| 坐标读取 | `analysis/position/2026-10-20_coordinate_flow/` | 实验 3.11：11 种输出格式的 496 次生成，全部 full-attention head 的逐一、逐层与留一 patching，11 种条件的图像 KV 子集替换，2.5e 的 698 个回答上 33 个位置 × 8 个位置组的残差追踪，560 张图像 6 个位置的位置场 |
| 坐标读取 | `analysis/position/2026-10-20_digit_failure/` | 实验 3.12：804 个输入、6909 行强制首位数字的析因记录（前两个数位的数字分布、第 8–32 层残差、逐组件与逐 head 的 direct logit attribution），72 张网格图像的贪心解码，264 个输入的 full-attention 层 attention |
| 坐标读取 | `analysis/position/2026-10-15_coordinate_value/` | 实验 3.6：密集单目标平移序列的逐层 residual、tokenizer 检查、坐标 probe 与几何、logit lens 与直接归因、数字强制、单位置 steering 与跨图像插值 |
| 多层视觉特征注入 | `analysis/deepstack/2026-10-15_ablation_light/` | 实验 4.3（轻量版）：Qwen3-VL-8B/4B 的 DeepStack 消融（Full、Off、Blur、Mean、Off-1/2/3），T1/T2 颜色、T5/T6 定位与 RefCOCO |
| 多层视觉特征注入 | `analysis/deepstack/2026-10-20_todo/` | 实验 4.4：Qwen3-VL-8B/4B 的 Blur（σ = 4、8、16 px）、Blank 与 Blank-flip 注入条件，POPE、TextVQA 与 BabyVision 的非坐标对照，270 张图像上 ViT、注入特征与 decoder 各层的逐 token ridge 读出（位置、高频能量、4 px 结构、粗颜色）；`represent/` 子目录约 6 GB |
| 视觉细节表征 | `analysis/representation/2026-10-16_sees_right/` | 实验 2.5e：2066 次直接作答与先推理后作答生成的 attention、probe 与行为三种意义的“看得准”检验，以及 value 复制与图像 KV 替换的因果检验 |
| 视觉细节表征 | `analysis/representation/2026-10-20_odd_entity/` | 实验 2.6：2496 张图像（6×6 网格 std/swap 配对四元组、3×3 与 10×10 网格、混合模板多数、半区图像与 RefCOCO 物体裁剪），1680 次直接作答与 192 次先推理后作答，ViT、merger 与 decoder 的逐实体状态，184 张图像 × 22 个位置 × 8 类上下文替换 |
| 视觉细节表征 | `analysis/representation/2026-10-16_high_norm/` | 实验 2.2：Qwen3.5-9B 与 Qwen3-VL-8B 的高范数 token 普查、来源干预、attention sink、内容与位置 probe 及任务消融 |
| 视觉细节表征 | `analysis/representation/2026-10-15_patching_extension/` | 实验 2.5b/2.5c：生成阶段 patching（33 个 site、direct/direct1/via 路径分解、单答案 token 替换）与失败条件（修订 T3、10×10/20×20、色差 10/20、1.3 的 T5 失败图像）的替换记录与汇总 |
| 行为诊断 | `analysis/behavior/2026-10-16_ordinal_mechanism/` | 实验 1.5：计数与位置的分离变体、直接作答与先推理后作答、逐层 probe 与控制任务、steering、误差起点模型拟合与行偏差统计 |
| 行为诊断 | `analysis/behavior/2026-10-15_tolerant_parse/` | 1.3、1.3 补充与 1.4 定位答案的格式容错评分：无效答案分类、容错前后的关键表格与配对比较；各运行目录另有 `scores_*_tolerant.jsonl` |
| 行为诊断 | `analysis/behavior/2026-10-15_ordinal_token_grid/` | 实验 1.4：单元尺寸×列数析因、相位偏移与行数条件共 13328 条，读取 head 的逐 token 列 attention 与序号 log 概率差，以及 1.3 各运行的坐标约定诊断列 |
| 行为诊断 | `analysis/behavior/2026-10-14_decomposition_supplement/` | 实验 1.3 补充条件：T3 问法修订、非方形与长轴网格、固定 768 画布、环形布局，共 32064 条 |
| 行为诊断 | `analysis/behavior/2026-10-13_decomposition_pilot/` | 实验 1.3 的开发运行（每单元 3 张图），只用于确定规模与格式，不进入正式评分 |
| 局部重读 | `analysis/reread/2026-10-16_reread_pilot/` | 实验 5.1 pilot：整图、整图放大、像素裁剪重编码与缓存窗口重读（多种 ViT 深度）在细节判别、原图定位、RefCOCO 小目标与 BabyVision 上的比较 |
| 局部重读 | `analysis/reread/2026-10-20_cached_reread/` | 实验 5.2：原生 zoom 工具协议下像素裁剪与缓存重读（末层与第 9 层、1× 与 2× 缓存、position IDs 与逆序变体）在 TextVQA、GQA、MMVP、POPE、BabyVision 与合成网格上的准确率、时间、显存与 KV cache 比较 |
| 行为诊断 | `analysis/behavior/2026-10-16_pe_scaling_t0/` | 实验 1.6：PE ×1/×2/×2.76 在 1.3 网格子集、BabyVision（Visual Tracking 与找不同）与 RefCOCO 上的比较，以及 T0 计数变化数据集（DATA_DIR 的 `stimuli_t0_count_variation_v1/`）的计数评测 |
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
