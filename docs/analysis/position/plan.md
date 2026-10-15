# 位置信息与坐标读取

## 1. 研究问题

本问题研究 Qwen3.5-9B 中 visual token 的二维位置如何被表示，以及模型在生成坐标答案时从哪里读取位置。位置读取是找不同、计数和 grounding 的共同基础：模型即使选中了正确实例，也需要把该实例对应到 point、bbox 或行列序数。本问题依次回答两个子问题：

1. **位置可读性。** 从 ViT、merger 到 decoder 各层的 visual-token hidden state 中，能够读出多少绝对位置与相对位置信息；读出结果如何随深度、probe 容量、训练数据量和图像内容变化。
2. **坐标读取路径。** 模型生成坐标 token 时，位置信息来自 ViT 写入 visual embedding 的位置特征、decoder 的 MRoPE position IDs、序列顺序还是视觉内容本身；这些信息经过哪些层和 attention head 到达输出。

第一个子问题由外部 probe 回答，只说明信息可以被恢复；第二个子问题由 attention 测量与因果干预回答，说明模型实际使用了哪一种信息。两类结论分别陈述，不以 probe 的可读性推断模型的使用方式。

本问题区分三种位置标签：

- **输入槽位位置**：token 在 processor 网格中的行列位置，由 processor 的排列规则唯一确定。
- **语义来源位置**：token 携带的某项视觉内容来自图像的哪个区域。经过 attention 混合后，一个 token 可以携带多个区域的内容，因此语义来源位置可以是一个分布。
- **查询目标位置**：问题所指定的目标在图像中的位置，例如被要求输出 bbox 的物体。

实验 3.1–3.4 测量输入槽位位置，实验 3.5 通过内容交换区分输入槽位位置与语义来源位置，并以查询目标位置作为模型输出的真值。

## 2. 研究历程

本问题的实验按以下顺序展开，每一项实验由前一项的结果引出。

1. **合成图坐标 probe（2026-09-30，已完成）。** 线性代数分析表明，decoder 的 full attention 只通过 RoPE 在 query 与 key 的内积中引入相对位置，linear attention 不含显式位置编码；但这两点不能推出 visual token 缺少位置信息，因为 ViT 已经向 visual embedding 注入 learned absolute position embedding。为检验这一判断，实验在重复几何图形上训练低容量线性 probe。结果显示 visual embedding 与所测 decoder 层均能部分读出槽位坐标，并能推广到未见模板、平移和尺寸；第 7–9 层误差最低，第 32 层误差明显上升。见[合成图实验](results_2026-09-30.md)实验 A。
2. **文本前缀对照（2026-09-30，已完成）。** 合成图 probe 只在一种 prompt 下训练，需要检验读出是否只依赖图像内部位置。在图像之前插入 169 个文本 token 而保持像素与标签不变后，visual embedding 的预测不变，decoder 第 1 层起出现非零预测位移，第 7 层平均位移约为 0–1 坐标下的 0.115。该结果说明 decoder 中的位置读出方向受前置上下文影响，且影响从 linear attention 层开始出现。见同一报告实验 B。
3. **真实图逐层 probe（2026-10-01，已完成）。** 合成图只有 24 张训练图，且 decoder 第 16 与第 32 层之间未采样。实验在 360 张 RefCOCO 图像上覆盖 ViT 全部 27 层与 decoder 全部 32 层。加入 ViT position embedding 后的 MAE 为 25.96（0–1000 坐标），patch content 为 248.72；decoder 误差在第 14 层最低（92.33），随后升至第 32 层的 185.29；最佳层的精确网格准确率仅为 5.59%。见[真实图实验](results_2026-10-01.md)。
4. **位置 probe 完善（已完成，见[结果](results_2026-10-14_probe_refinement.md)）。** 真实图 probe 使用 256 维随机投影和 ridge 回归（514 个参数），训练图像只有 240 张，无法区分“位置信息在深层减少”与“低容量读出无法提取”。此外，MRoPE 只在 attention 内部临时注入相对位置，decoder 的 residual stream 可能以相对形式保留位置。实验 3.4 提高 probe 容量、扩大数据，并加入相对位置 probe。设计见[位置 probe 完善](design_probe_refinement.md)。
5. **坐标读取路径（部分完成，见[结果](results_2026-10-14_coordinate_readout.md)）。** 实验 1.2 中红色目标的行列答案 3/3 正确而中心点 0/3 正确，RefCOCO 的 bbox 8/8 达到 IoU≥0.5，说明坐标输出能力依赖场景与答案格式。probe 不能回答模型在生成坐标时使用哪一种位置来源，实验 3.5 在完整模型生成坐标时测量 attention 并干预各位置来源。设计见[坐标读取路径](design_coordinate_readout.md)。

## 3. 子实验与进度

| 编号 | 子实验 | 设计目的 | 状态与文档 |
| --- | --- | --- | --- |
| 3.1 | 合成图坐标 probe | 检验重复几何图形中单个 visual token 的槽位坐标能否被低容量线性 probe 读出，并测试跨模板、平移和尺寸的推广 | 已完成；[结果](results_2026-09-30.md) |
| 3.2 | 文本前缀对照 | 保持像素与标签不变，在图像前插入文本，用固定 probe 测量预测位移 | 已完成；[结果](results_2026-09-30.md) |
| 3.3 | 真实图逐层 probe | 在 RefCOCO 图像上覆盖 ViT 与 decoder 全部层，测量位置可读性随深度的变化 | 已完成；[结果](results_2026-10-01.md) |
| 3.4 | 位置 probe 完善 | 以全维 ridge、MLP probe、约 3000 张图像和学习曲线确定可读性上限，并区分绝对位置与相对位置的可读性 | 已完成；[设计](design_probe_refinement.md)，[结果](results_2026-10-14_probe_refinement.md) |
| 3.5 | 坐标读取路径 | 测量生成坐标 token 时的 attention，并通过 position IDs、ViT 位置编码、序列顺序与视觉内容的干预确定坐标的实际来源 | 部分完成（C0–C4 与网格失败定向分析；RefCOCO、C4 输出 patching、C5 待开展）；[设计](design_coordinate_readout.md)，[结果](results_2026-10-14_coordinate_readout.md) |

旧计划中的两项待实施内容并入实验 3.5：

- **位置来源干预**（分别控制 ViT 位置编码、decoder position IDs 与 linear attention 的状态）对应 3.5 的 C2。C2 以完整模型的坐标输出作为主要测量对象、以固定 probe 的读出作为辅助指标，因为坐标输出直接反映模型对位置的使用，而 probe 读出只反映信息的可恢复性。
- **语义来源与槽位区分**（交换两处局部内容，分别标注槽位坐标与内容来源坐标）对应 3.5 的 C3。目标与空白区域的内容交换提供了明确可识别的视觉内容和交换边界，同时可以观察模型输出跟随内容还是跟随槽位。

## 4. 与其他研究问题的关系

1. **问题 1（行为诊断）。** 实验 1.3 的 T6 显著目标定位测量红色目标的行列、中心点与 bbox 随网格规模的变化（见[能力分解](../behavior/design_decomposition.md)）。T6 的成功与失败样本是实验 3.5 中行列分析（C5）的输入；若 T6 的中心点随网格规模下降而行列保持正确，3.5 将优先比较两种答案格式的读取路径。
2. **问题 2（视觉细节表征）。** 实验 3.4 与 2.3 属性 probe 共用特征提取流程与数据划分方式（见[视觉细节表征](../representation/plan.md)）。问题 2 中与图像位置相关的成分和高范数 token，可以用 3.4 的位置 probe 方向检验其是否编码槽位位置；3.5 的内容交换与问题 2 的 activation patching 使用相同的视觉 embedding 替换接口。
3. **问题 5（局部重读）。** 局部重读需要为重新输入的窗口 token 分配 position IDs（见[局部重读](../reread/plan.md)）。实验 3.5 的结论决定这一分配方式：若坐标主要由 decoder position IDs 决定，窗口 token 应保留其在原图中的 position IDs；若坐标主要由 visual embedding 中的位置特征决定，缓存特征本身已携带位置，position IDs 的分配对坐标输出的影响较小。

## 5. 实施顺序

1. 实验 3.4 属于总计划的第一阶段，与实验 2.3 共用特征提取流程，可以并行实现。
2. 实验 3.5 属于第二阶段，其中 C0 与 C1 可以在 3.4 完成前开展；C2 需要 C0 提供答对样本，C4 需要 C1 提供候选 head，C5 在 C2 之后开展。
3. 若实验 1.3 显示 T6 显著目标定位是主要瓶颈，则 3.5 提前至第一阶段末开展。
4. 若实验 3.4 显示深层以相对形式保留位置，3.5 的 C2 优先运行 position IDs 的干预；若显示均匀画布上位置仍可读出，C2 同时运行 ViT 位置与序列顺序的干预，以区分无内容条件下的位置来源。

## 6. 实验约定

1. 主模型为 Qwen3.5-9B，bfloat16，thinking off；对照模型为 Qwen3-VL-8B-Instruct。层编号从 1 开始，与已有结果文档一致；decoder 的 full attention 位于第 4、8、12、16、20、24、28、32 层，其余为 Gated DeltaNet linear attention。
2. 训练、验证和测试按整张图像划分，同一图像的不同 referring expressions 和全部 visual tokens 只出现在一个划分中。probe 的输入不含 token index、position IDs 或图像尺寸，网格尺寸只用于构造标签与评价。
3. 记录原图尺寸、实际输入尺寸、processor 网格、位置采样、数据 hash、模型版本、提取层和训练参数；probe 结果同时报告常数基线与打乱训练标签的基线。
4. 连续指标报告以图像为单位的 bootstrap 95% 区间，比例报告 Wilson 区间并给出分母。干预实验保留全部样本，失败与格式错误的回答保留在分母中。
5. 产物保存于 WORK_DIR 的 `analysis/position/<日期>_<名称>/`，文档用图复制到本目录的 figures 目录。
