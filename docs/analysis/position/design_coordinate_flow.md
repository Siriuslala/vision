# 实验 3.11：坐标读出的信息流

本实验属于[位置信息与坐标读取](plan.md)研究，模型为 Qwen3.5-9B（bfloat16、SDPA、greedy decoding、thinking off，全部问题为只输出答案的 direct-answer 格式）。上游结果为[坐标读取路径](results_2026-10-14_coordinate_readout.md)（实验 3.5）、[生成阶段的 patching](../representation/results_2026-10-15_patching_extension.md)（实验 2.5b/2.5c）、[“看得准但答错”的检验](../representation/results_2026-10-16_sees_right.md)（实验 2.5e）、[坐标数值的数位输出机制](results_2026-10-16_coordinate_value_r2.md)（实验 3.6 第二轮）与 [ViT 位置与布局表征第二轮](results_2026-10-16_vit_layout_r2.md)（实验 3.7 第二轮）。

## 1. 研究问题与动机

已有实验分别给出了坐标读出的若干片段：答案位置在 full attention 层 20 从图像读取坐标（2.5b、3.5）；层 20 的单个 head L20 H11 的输出 patching 权重约为 0.48（3.5）；L20 H11 在横向 query 处的 attention 峰值落在目标上，在纵向 query 处落在目标所在行（2.5e）；只修改目标 token 的 value 几乎不改变坐标，替换全部图像 token 的 key 与 value 则完全改变坐标（2.5e）；层 24 之后坐标集中于文本位置，首位数字由 MLP 28–30 写入（3.6）。这些片段之间缺少一条从图像到数字的连续叙述。本实验以少数决定性的因果实验把它们连接起来，回答以下六个问题：

1. **L20 H11 是否特殊。** 坐标读取由单个 head 完成，还是由一组 head 共同完成；x 与 y 是否由不同的 head 读取；行与列的 attention 模式差异是否普遍，其来源是什么。
2. **attention 解码的答案与模型答案。** 由 attention 峰值直接解码出的坐标，其准确率与误差相对模型自己写出的答案如何，按模型答对与答错分别比较。
3. **位置信息在 visual token 上的分布。** 目标位置的信息存放在哪些 visual token 的 key 与 value 中：目标本身、同行、同列、目标之后的行、远处背景；decoder 是否偏向从目标某一侧（光栅顺序之后）读取，这与 causal mask 的方向有何关系。
4. **输出顺序。** “先输出的坐标读取目标、后输出的坐标复用已读内容”这一假设是否成立；以改变输出顺序的提问（[x, y] 与 [y, x]、行列与列行、bbox 两种顺序）检验，并给出每个输出数值的“层 × token 组”因果归因。
5. **全过程。** 调和“图像 attention 由浅到深递减”（2.5b）与“层 24 之后坐标集中于文本位置”（3.6），给出每个层段的 attention 去向、坐标线性可读的位置与因果写入坐标的组件，并回答答案位置对图像 KV 的读取是否发生在整体 attention 已转向文本之后。
6. **位置场质量与正确性。** 从关键 site 的表征重建每个 visual token 的解码位置（位置场），检验位置场的局部失序、畸变与低分辨率能否预测模型的失败（列号多计 1、压缩、500 吸附）。

## 2. 总体设计

### 2.1 配对与恢复比例

全部因果测量采用同一种配对设计。

1. **run A 与 run B。** run A 为原始输入。合成图像的 run B 把目标重新画在沿横向或纵向移动后的位置（单目标圆移动 4 个 merged token，即 128 pixels；10×10 网格移动 2 个单元，即 4 个 token）；RefCOCO 的 run B 采用实验 3.5 的内容交换，把目标所在的 merged embedding 与等大的平移区域交换，position IDs 与槽位不动。
2. **decoder 内部的拼接对照（splice）。** ViT 的 attention 是全局的，真实移动目标后全部 visual embedding 都可能改变。为分离 decoder 内部的信息流，另构造拼接的 run B：全部 visual embedding 取自 run A，只有变化 token 集合 $\mathcal{D}$（目标在 run A 与 run B 中的 token 之并）取自真实移动的 run B。此时位于 $\mathcal{D}$ 第一个 token 之前的 visual token 在两次运行中的 key 与 value 完全相同（causal decoder），$\mathcal{D}$ 以外 token 上的任何差异都只能由 decoder 在光栅顺序上向后传递产生。RefCOCO 的内容交换天然属于这一类。
3. **teacher forcing。** run A 与 run B 都以 run A 自己的 greedy 答案 teacher forcing，二者共享全部文本 token。

记坐标值 $k$ 的首位数字在答案中的位置为 $i_k$，其 query 位置（next-token prediction 为该数字的输入位置）为 $m_k$。run A 在 $m_k$ 处的首位数字为 $d_A$，run B 在同一位置 logit 最大的数字为 $d_B$，定义

$$
\mathrm{LD} = z_{d_B} - z_{d_A},
$$

其中 $z \in \mathbb{R}^{V}$ 为 $m_k$ 处的 logits，$V$ 为词表大小。对 run A 施加某种干预 $e$ 后，恢复比例为

$$
r_e = \frac{\mathrm{LD}_e - \mathrm{LD}_A}{\mathrm{LD}_B - \mathrm{LD}_A}.
$$

$r_e = 1$ 表示被干预的组件携带了 run B 与 run A 在该数字上的全部差异，$r_e = 0$ 表示没有作用。只统计沿位移轴的坐标值（横向位移对应 x、x_min、x_max 与列号，纵向位移对应 y 与行号），并排除 $d_A = d_B$ 或 $\lvert \mathrm{LD}_B - \mathrm{LD}_A \rvert < 2$ 的数值。选用首位数字有两个原因：实验 3.6 表明首位数字是唯一锐利地依赖图像的数位，十位接近均匀；首位数字的 logit 差是连续量，同一次前向即可测量全部干预，不需要逐条件重新生成。

### 2.2 子实验

| 子实验 | 问题 | 干预或测量 | 样本 |
| --- | --- | --- | --- |
| E1 head 级 patching | Q1 | 在答案位置（prompt 最后一个 token 起）把 run A 的单个 head 输出（gate 之后、output projection 之前）替换为 run B 的值，128 个 head 逐一替换；另做 8 个整层、全部 128 个 head、全部 head 去掉一个（层 16、20、24 的 48 个 head） | 单目标圆 point 48 张、10×10 网格行列与 point 各 40 张、RefCOCO point 48 张，每张横纵两个位移 |
| E2 attention 的行列结构 | Q1 | 2.5e 已记录的 attention：每个 head 在横向与纵向 query 处落在目标、目标所在 token 行（左、右）、目标所在 token 列（上、下）的质量，以及沿目标行的逐列分布 | 2.5e 的 5 个条件，每条件最多 240 个回答 |
| E3 attention 解码 | Q2 | 以 L20 H11、层 20 与层 16 的 head 平均、交叉选择的最佳 head 的 attention 峰值（及峰值 3×3 邻域的加权中心）解码坐标，与模型答案配对比较 | 2.5e 的 14 个直接作答条件，2054 个回答 |
| E4 图像 KV 子集 patching | Q3 | 在 run A 的 prompt cache 中把某个 visual token 子集在 8 个 full attention 层的 key 与 value 换成 run B 的值，再 teacher forcing 答案 | E1 的四个条件；真实移动与拼接对照两类配对 |
| E5 输出顺序 | Q4 | E4 的主要 token 组与“层 × token 组”归因，按输出的每个数值分别统计；同时记录每个答案位置的 attention 在目标、目标行、目标列、其他图像与文本上的质量 | 单目标圆 5 种格式、10×10 网格 4 种格式、RefCOCO 2 种格式 |
| E6 全过程 | Q5 | 以 2.5e 的答案 teacher forcing，记录 33 个 site（decoder 输入与 32 层输出）在 8 个位置组的 residual，以及图像之后每个 query 的图像 attention 占比；以 ridge 估计坐标在各位置组、各层的线性可读性 | 单目标圆 240、10×10 point 160、RefCOCO 298 |
| E7 位置场 | Q6 | 以图像奇偶二折训练从 visual token 状态到其自身位置的 ridge 解码器（site 0、4、8、12、16、20），在留出折上解码每张图的位置场，计算目标附近的失序比例、局部斜率与目标 token 的解码偏差，与模型对错比较 | 20×20 行列 160、10×10 point 160、单目标圆 240 |

E4 的 token 组相对变化集合 $\mathcal{D}$ 定义（以 $\mathcal{D}$ 的外接矩形划分）：

| token 组 | 定义 |
| --- | --- |
| obj | $\mathcal{D}$ 本身 |
| inbox | 外接矩形内不属于 $\mathcal{D}$ 的 token |
| row_left / row_right | 外接矩形所在 token 行中位于其左侧 / 右侧的 token |
| below_cols / below_left / below_right | 外接矩形下方各行中位于其列范围内 / 左侧 / 右侧的 token |
| above | 外接矩形上方的全部行；拼接对照下其 key 与 value 在两次运行中相同，是精确的零对照 |
| ring1、ring2、ring3_4、ring5p | 与 $\mathcal{D}$ 的 Chebyshev token 距离为 1、2、3–4、不小于 5 的 token |
| row$r$、col$c$ | 第 $r$ 个 token 行、第 $c$ 个 token 列的全部 token |
| all_but_obj、all_img、text | $\mathcal{D}$ 以外的全部 visual token、全部 visual token、图像之后的 prompt 文本 |

### 2.3 输出顺序的提问

| 格式 | 要求的输出 |
| --- | --- |
| point | `[{"point_2d": [x, y]}]`（默认格式） |
| point_xyk / point_yxk | `{"x": x, "y": y}` / `{"y": y, "x": x}`，并注明先后（“x first, then y” / “y first, then x”） |
| bbox / bbox_yxk | `[{"bbox_2d": [x_min, y_min, x_max, y_max]}]` / `{"y_min", "x_min", "y_max", "x_max"}` |
| rowcol / colrow | `{"row": r, "column": c}` / `{"column": c, "row": r}` |

可行性检查中，单目标圆与 RefCOCO 的 `{"y": y, "x": x}` 在不注明先后时有一部分回答改用 bbox 列表，注明先后后 8/8 遵从；正式运行中全部 496 个回答都给出了所要求的键。

## 3. 判断标准

1. **Q1。** 若单个 head 的恢复比例占全部 head 恢复比例的大部分，且从全部 head 中去掉它时恢复比例大幅下降，则该 head 在功能上特殊；若单个 head 的恢复比例之和远小于全部 head，则读取以非加性的组合完成。
2. **Q3。** 在拼接对照下，若 row_right、below 等光栅顺序之后的 token 组有显著恢复而 above 为零，则 decoder 把目标位置沿光栅顺序向后写入后续 token；真实移动与拼接对照之差给出 ViT 全局上下文的贡献。
3. **Q4。** 若“先输出的坐标读目标”成立，交换输出顺序后读取目标的应是新的首个坐标；若横向坐标无论先后都依赖目标、纵向坐标无论先后都不依赖目标，则分工由坐标轴决定，与输出顺序无关。
4. **Q6。** 只有当位置场指标区分答对与答错的 AUROC 明显高于 0.5（例如不低于 0.65）时才作为失败的解释，否则报告为阴性结果。

## 4. 与已有工作的关系

本实验沿用实验 3.5 的输出 patching、实验 2.5e 的 KV cache 编辑与实验 3.7 第二轮的 ViT 上下文分析，新的部分是：以 teacher-forced 首位数字 logit 差在一次前向中测量数百个干预，使 head 级与 token 组级的完整扫描成为可能；以拼接对照把 ViT 全局上下文与 decoder 光栅顺序传递分开；以输出顺序变换区分“轴”与“顺序”两种解释。电路发现文献中的 path patching（Wang et al., 2022, Interpretability in the Wild）与 activation patching 的度量选择（Zhang and Nanda, 2023）提供了方法依据；本实验的 logit 差度量与其推荐的 logit difference 一致。

## 5. 实现

- 代码：`analysis_tools/position/coordinate_flow.py`（GPU 阶段）、`coordinate_flow_analysis.py`（CPU 分析）、`coordinate_flow_figures.py`（作图）；入口 `scripts/analysis/position/coordinate_flow.sh`；测试 `tests/analysis/position/test_coordinate_flow.py`。
- 产物：`$WORK_DIR/analysis/position/2026-10-20_coordinate_flow/`。
- 数值检查：带 cache 的分段前向与完整前向的 logits 最大差为 0.72（bfloat16），argmax 全部一致；拼接对照下 above 组的恢复比例应为 0；全部 visual token 的 KV patching 与全部 head 的输出 patching 应接近 1。
