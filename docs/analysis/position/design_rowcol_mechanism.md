# 行列序号的统一机制：实验设计

本文为实验 3.10 的设计，属于[位置信息与坐标读取](plan.md)研究，结果见 [results_2026-10-20_rowcol_mechanism.md](results_2026-10-20_rowcol_mechanism.md)。模型为 Qwen3.5-9B，bfloat16、SDPA、greedy decoding，默认 thinking off。

## 1. 动机与问题

此前四项实验分别从行为、DeltaNet 门控、attention 与 probe 描述了行列序号的产生方式，但结论之间存在两处没有解释的矛盾。第一，实验 1.5 发现行与列的偏差服从同一条规律（前约 5 个实例精确，此后比例高估随序号增长），而实验 2.5b、2.5e 与 3.8 发现行与列由不同的 token 携带、由不同的 attention 模式读出。第二，实验 1.5 发现列方向的计数在约 17%–30% 的回答中从内部空白之后重新开始，“依赖局部边界”在机制上意味着什么没有被检验。本实验以一个统一的计算模型回答以下问题：

1. **局部边界（Q1）。** 列方向“从空白重新计数”是否对应 DeltaNet 状态在边界处的复位，计数的精度是否随累积步数下降。
2. **行偏差（Q2）。** 行的 +1 与 −1 是否需要与列相同的解释；同一条行为规律如何由不同的机制产生。
3. **架构因素（Q3）。** decoder 序列的 causal 顺序与线性注意力的递推是否损害 grounding，即目标定位与序号读出。
4. **线性注意力机制（Q4）。** 3.8 中未确定的部分（按 head 类别的作用）；不写推理文本时模型能否在 prefill 中完成计数（隐式计数）。
5. **输出顺序（Q5）。** 纵向读出不落在目标 token 上，是否因为另一个轴先被输出。
6. **综合（Q6）。** attention、DeltaNet、计数行为与 probe 的结论能否合成一个关于行列序号读出的解释。

## 2. 计算模型与可检验的预测

### 2.1 符号

图像的 merged token 网格为 $H \times W$，token 按 raster 顺序编号 $t = 1, \dots, N$，$N = HW$。对 Gated DeltaNet 的某一层某一 value head：

- $\alpha_t \in (0, 1)$：位置 $t$ 的衰减系数，标量；
- $\beta_t \in (0, 1)$：位置 $t$ 的写入强度，标量；
- $S_t \in \mathbb{R}^{128 \times 128}$：递推状态；
- $k_t, q_t \in \mathbb{R}^{128}$（单位向量）与 $v_t \in \mathbb{R}^{128}$：key、query 与 value。

递推为

$$
S_t = \alpha_t \big(I - \beta_t k_t k_t^\top\big) S_{t-1} + \beta_t k_t v_t^\top ,
$$

其推导与展开见 [3.8 第二轮第 2 节](results_2026-10-16_sequence_order_r2.md)。对长轴上的序号，记 $n$ 为读者（沿 raster 顺序）到达某个单元之前已经经过的实例线数，$k$ 为该单元的真实计数（包含自身），$k_{\mathrm{gap}}$ 为从最近一条内部空白之后重新开始的计数。

### 2.2 列：行内累加器

设某个 head 在每个“事件” token（新的一个实例开始）以 key $u \in \mathbb{R}^{128}$、value $v^\ast \in \mathbb{R}^{128}$ 写入，其余 token 的 key 与 $u$ 正交。令 $m_t = S_t^\top u \in \mathbb{R}^{128}$ 为状态在 $u$ 方向上存储的 value。若 head 在 token 行的行首令 $\alpha = 0$、行内令 $\alpha \approx 1$，经过 $n$ 个事件后

$$
m^{(n)} = \big(1 - (1 - \beta)^n\big)\, v^\ast .
$$

该式描述一个在行首清零的饱和累加器，$m^{(n)}$ 单调地编码“本行已经经过的实例数”，即列序号。由此有两条推论：

1. **复位位置决定计数起点。** 若某个内部空白之后的第一个图形 token 在 ViT 表示中与行首 token 相似（左侧没有图形），而门控只依赖当前 token 的内容，那么同一批 head 也会在这里降低 $\alpha$，累加器从空白之后重新计数，答案落在 $k_{\mathrm{gap}}$。这是“依赖局部边界”最直接的机制候选（预测 P1）。
2. **精度随累积步数下降。** 若读出带有固定幅度的噪声 $\varepsilon$，由 $m$ 反推的计数误差为

$$
\sigma_n \approx \frac{\lVert \varepsilon \rVert}{\lVert \partial m^{(n)} / \partial n \rVert} = \frac{\lVert \varepsilon \rVert}{-\ln(1 - \beta)\,(1 - \beta)^n\, \lVert v^\ast \rVert},
$$

即误差随 $n$ 几何增长。$\sigma_n$ 低于半个实例时读出可以四舍五入到正确的整数，超过之后开始出现 ±1 错误；这给出“前若干个实例精确、之后近似”的行为，而不需要单独的 subitizing 机制（预测 P2）。

### 2.3 行：跨行链式估计

行号需要跨越 32–224 个 token 的记忆。3.8 的结果支持如下的链式估计：

$$
\hat r_j = \lambda\, (\hat r_{j-1} + \delta_j) + (1 - \lambda)\, \rho_j ,
$$

其中 $\hat r_j$ 为第 $j$ 个单元行的行号估计（标量），$\delta_j \in [0, 1]$ 为该行贡献的增量，$\rho_j$ 为该行 ViT 行位置给出的估计，$\lambda \in [0, 1]$ 为序列上下文的权重。增量 $\delta_j$ 需要一个“新的一行图形开始”的检测器，检测器必须能把当前行与上一行区分开，因而依赖带遗忘的 head；累积 $\hat r_{j-1} + \delta_j$ 依赖 $\alpha \approx 1$ 的长记忆 head（integrator）的写入。由此有三条预测：

1. **空白行不复位。** 空白行没有图形，事件检测器不触发，$\delta_j = 0$，链式估计原样跨过空白，不出现 $k_{\mathrm{gap}}$ 型的重新计数（P3）。
2. **按 head 类别的分工。** 禁止 integrator 写入应使行号少计；令带遗忘的 head 不遗忘（$\alpha = 1$）应使增量变弱、同样少计；令 integrator 不遗忘应几乎无作用，因为它们的 $\alpha$ 本已接近 1（P4）。
3. **误差的累积受 ViT 校准约束。** 若每行增量带有独立噪声，$\lambda < 1$ 时估计的方差有上界 $\sigma^2 / (1 - \lambda^2)$，行方向的精度随行号下降得比列方向慢（P5，弱预测）。

### 2.4 读出与输出顺序

图像 token 在序列中位于问题文本之前（Qwen3.5 的对话模板先放图像、后放文本），在 causal decoder 中，图像 token 的全部表示与问题文本无关。因此：

1. **隐式计数（P6）。** 两种累加器都在 prefill 中、在模型读到问题之前完成；只要求答案的 direct prompt 与允许先推理的 think prompt 共享完全相同的图像 token 表示。对 prefill 中图像 token 的门控干预应以同样的方向影响两种 prompt；若 think 下的推理文本能独立地重新计数，它应部分抵消干预的作用。
2. **输出顺序不改变载体（P7）。** 行号由目标所在单元行的所有 token 共享，列号只在目标附近；答案位置的 full attention 读出列号时看向目标 token，读出行号时看向目标所在行即可。这一差异来自图像 token 的表示，与先输出哪个轴无关；交换输出顺序至多改变读出的细节。

## 3. 子实验

| 编号 | 名称 | 刺激 | 操作 | 检验 |
| --- | --- | --- | --- | --- |
| E1 | 边界门控 | 1.5 的 H32、H64、V32、V64 全部 1280 张图像 | prefill 中记录 24 个 DeltaNet 层每个 head 在每个 visual token 的 $\alpha_t, \beta_t$；比较边界 token（内部空白之后的第一个图形 token）与匹配对照 token（同一布局向后移 3 条实例线、前面没有空白） | P1、P3 |
| E2 | 运行计数 probe | 同上 | 每张图像目标所在的长轴线与另一条线上 42 个单元的 residual 平均（层 4–28）；在“前面没有内部空白”的单元上训练计数 $k$ 的 ridge probe（按 base 5 折），应用于空白之后的单元 | P1、P2、P3、行偏差 |
| E3 | 边界门控干预 | insL 与 delL 图像（每条件 128 张）；base 图像中 $c \ge 6$ 的目标 | 在边界 token 上对 E1 选出的边界复位 head 令 $\alpha = 1$，对照为匹配位置与同数量随机 head；对 integrator 在边界处令 $\alpha = 0$；在 base 图像中目标之前 3 条线处制造人工边界（$\alpha = 0$） | P1 的因果检验 |
| E3b | 空白段门控干预 | H32、H64 的 insL 与 delL | 对空白段全部 token 与边界 token，分别对边界 head、层 9–16 全部 head、全部 DeltaNet 层全部 head 令 $\alpha = 1$，以及全部层令 $\beta = 0$ | E3 阴性时排除 DeltaNet 门控 |
| E4 | head 类别干预 | 3.8 的 8×8 网格 128 张 | 层 9–16 中按 3.8 的 head 类别（integrator、row-reset、cell-row onset、其他）分别令 $\alpha = 1$ 或 $\beta = 0$ | P4 |
| E5 | 隐式计数 | 8×8 网格 128 张；1.5 V64 base 64 张 | direct 与 think（thinking off）在同样的 prefill 门控干预（无、层 9–16 $\alpha = 1$、层 9–16 $\beta = 0$）下作答；8×8 中 8 张另以 thinking on 作答 | P6 |
| E6 | 输出顺序 | 8×8 网格 128 张；1.5 四个条件的 base 各 64 张 | 默认顺序 {"row", "column"} 与交换顺序 {"column", "row"}；teacher forcing 下记录层 16、20 全部 head 在预测每个数值首位数字时对图像的 attention | P7 |

## 4. 指标

1. 比例给出 Wilson 95% 区间，均值给出 bootstrap 95% 区间，同一图像上的两种条件以精确 McNemar 检验比较。
2. E2 中，空白之后单元的 probe 读出 $\hat k$ 拟合为

$$
\hat k - k = a + b\,(k_{\mathrm{gap}} - k),
$$

其中这些单元的槽位序号恰为 $k + 1$，因此截距 $a$ 度量读出向槽位（像素位置）偏移的程度，$b$ 度量读出向“从空白重新计数”偏移的程度：$b = 1$ 表示完全重新计数，$b = 0$ 表示保持运行计数。
3. E2 的精度以“前面没有空白”的单元上交叉验证读出误差的标准差随 $k$ 的变化度量。
4. E6 以 L20 H11（实验 2.5e 的主要读出 head）的 attention 峰值是否落在目标 token 上、是否落在目标所在的 token 行（行查询）或 token 列（列查询）上度量读出位置。

## 5. 运行

入口为 `scripts/analysis/position/rowcol_mechanism.sh`，`COMMAND=run` 运行 GPU 阶段，`COMMAND=heads` 由 E1 选出边界复位 head（E3 需要），`COMMAND=analyze` 生成统计与图。程序为 `analysis_tools/position/rowcol_mechanism.py`（几何、计数、prompt 与解析）、`rowcol_mechanism_run.py`（GPU 阶段）与 `rowcol_mechanism_analysis.py`（统计与作图），测试为 `tests/analysis/position/test_rowcol_mechanism.py`。产物写入 WORK_DIR 下的 `analysis/position/2026-10-20_rowcol_mechanism/`。
