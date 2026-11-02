# 因果掩码与线性注意力：实验设计

本文为实验 3.14 的设计，属于[位置信息与坐标读取](plan.md)研究，结果见 [results_2026-10-21_causal_vs_linear.md](results_2026-10-21_causal_vs_linear.md)。比较对象为 Qwen3.5-9B（24 个 Gated DeltaNet 层与 8 个 full attention 层，均为 causal）与 Qwen3-VL-8B-Instruct（36 个 full attention 层，causal，带 DeepStack）。两个模型均使用 bfloat16、SDPA 与 greedy decoding，thinking off。

## 1. 动机与问题

实验 3.10 与 3.11 在 Qwen3.5 的网格图像上得到一个稳定的不对称：横坐标（列）的信息位于目标 token 上，纵坐标（行）的信息位于目标所在 token 行中 raster 顺序之后的 token 上（[3.11 第 3.3 节](results_2026-10-20_coordinate_flow.md)）；答案位置读纵坐标时 attention 铺在目标行上并偏向行首 token。3.11 把这一现象解释为线性注意力按 raster 顺序沿行向后传播目标信息。这一解释混合了两个因素：

1. **causal 顺序。** decoder 中每个 token 只能接收序列中更早的 token 的信息，这一约束对 full attention 与线性注意力同样成立；
2. **线性注意力的递推。** Gated DeltaNet 的状态按序列距离衰减、并可在行首被清空，信息在序列中“流动”的方式与 softmax attention 不同。

本实验回答以下问题：

1. **Qwen3-VL 中是否存在同样的不对称（Q1）。** Qwen3-VL 没有线性注意力，只有 causal mask；用与 3.10、3.11 相同的测量检验它，并比较 DeepStack 开与关。
2. **理论（Q2）。** 两个因素各自能产生什么、不能产生什么；线性注意力是否依赖 causal mask；双向线性注意力能否工作；每个因素单独能否产生行列读出差异。
3. **分离两个因素的实验（Q3）。** (a) Qwen3-VL 与 Qwen3.5 的对照；(b) 在 Qwen3.5 内只编辑 DeltaNet（冻结状态、不衰减），保留 causal full attention；(c) 在合成 raster 网格上训练的小模型，比较 causal 与双向、softmax 与线性注意力。
4. **32 pixels 网格（Q4）。** 3.10 未解释单元 32 pixels 网格中的“从空白重新计数”，以及读出阶段的比例高估；先分析 32 pixels 的特殊性，再以 full attention 的 knockout 与运行计数 probe 检验。

## 2. 理论分析

### 2.1 符号与设定

图像的 merged token 网格为 $H \times W$，token 按 raster 顺序编号，位于第 $i$ 行第 $j$ 列的 token 的序号为

$$
n(i, j) = iW + j, \qquad 0 \le i < H, \; 0 \le j < W .
$$

目标 token 记为 $\tau = n(i_0, j_0)$（目标覆盖多个 token 时取第一个）。第 $\ell$ 层输出中 token $n$ 的 hidden state 记为 $h_n^{\ell} \in \mathbb{R}^{D}$，$D$ 为 hidden size。相对目标定义四个 token 集合：

- **行后** $R^{+} = \{ n(i_0, j) : j > j_0 \}$，目标行中 raster 顺序之后的 token；
- **行前** $R^{-} = \{ n(i_0, j) : j < j_0 \}$；
- **列下** $C^{+} = \{ n(i, j_0) : i > i_0 \}$，目标列中目标下方的 token；
- **上方** $A = \{ n(i, j) : i < i_0 \}$。

“载体”指这样的 token 集合：把它在答案位置被读取的 key 与 value 换成目标移动后的版本，答案随之移动（3.11 的图像 KV 子集 patching）。

### 2.2 只有 causal mask：可达性约束，没有行列偏好

**causal mask 只规定信息能否到达，不规定信息如何到达。** 在 causal full attention 中，第 $\ell$ 层 token $n$ 的输出为

$$
h_n^{\ell} = h_n^{\ell-1} + \sum_{m \le n} a_{nm}^{\ell}\, W_O v_m^{\ell-1}, \qquad a_{nm}^{\ell} = \frac{\exp\big(q_n^{\top} R_{n-m}\, k_m / \sqrt{d}\big)}{\sum_{m' \le n} \exp\big(q_n^{\top} R_{n-m'}\, k_{m'} / \sqrt{d}\big)} ,
$$

其中 $q_n, k_m \in \mathbb{R}^{d}$ 为单个 head 的 query 与 key，$v_m$ 为 value，$W_O$ 为输出投影，$R_{n-m} \in \mathbb{R}^{d \times d}$ 为 MRoPE 的相对旋转（对图像 token 由行差与列差决定），$d$ 为 head 维度。由此得到三条结论：

1. **目标之前的 token 不可能携带目标信息。** 对所有 $m < \tau$，$h_m^{\ell}$ 与目标处的输入无关，因此 $A$ 与 $R^{-}$ 中位于目标首行的部分在 decoder 拼接对照下的恢复比例必须精确为 0。这是 causal mask 唯一的硬性后果，它产生“前后”不对称。
2. **行后与列下同样可达。** $R^{+}$ 与 $C^{+}$ 都位于 $\tau$ 之后，二者的 token 都可以直接 attend 到目标。attention 权重由内容与相对位置 $R_{n-m}$ 决定，与序列距离 $n - m$ 无关：$C^{+}$ 中的 token 与目标相隔 $W, 2W, \dots$ 个序列位置，这对 softmax attention 没有额外代价。MRoPE 对行差与列差的编码方式相同（交错分配频率），因而没有偏向同一行的结构性理由。
3. **行列两个序号都只需要目标之前的信息。** 列序号需要目标行中左侧的实例数，行序号需要目标上方的实例行数，二者都位于 $\tau$ 之前，目标 token 自身原则上可以同时计算两者。

因此，**仅凭 causal mask 只能预测“载体在目标之后”，不能预测“纵坐标在行后、横坐标在目标”**。若某个只有 causal full attention 的模型出现这种行列不对称，它来自训练得到的解，而不是结构约束；结构上同样允许对称的解（纵坐标在 $R^{+}$，横坐标在 $C^{+}$），或两者都留在目标上。

### 2.3 Gated DeltaNet：按序列距离衰减、可清空的累加器

Gated DeltaNet 的单个 head 在 token $n$ 处更新状态 $S_n \in \mathbb{R}^{d_k \times d_v}$：

$$
S_n = \alpha_n \big( I - \beta_n k_n k_n^{\top} \big) S_{n-1} + \beta_n k_n v_n^{\top}, \qquad o_n = S_n^{\top} q_n ,
$$

其中 $\alpha_n \in (0, 1)$ 为衰减，$\beta_n \in (0, 1)$ 为写入强度，二者由 token $n$ 的输入计算；$k_n, q_n \in \mathbb{R}^{d_k}$ 为单位向量，$v_n \in \mathbb{R}^{d_v}$，$o_n$ 为输出。忽略 delta 修正项 $-\beta_n k_n k_n^{\top}$（它只在 key 相近时擦除旧写入），目标在 $\tau$ 处写入的内容到达之后的 token $n$ 时的权重为

$$
w_{n,\tau} = \beta_{\tau} \prod_{u = \tau + 1}^{n} \alpha_u , \qquad n > \tau .
$$

**这一权重只依赖序列路径上的门控，与 $n$ 的内容无关。** 由此得到与 2.2 节不同的三条性质：

1. **递推本身就是 causal 的。** $S_n$ 只由 $S_{n-1}$ 与 token $n$ 计算，扫描没有访问未来的途径；这里不存在可以去掉的 mask，因果性是扫描方向的属性。分块并行训练时，衰减矩阵 $\exp(G_n - G_m)$（$G_n = \sum_{u \le n} \log \alpha_u$）只在 $m \le n$ 处有定义，它与 causal mask 等价。
2. **目标行被自动广播，目标列不被广播。** 对 $n \in R^{+}$，路径长度小于 $W$ 且不经过行首；对 $n \in C^{+}$，路径长度至少为 $W$ 且至少经过一次行首。3.8 测得 132 个 head 在 token 行首把 $\alpha$ 降到接近 0，于是

$$
w_{n,\tau} \approx \beta_{\tau}\, \bar\alpha^{\,n - \tau} \;\; (n \in R^{+}), \qquad w_{n,\tau} \approx 0 \;\; (n \in C^{+} \cup \text{目标行之后的各行}),
$$

其中 $\bar\alpha$ 为行内的典型衰减。目标写入的内容（例如“本行含有目标”）因此成为行后 token 共享的属性，并随行内距离几何衰减；目标列下方的 token 得不到它。
3. **行计数是跨行共享的变量。** 长记忆 head（$\alpha \approx 1$）在每个新图形行处写入一次，3.10 显示它们承担行计数的累积；同一行所有 token 读出的行计数相同。行后 token 同时拥有“本行含有目标”与“本行的行号”，纵坐标因而可以从行后读出；列计数在行内累加、在行首清零，每个 token 的列计数各不相同，横坐标只能从目标本身（及紧邻的 token）读出。

因此，**Gated DeltaNet 在 raster 顺序下结构性地产生行列不对称：行后承载纵坐标，列下不承载任何目标信息**。这一不对称依赖递推的序列距离依赖与行首复位，与 causal mask 无关。

### 2.4 双向线性注意力

**线性注意力不依赖 causal mask 才能工作。** 不带门控的线性注意力的非因果形式 $O = \phi(Q)\big(\phi(K)^{\top} V\big)$ 本来就是双向的，因果版本只是把求和限制为前缀和（Katharopoulos 等，arXiv:2006.16236）。带数据依赖衰减的递推（Mamba、Gated DeltaNet）按定义有方向，双向版本用两次扫描实现：一次正向、一次反向，输出相加或拼接。视觉骨干中这类双向或多方向扫描已经普遍使用并且有效，例如 Vision Mamba 的双向扫描（arXiv:2401.09417）、VMamba 的四方向交叉扫描（arXiv:2401.10166）与 Hydra 的准可分双向混合器（arXiv:2407.09941）；Gated DeltaNet 本身见 arXiv:2412.06464。

在 raster 顺序下，双向 Gated DeltaNet 的目标写入会同时到达行后与行前（$R^{+} \cup R^{-}$），仍然到不了 $C^{+}$；于是“前后”不对称消失，行列不对称保留。若再加入按列展开的扫描（VMamba 的列优先方向），目标列也会被广播，行列不对称一并消失。对自回归 VLM 而言，答案 token 必须 causal；图像 token 的 prefill 在原理上可以双向（PaliGemma 对图像与前缀使用双向 attention，arXiv:2407.07726），这属于训练与推理方式的选择，与线性注意力能否工作无关。

### 2.5 对四个问题的回答（理论层面）

| 问题 | 理论回答 |
| --- | --- |
| 根源是线性注意力还是 raster causal 顺序 | 两者分工不同。causal 顺序（mask 或扫描方向）决定载体只能位于目标之后；raster 顺序下的递推（序列距离衰减与行首复位）决定载体是目标行而非目标列。只有 causal full attention 时，行列不对称不是结构性的 |
| 线性注意力是否依赖 causal mask | 不依赖。递推形式没有 mask，因果性是扫描方向的内在属性；分块并行形式中的下三角衰减矩阵与 causal mask 等价 |
| 没有 causal mask 能否工作 | 能。非因果线性注意力与双向、多方向扫描都是成熟做法；双向扫描去掉“前后”不对称，多方向扫描去掉行列不对称 |
| 每个因素单独能否产生行列读出差异 | causal full attention 单独只能产生“载体在目标之后”，行列差异需要训练得到的解；raster 递推单独（双向扫描）即可产生行列差异，但载体分布在目标两侧 |

### 2.6 可区分两种解释的预测

| 编号 | 预测 | 若成立，支持 | 检验 |
| --- | --- | --- | --- |
| P1 | 两个模型在拼接对照下“上方”恢复精确为 0 | causal 顺序（两模型共有） | E2 |
| P2 | Qwen3-VL 的纵坐标载体不集中在行后：留在目标上，或行后与列下对称 | 不对称来自 DeltaNet | E2 |
| P3 | Qwen3-VL 在纵坐标 query 处不出现行首（第 0 列）attention 偏好 | 行首读取依赖 DeltaNet 的行首写入 | E1 |
| P4 | Qwen3.5 冻结图像 token 上的 DeltaNet 后，纵坐标的行后份额下降、目标份额上升 | 行后载体由 DeltaNet 写入 | E3 |
| P5 | Qwen3.5 只取消衰减后，纵坐标载体扩展到目标下方各行 | 行局部性来自衰减与复位 | E3 |
| P6 | 小模型中只有含 causal 线性注意力的结构出现“纵坐标在行后”；causal softmax 不出现；双向线性注意力使行前与行后对称 | 2.2–2.4 节的全部区分 | E4 |

若 P2、P3 不成立（Qwen3-VL 出现同样的不对称），则不对称来自 raster causal 顺序下训练得到的读出策略，线性注意力只是 Qwen3.5 中的实现方式。

### 2.7 32 pixels 网格的特殊性

3.10 中单元 32 pixels 的列方向“从空白重新计数”（H32 约 24%）不受 DeltaNet 门控与视觉内容编辑影响。32 pixels 有四点特殊：

1. **一个实例恰为一个 merged token。** ViT 的 2×2 merger 把 32×32 pixels 合成一个 token，实例之间没有空白 token，实例内部也没有多个 token；密集行中“行首以来的实例数”与 token 列号相差一个常数，计数与位置混淆。3.10 的 probe 在 H32 上的槽位偏移 $a = 0.99$（H64 为 0.42）与此一致。
2. **空白只有一个 token。** 3.10 在 64 pixels 网格中确认的复位依赖跨越两个空白 token 的遗忘；32 pixels 时空白 token 两侧的 token 在宽度为 4 的 causal 短卷积（每个 DeltaNet 层在 $q, k, v$ 上都有）中同时可见，局部边界可以不经状态复位、由卷积直接检测。
3. **ViT 位置编码的尺度。** 一个实例对应一个 token 步长，相邻实例的位置编码差异最小，位置码本身可以充当计数。
4. **计数方式可能切换。** 若每个 token 都是一个实例，“局部计数”可以由 full attention 在若干相邻 token 上完成，不需要 DeltaNet 的累积。

据此的检验：在 L4、L8、L12、L16 各自以及四层同时去掉图像 token 之间的 full attention（图像 query 只能 attend 自身与文本），并以“同行左侧 3 个 token”的局部 knockout 作对照；若重新计数来自 full attention 的局部计数，它应在某个 knockout 下消失，并且运行计数 probe 的重新计数权重 $b$ 随之下降。若都不变，则重新计数来自 DeltaNet 的局部通路（短卷积）或读出阶段。Qwen3-VL 在同一批刺激上的行为提供另一条证据：若 Qwen3-VL 也在空白处重新计数，重新计数不依赖 DeltaNet。

## 3. 实验

### 3.1 刺激与模型

- **坐标与行列刺激。** 3.11 的 cases（[3.11 第 2.2 节](results_2026-10-20_coordinate_flow.md)）：10×10 网格（单元 64 pixels，20×20 token）的行列格式（行先、列先两种顺序）与 point 格式，单目标红色圆的 point 格式，RefCOCO 的 point 格式。
- **序号刺激。** 1.5 的槽位网格 H32、H64、V32、V64（每个 320 张，含 base、shift、insL、delL、delR）。
- **模型模式。** Qwen3-VL：DeepStack 开（`ds_on`）与关（`ds_off`，不注入 DeepStack 特征）；Qwen3.5：无编辑、冻结（图像 token 上全部 24 个 DeltaNet 层的 $\alpha = 1, \beta = 0$）、不衰减（$\alpha = 1$）、第 9–16 层冻结（只在 3.8 确定的行计数层上令 $\alpha = 1, \beta = 0$）、目标 token 不写入（只在目标 token 上令全部 DeltaNet 层的 $\beta = 0$，检验行后载体是否需要目标自身的状态写入）。

### 3.2 子实验

| 编号 | 内容 | 模型 / 模式 | 规模 |
| --- | --- | --- | --- |
| E1 | 答案位置 attention 剖面：每层每个 head 在目标、目标行、目标列、行前、行后、第 0 列、下方、上方与文本上的质量 | Qwen3-VL 两种模式；Qwen3.5 五种模式 | 每条件 16 个回答 |
| E2 | 图像 KV 子集 patching（真实移动与拼接对照），组定义同 3.11；全部 full attention 层同时替换，另按层组替换 | 同 E1 | 每条件 16 个回答 |
| E3 | Qwen3.5 的 DeltaNet 编辑下重复 E1、E2，并记录答案正确率 | Qwen3.5 冻结、不衰减、第 9–16 层冻结、目标 token 不写入 | 10×10 行列、10×10 point、圆 point |
| E4 | 小模型：5 种混合器 × 2 种布局 × 8 个种子，3.11 式拼接与读出 KV patching | CPU | 每个模型 150 对 |
| E5 | 1.5 槽位网格的行为与运行计数 probe | Qwen3-VL 两种模式 | 1280 张 × 2 |
| E6 | 32 pixels：full attention knockout 下的重新计数与 probe | Qwen3.5 | H32、H64 的 base、insL、delL，各 192 张 × 7 种条件 |

E2 的恢复比例沿用 3.11 的定义：

$$
r_e = \frac{\mathrm{LD}_e - \mathrm{LD}_A}{\mathrm{LD}_B - \mathrm{LD}_A}, \qquad \mathrm{LD} = z_{d_B} - z_{d_A},
$$

其中 $z \in \mathbb{R}^{V}$ 为预测坐标首位数字的位置上的 logits（$V$ 为词表大小），$d_A$ 为 run A 的首位数字，$d_B$ 为 run B 在同一位置 logit 最大的数字，$\mathrm{LD}_A$、$\mathrm{LD}_B$、$\mathrm{LD}_e$ 分别为 run A、run B 与干预后 run A 的 logit 差。排除 $d_A = d_B$ 或 $\lvert \mathrm{LD}_B - \mathrm{LD}_A \rvert < 2$ 的数值。Qwen3-VL 的拼接对照同时拼接 DeepStack 特征（变化 token 集合取 run B，其余取 run A）；Qwen3.5 的 DeltaNet 编辑同时作用于 run A、run B 与全部干预行。

### 3.3 小模型（E4）

每个样本为 $h_c \times w_c$ 个单元的网格，每个单元覆盖 $k \times k$ 个 token；每条单元行与单元列以 0.25 的概率为空白线，两条图形线交叉处的单元含图形，其中一个为目标。token 输入为内容 embedding（空白、图形、目标）与可学习的绝对行、列 embedding 之和（对应 ViT 的位置码）。三层混合器之后，一个可学习的 ROW 或 COLUMN query 以 softmax attention 读取全部图像 token 的最终状态并分类序号。五种混合器为 causal softmax、双向 softmax、causal 门控线性注意力（$S_n = \alpha_n S_{n-1} + \beta_n v_n k_n^{\top}$，含宽度 4 的 causal 短卷积）、双向门控线性注意力（正反两次扫描）与混合结构（两层 causal 线性注意力加一层 causal softmax，对应 Qwen3.5 的 3:1 比例）。布局为 6×8 单元、$k = 1$（对应 32 pixels）与 4×5 单元、$k = 2$（对应 64 pixels）。载体测量与 E2 相同：run B 把目标移到下一个图形单元（行 query 向下、列 query 向右），拼接输入只在变化 token 上取 run B，再把读出层某个 token 组的 key 与 value 换成 run B 的值。

### 3.4 统计

比例给出 Wilson 95% 区间；恢复比例给出均值与按回答的 bootstrap 95% 区间；成对比较用精确 McNemar 检验。失败与截断的回答计入分母。

## 4. 与已有工作的关系

3.10 与 3.11 只在 Qwen3.5 中描述了行列载体的不对称，并把它归于线性注意力。本实验把“causal 顺序”与“递推结构”分开，给出每个因素单独的结构性后果，并用第二个架构（Qwen3-VL）、架构内编辑（DeltaNet 冻结）与受控小模型三条证据检验。视觉 Mamba 类工作（Vim、VMamba）从效率与精度角度引入双向与多方向扫描，没有分析扫描方向对位置信息存放位置的影响；本实验以载体 patching 直接测量这一影响。

## 5. 停止条件与限制

- 小模型只用于检验结构性预测，不推断大模型的具体参数。
- 冻结 DeltaNet 会大幅改变模型的计算，若答案大多无效，只报告仍满足条件的数值并标注样本量。
- Qwen3-VL 与 Qwen3.5 在训练数据与规模上也不同，模型对照只能区分“结构上必然”与“训练得到”，不能单独归因到某一层。
