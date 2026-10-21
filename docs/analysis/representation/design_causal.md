# 实验 2.5：Activation patching 与概念擦除

本实验属于[视觉细节表征](plan.md)，为实验 2.2 至 2.4 的读出结果提供因果检验。probe 与特征反演只能说明信息可以从表征中恢复，不能说明模型在生成答案时使用了这些信息。本实验在完整模型的前向计算中直接替换或擦除表征，测量答案的变化，回答模型是否依赖目标实例 token 上的信息、依赖发生在哪些层，以及是否依赖 2.3 识别的属性子空间。

## 1. 研究问题与动机

2.1 表明配对差分可以在 ViT 与 merger 上定位局部改动；2.3 将测量单图中属性的可读性。若属性可读而模型仍答错，需要确定信息在何处中断：是目标实例 token 上的信息没有被问题文本读取，还是读取后没有转化为答案。本实验包含两个子实验：

1. **2.5a activation patching。** 在严格配对的正常图与异常图之间替换特定层、特定 token 组的 hidden states，测量答案 logit 差的恢复比例，确定承载异常信息的 token 与层，以及信息从视觉 token 转移到文本位置的层。
2. **2.5b 失败样本的中断位置。** 在模型答错或判别分数差很小的条件上（大网格、低色差颜色变化、两实例比较成功而完整找不同失败的图像），以 2.5a 的恢复比例与 logit 位移确定异常信息在何处中断：目标实例 token 上是否存在差异，差异是否转移到文本 token，转移之后是否丢失。
3. **2.5c 生成阶段的 patching。** 对答案需要生成的定位任务（红色目标与找不同的行列、中心点答案），在 teacher forcing 的答案序列上替换各层 hidden states，测量生成坐标数字时 visual token 与 prompt 侧传递 token 各自的作用，区分“生成时直接读取图像”与“读取 prompt 中已写入的结果”两条路径。
4. **2.5d 概念擦除。** 在视觉 token 中擦除 2.3 probe 识别的属性线性子空间，测量完整模型准确率的变化，检验模型是否使用该子空间。

**编号。** 2.5b 与 2.5c 由 2.5a 的结果直接导出：2.5a 只覆盖答对的样本与回答首个 token 的 prefill，实验 3.5 又表明坐标数字在生成阶段重新读取目标 token，因此这两项扩展排在概念擦除之前，概念擦除的编号为 2.5d，其设计内容不变。

2.5a 只需要已有的配对图像与 forward hooks，不依赖其他子实验，安排在第一阶段；2.5b 与 2.5c 复用 2.5a 与 3.5 的代码，安排在第一阶段末；2.5d 依赖 2.3 的 probe，安排在第三阶段。

## 2. 任务与数据

### 2.1 任务条件

activation patching 要求模型在 clean 与 corrupted 两种输入上都能给出正确答案，否则答案 logit 差的变化没有明确含义。因此，任务条件取自实验 1.3 中模型准确率较高的条件。若实施时实验 1.3 尚无结果，则采用以下条件：

1. **图像。** [共享合成刺激集](../stimuli.md)中的 6×6 网格、64-pixel 实例单元、四个开发模板，变化类型取显著幅度：纯红色颜色变化与 16-pixel 缺口。
2. **问题。** 实验 1.3 的 T3 指定实例判断，两种指定方式分别运行：“Is the figure in row r, column c exactly identical to the other figures?”，以及在实例单元边缘绘制蓝色方框后询问“Is the figure inside the blue box exactly identical to the other figures?”。缺口条件另设属性问题“Does the figure in the blue box have a gap?”。回答要求只输出 Yes 或 No。

### 2.2 配对构造

每个配对由两张仅在一个实例上不同的图像组成：异常图中指定实例经过变化，正常图中该实例为原模板，其余像素完全相同；蓝色方框在两张图中位于同一位置。同一问题分别输入两张图像，二者的 token 序列长度与位置完全一致，因此 hidden states 可以逐 token 替换。

每个“变化类型 × 指定方式”条件取 64 个配对，目标位置均匀抽样。只保留两种输入均回答正确且答案 logit 差相差至少 1 的配对，被排除的配对数量与原因单独报告。

## 3. 方法

### 3.1 符号约定

- $c$：clean 输入，即异常图与问题；$\tilde{c}$：corrupted 输入，即对应的正常图与相同问题。
- $h^{\ell}_{t}(c) \in \mathbb{R}^{4096}$：输入 $c$ 在 decoder 第 $\ell$ 层输出处、第 $t$ 个 token 的 residual stream hidden state。
- $G$：一组 token 索引，称为 token 组。
- $z_{\mathrm{Yes}}$ 与 $z_{\mathrm{No}}$：回答首个 token 处“Yes”与“No”两个 token 的 logit。

### 3.2 答案 logit 差与恢复比例

答案的判别分数沿用实验 1.3 的定义：

$$
s = z_{\mathrm{Yes}} - z_{\mathrm{No}}.
$$

clean 输入的正确答案为 No，$s$ 为负；corrupted 输入的正确答案为 Yes，$s$ 为正。记 $s_{\mathrm{clean}}$ 与 $s_{\mathrm{corrupt}}$ 为两种输入的判别分数。patching 操作在 corrupted 输入的前向计算中，把第 $\ell$ 层输出处 token 组 $G$ 的 hidden states 替换为 clean 输入的对应值：

$$
h^{\ell}_{t}(\tilde{c}) \leftarrow h^{\ell}_{t}(c), \qquad t \in G,
$$

随后继续计算后续各层，得到判别分数 $s_{\mathrm{patched}}$。恢复比例定义为

$$
r(\ell, G) = \frac{s_{\mathrm{patched}} - s_{\mathrm{corrupt}}}{s_{\mathrm{clean}} - s_{\mathrm{corrupt}}}.
$$

分母是两种输入之间判别分数的总差，分子是 patching 使 corrupted 输入的分数向 clean 输入移动的量。$r = 1$ 表示替换该组 hidden states 足以使模型给出 clean 输入的答案，$r = 0$ 表示该组 hidden states 不携带决定答案的异常信息。本实验以 logit 差作为测量量：答案准确率只反映判别分数是否越过阈值，logit 差则对部分恢复同样敏感，能够区分“信息部分转移”与“完全没有转移”。

上述方向称为恢复方向，检验 token 组上的信息是否足以改变答案。另运行破坏方向：在 clean 输入中替换为 corrupted 输入的 hidden states，以 $(s_{\mathrm{clean}} - s_{\mathrm{patched}}) / (s_{\mathrm{clean}} - s_{\mathrm{corrupt}})$ 衡量该组信息对答案是否必要。两个方向的结论一致时，才将某一 token 组认定为承载异常信息的位置。

每个条件报告 $r$ 的配对中位数与以配对为单位的 bootstrap 95% 区间，并报告比值的均值形式（先对分子与分母分别求均值再相除），后者不受个别分母较小配对的影响。

### 3.3 扫描范围

扫描在“提取位置 × token 组”的网格上进行。

提取位置包括：

1. **视觉输入侧。** ViT 第 3、9、18、27 层输出与 merger 输出。替换 ViT 层输出时，只替换对应 token 组的 patch 特征，ViT 后续层与 merger 重新计算；替换 merger 输出即替换 decoder 的视觉输入。
2. **decoder。** 第 1 至 32 层的全部输出。

token 组包括：

| token 组 | 定义 | 检验内容 |
| --- | --- | --- |
| 目标实例 | 指定实例单元覆盖的 visual tokens（6×6 网格下为 2×2 个） | 异常信息是否停留在目标位置 |
| 相邻实例 | 与目标实例上下左右相邻的实例的 visual tokens | 信息是否扩散到邻近实例 |
| 高范数 | 实验 2.2a 标记的高范数 token；2.2 完成前按图像内范数最高的 5% 选取 | 高范数 token 是否汇集全局信息 |
| 全部视觉 | 全部 visual tokens | 视觉侧信息的总量 |
| 问题文本 | 图像之后、最后一个 prompt token 之前的文本 token | 信息是否已转移到问题文本 |
| 最后一个 prompt token | 生成第一个回答 token 前的最后一个输入 token | 信息是否已汇集到回答位置 |
| 随机视觉 | 与目标实例数量相同、随机选取的非目标 visual tokens（5 个种子） | 对照 |

由此得到每个条件的“层 × token 组”恢复比例热图。预期的典型形态是：浅层中目标实例 token 组的 $r$ 接近 1，随层数增加逐渐下降，同时最后一个 prompt token 的 $r$ 逐渐上升；二者交叉的层即异常信息从视觉位置转移到回答位置的层。

### 3.4 Gated DeltaNet 层的影响

Qwen3.5-9B 的 decoder 中，24 个 Gated DeltaNet 层以递推状态累积此前 token 的信息，只有第 4、8、12、16、20、24、28、32 层为 full attention。patching 第 $\ell$ 层输出时，第 $\ell$ 层及之前各层的递推状态仍来自 corrupted 输入，而第 $\ell + 1$ 层及之后各层的递推状态由替换后的 hidden states 重新计算。因此：

1. 在第 $\ell$ 层输出处替换目标实例 token，等价于让第 $\ell$ 层之后的计算看到 clean 的目标信息；在第 $\ell$ 层之前已经通过递推状态或 attention 转移到后续 token 的信息不被替换。目标实例 token 组的 $r$ 随层数下降，正是由于越来越多的异常信息已在较浅层转移到后续 token。
2. 递推状态是一个压缩的记忆，其中的信息不归属于单个 token。若最后一个 prompt token 的 $r$ 在某一层之前始终较低，而目标实例 token 的 $r$ 也已下降，则信息可能存放于中间 token 的递推状态中。为此另设可选实验：直接替换某一 Gated DeltaNet 层在最后一个 visual token 处的递推状态，检验视觉段结束时的状态是否承载异常信息。

### 3.5 实现

patching 以 PyTorch forward hooks 实现：clean 前向计算缓存各层输出，patched 前向计算在指定层的 hook 中替换 token 组的 hidden states。所有前向计算使用 bfloat16、batch size 1、thinking off；另在 16 个配对上以 float32 复核 $r$，确认数值精度不影响结论。每个配对的扫描约需 $(5 + 32) \times 7$ 次前向计算，64 个配对与 4 个条件合计约 7 万次 prefill。

### 3.6 失败样本的中断位置（2.5b）

2.5a 只扫描两种输入均答对的配对，其热图给出了答对时的读取路径。2.5b 在模型答错的条件上沿用同一套替换操作，并增加不依赖答案的信息读出，以确定异常信息在“目标 token → 文本 token → 答案”这条路径上的哪一段中断。

**任务条件。** 图中指定实例判断统一采用修订后的 T3 问法（总计划第 6 节第 8 条）：“Is there anything about the figure in row r, column c that differs from the majority of the figures? Rows are counted from top to bottom and columns from left to right, starting at 1. Answer with only Yes or No.”。clean 输入的指定单元为异常实例，正确答案为 Yes；corrupted 输入为同一规格去掉异常后的正常图，正确答案为 No。条件如下，每个条件 40 个配对，目标位置由 seed 0 均匀抽样，模板循环使用四个开发模板：

| 条件 | 网格 | 异常 | 作用 |
| --- | --- | --- | --- |
| t3_red_g06 | 6×6 | 纯红色 | 答对条件的参照，对应 2.5a 的 red_rowcol |
| t3_gap16_g06 | 6×6 | 16-pixel 缺口 | 答对条件的参照，对应 2.5a 的 gap_rowcol |
| t3_gap16_g10、t3_gap16_g20 | 10×10、20×20 | 16-pixel 缺口 | 网格规模 |
| t3_color20_g06、t3_color10_g06 | 6×6 | CIELAB 色差 20 与 10 | 低对比度颜色 |
| t3_fail | 3×3、6×6、10×10 | 实验 1.3 的原图像 | 两实例比较（T2）成功而完整找不同（T5）答错的图像，指定真实目标 |
| t5_fail | 同上 | 同上 | 同一批图像上的 T5 行列答案 |

t3_fail 与 t5_fail 的图像取自实验 1.3 正式运行中 T2 配对答对、T5 行列答案判为错误的记录。其中行列数值正确、只是 JSON 闭合括号写错的答案属于格式失败，不进入本实验；其余答案的行或列与真值不同，称为内容失败。

**输出层面的度量。** 记第 $i$ 个配对的判别分数差为

$$
\Delta_i = s_{\mathrm{clean}, i} - s_{\mathrm{corrupt}, i},
$$

它衡量异常在答案 logit 上留下的总效应。每个条件报告 clean 输入回答 Yes 的比例、corrupted 输入回答 No 的比例、$s$ 区分两类输入的 AUROC，以及 $\Delta_i \ge 1$ 的配对比例。失败条件中许多配对的 $\Delta_i$ 接近 0，单个配对的恢复比例 $r$ 没有稳定的分母，因此除了 2.5a 的配对中位数与比值的均值形式之外，另报告 logit 位移

$$
\bar{u}(\ell, G) = \frac{1}{N} \sum_{i=1}^{N} \left(s_{\mathrm{patched}, i}(\ell, G) - s_{\mathrm{corrupt}, i}\right),
$$

其中 $N$ 为条件内全部配对数，$s_{\mathrm{patched}, i}(\ell, G)$ 为恢复方向替换后的判别分数。$\bar{u}$ 以 logit 为单位，在 $\Delta$ 很小时仍有定义，并且与平均总效应 $\bar{\Delta}$ 直接可比。

**不依赖答案的信息读出。** 恢复比例只能反映传到答案的那部分信息。若异常信息已经进入文本 token、却没有被答案使用，替换实验无法与“信息从未转移”相区分。为此在每个配对、每个提取位置 $\ell$ 与位置组 $g$ 上记录配对差分

$$
\delta_i(\ell, g) = \bar{h}^{\ell}_{g}(c_i) - \bar{h}^{\ell}_{g}(\tilde{c}_i) \in \mathbb{R}^{4096},
$$

其中 $\bar{h}^{\ell}_{g}$ 为位置组 $g$ 内 hidden states 的均值。位置组取目标实例的 4 个 visual tokens、问号 token、问题末尾 token、`</think>` 与最后一个 prompt token。在配对之间做 8 折交叉验证：以训练折的平均差分 $w = \operatorname{mean}_{i \in \mathrm{train}} \delta_i$ 为方向，验证折配对的投影 $w^{\top}\delta_i > 0$ 记为正确，正确比例称为配对 probe 准确率，随机水平为 0.5。该读出器只有一个由均值确定的方向，没有可训练参数，因此在 40 个配对上不会过拟合；它检验的是“clean 与 corrupted 之间是否存在跨配对一致的线性差异”。另以 t3_gap16_g06 的平均差分作为固定方向，在其他条件上计算同一准确率（迁移准确率），检验失败条件中是否存在与答对条件相同的“有差异”编码。

**判读。** 三个层面的结果组合给出中断位置：

| 目标 token 上的读出 | 文本 token 上的读出 | 答案 $\Delta$ | 中断位置 |
| --- | --- | --- | --- |
| 低 | 低 | 小 | 编码阶段：异常没有进入目标实例的表征 |
| 高 | 低 | 小 | 视觉到文本的读取失败 |
| 高 | 高 | 小 | 读取后丢失：文本 token 含有差异，而最后几层没有使用 |
| 高 | 高 | 大 | 路径完整（答对条件） |

**t5_fail 的判别分数。** 找不同的答案需要生成，判别分数在第一个答错的数字上定义：以模型自己生成的答案作 teacher forcing，取答案中第一个与真值不同的数字位置，记真值数字的 logit 为 $z_{\mathrm{true}}$、模型实际生成数字的 logit 为 $z_{\mathrm{pred}}$，

$$
s = z_{\mathrm{true}} - z_{\mathrm{pred}}.
$$

clean 输入上 $s < 0$（模型答错）。corrupted 输入去掉了异常，若异常仍把答案推向真值，则 $\Delta > 0$，替换实验可以定位这部分不足以改变答案的信息经由哪些 token 传递。

### 3.7 生成阶段的 patching（2.5c）

2.5a 的结论“第 16 层之后 visual token 不再被使用”只针对回答首个 token 的 prefill。实验 3.5 的 attention 读出显示生成坐标数字时模型在第 12–24 层重新读取目标 token，但 attention 权重不能说明读取的内容是否决定了答案。2.5c 以替换实验回答两个问题：生成坐标数字时 visual token 在哪些层仍有因果作用；答案所用的信息来自生成时对图像的直接读取，还是来自 prompt 侧在 prefill 中已经写入的结果（问题末尾、`</think>`、最后一个 prompt token）。

**任务与配对。** 均为 6×6 网格、64-pixel 单元，问题与格式要求取自实验 1.3：

| 条件 | 问题 | 答案格式 | 配对 | 计分的数字 |
| --- | --- | --- | --- | --- |
| t6_rowcol_col、t6_rowcol_row | T6 红色目标 | 行列 | 红色目标沿列（或行）移动，另一坐标不变 | 列号（或行号） |
| t6_point_x | T6 红色目标 | 中心点 | 红色目标沿列移动 | x 坐标第一个不同的数字 |
| t5_rowcol_col、t5_rowcol_row | T5 找不同（16-pixel 缺口） | 行列 | 缺口实例沿列（或行）移动 | 列号（或行号） |
| t5_rowcol_remove | T5 找不同 | 行列 | corrupted 图像去掉缺口 | 行号 |

每个条件 24 个配对。两种输入先以 3.5 的 greedy 解码器（`coordinate_readout.greedy`）生成答案。移动配对要求两种输入都答对；去除配对只要求 clean 输入答对。

**计分位置与判别分数。** 记 clean 答案的 token 序列为 $a_{1:T}$。在被移动的坐标上找到 clean 与 corrupted 两个数值第一个不同的数字，设其在 clean 答案中的位置为 $t^{*}$。接收序列为“prompt + $a_{1:t^{*}-1}$”，两种输入的接收序列逐 token 相同，最后一个位置（称为 query token）预测第 $t^{*}$ 个答案 token。判别分数为

$$
s = z_{d} - z_{\tilde{d}},
$$

其中 $d$ 为 clean 答案在 $t^{*}$ 处的数字，$\tilde{d}$ 为 corrupted 答案在同一坐标同一数位上的数字。两个答案的格式可能不同（例如是否带代码块标记），计分只比较数值，corrupted 输入同样在 clean 答案的前缀上计算。去除配对中 corrupted 答案若不含坐标，则 $\tilde{d}$ 取 corrupted 输入在该位置上 logit 最高的、不同于 $d$ 的数字。合格条件与 2.5a 相同：$s_{\mathrm{clean}} > 0 > s_{\mathrm{corrupt}}$ 且 $\Delta \ge 1$。

**提取位置。** 记 site $k = 0$ 为 decoder 的输入 embedding（visual token 处即 merger 输出），site $k \ge 1$ 为 decoder 第 $k$ 层输出，扫描 $k = 0, 1, \dots, 32$。

**token 组。** 在 2.5a 的基础上区分 prompt 与答案：

| token 组 | 定义 |
| --- | --- |
| clean 目标、corrupted 目标、两处目标 | clean 图像异常单元、corrupted 图像异常单元的 4 个 visual tokens，以及二者的并集 |
| 随机视觉、全部视觉 | 4 个随机非目标 visual tokens；全部 144 个 visual tokens |
| 问题末尾、`</think>`、最后一个 prompt token | 2.5a 中承载传递信息的 prompt 位置 |
| 全部 prompt 文本 | 图像之后的全部 prompt token |
| 答案前缀 | 已生成、位于 query token 之前的答案 token |
| query token | 预测计分数字的位置 |

**路径分解。** 设接收输入为 $\tilde{c}$、来源输入为 $c$（恢复方向），prompt 长度为 $P$，序列中位置 $t < P$ 为 prompt，$t \ge P$ 为答案。除 2.5a 的单点替换（记为 std）之外，定义三种多层替换：

1. **direct（直接读取）。** 对全部 $k \ge \ell$，visual token 组 $G$ 取来源值，其余 prompt 位置固定为接收输入自身的值：

$$
h^{k}_{t} \leftarrow
\begin{cases}
h^{k}_{t}(c), & t \in G, \\
h^{k}_{t}(\tilde{c}), & t < P,\ t \notin G,
\end{cases}
\qquad k \ge \ell .
$$

答案位置照常计算。prompt 文本被固定后无法把 clean 信息传给答案，效应只能来自答案 token 在第 $\ell + 1$ 层及以后直接读取 $G$。

2. **direct1（单层读取）。** 只在 site $\ell$ 上替换 $G$，site $k > \ell$ 上全部 prompt 位置固定为接收值。答案 token 只在第 $\ell + 1$ 层读到来源值，用于定位读取发生的具体层。

3. **via（经 prompt 传递）。** 先运行 std 替换（site $\ell$ 上替换全部 visual tokens，后续自由计算），记录图像之后 prompt 文本 token 在各层的值 $h^{k}_{t}(\tilde{c} \mid \ell)$；再在新的前向计算中令

$$
h^{k}_{t} \leftarrow
\begin{cases}
h^{k}_{t}(\tilde{c} \mid \ell), & t \in T, \\
h^{k}_{t}(\tilde{c}), & t \in V,
\end{cases}
\qquad k > \ell ,
$$

其中 $T$ 为图像之后的 prompt 文本位置，$V$ 为全部 visual token 位置。答案位置看到的是已读取 clean 图像的 prompt 文本与 corrupted 的 visual tokens，效应只能经由 prompt 侧传递。

若 std 的效应约等于 direct，via 接近 0，则答案信息在生成时从图像直接读取；若 via 约等于 std，则答案使用的是 prompt 中已写入的结果。Gated DeltaNet 层的递推状态由各位置的层输入决定，固定 prompt 位置的层输出即同时固定了递推状态与 short convolution 中来自 prompt 的部分，因此这一分解对两类层同样成立。

**规模与检查。** 恢复方向扫描全部 token 组与三种路径，破坏方向只扫描两处目标、全部视觉、全部 prompt 文本、最后一个 prompt token、query token 以及全部视觉的 direct 与 via。每个配对另运行 2.5a 的两项数值检查：以接收输入自身的值替换全部位置时分数不变；在 site 1 与 16 替换全部位置时 $r = 1$。

**greedy 解码一致性。** 本实验的答案由 3.5 的 greedy 解码器在 batch size 2 下生成，另取 20 个配对与 `qwen_runtime.generate` 比较答案。

### 3.8 概念擦除（2.5d）

概念擦除使用 LEACE: Perfect Linear Concept Erasure in Closed Form（Belrose et al., 2023, arXiv:2306.03819）。设第 $\ell$ 层的 visual token 特征为随机向量 $x \in \mathbb{R}^{D}$，概念标签 $y$ 以 one-hot 向量表示，共 $k$ 类。由训练数据估计 $x$ 的协方差 $\Sigma_{xx} \in \mathbb{R}^{D \times D}$ 与 $x$、$y$ 的交叉协方差 $\Sigma_{xy} \in \mathbb{R}^{D \times k}$，记白化矩阵 $W = \Sigma_{xx}^{-1/2}$，擦除函数为

$$
x' = x - W^{+} P \, W \left(x - \mathbb{E}[x]\right),
$$

其中 $P \in \mathbb{R}^{D \times D}$ 是投影到 $W\Sigma_{xy}$ 列空间的正交投影矩阵，$W^{+}$ 是 $W$ 的伪逆。该式先把特征白化，在白化空间中去除与标签相关的方向，再映射回原空间。LEACE 保证擦除后不存在任何线性分类器能以优于常数预测的效果预测 $y$，且在满足这一条件的仿射变换中对特征的改变最小（以均方距离计）。最小改变的性质使擦除对其他信息的影响尽可能小，这是选择 LEACE 的原因；直接投影掉 probe 权重方向的做法既不能保证完全擦除线性信息，也可能删除不必要的成分。擦除子空间的维度不超过 $k - 1$，二分类概念只擦除一个方向。

概念标签取自实验 2.3 在 token 级别的定义：token 是否属于经过变化的实例（二分类），以及该实例的变化类型（多分类）。擦除函数在 2.3 的训练图像上拟合。实施方式如下：

1. **擦除位置。** 主条件在 merger 输出（decoder 视觉输入）上擦除；辅助条件在 decoder 第 1 至 $\ell$ 层的输出上逐层擦除，各层使用各自拟合的擦除函数，$\ell$ 取 2.5a 中信息开始转移到文本位置之前的层。逐层擦除可以防止后续层从未擦除的成分中重新计算出该属性。
2. **作用范围。** 擦除作用于全部 visual tokens，文本 token 不变。
3. **擦除检验。** 在擦除后的特征上重新训练线性 probe，确认其准确率降至随机水平；同时训练 MLP probe，测量残余的非线性可读性。
4. **对照。** 在白化空间中随机选取同样维度的子空间执行相同擦除，5 个随机种子。

测量指标为实验 1.3 的 T3 与 T5 准确率与 AUROC，以及 COCO caption 在 teacher forcing 下的平均 token loss，后者用于检查擦除是否广泛损害模型。

## 4. 指标

| 指标 | 定义 | 子实验 |
| --- | --- | --- |
| 恢复比例 $r$ | patching 后判别分数向 clean 输入移动的比例 | 2.5a |
| 破坏比例 | 反向 patching 后判别分数向 corrupted 输入移动的比例 | 2.5a |
| 转移层 | 图像之后全部文本 token 的 $r$ 超过全部视觉 token 的 $r$ 的最浅层。异常信息先由视觉 token 进入问题文本与 chat template，再汇集到最后一个 prompt token，因此以“全部图像后文本”与“全部视觉”的比较确定信息离开视觉 token 的层 | 2.5a |
| logit 位移 $\bar{u}$ | 恢复方向替换后判别分数相对 corrupted 输入的平均移动，以 logit 为单位 | 2.5b |
| 配对 probe 准确率 | 以训练折平均差分为方向、验证折配对差分投影为正的比例 | 2.5b |
| 路径恢复比例 | direct、direct1、via 三种替换下的恢复比例 | 2.5c |
| 准确率与 AUROC 变化 | 擦除后相对未擦除的变化，减去随机子空间擦除的变化 | 2.5d |
| 残余可读性 | 擦除后线性与 MLP probe 的 AUROC | 2.5d |
| caption loss 变化 | 擦除前后 COCO caption 的平均 token loss 之差 | 2.5d |

比例使用 Wilson 95% 区间，连续指标使用以配对或图像为单位的 bootstrap 95% 区间。

## 5. 可视化

1. **恢复比例热图。** 行为 token 组，列为提取位置（ViT、merger 与 decoder 各层），颜色表示 $r$ 的中位数；恢复与破坏方向各一张，按变化类型与指定方式分面。
2. **转移曲线。** 目标实例、问题文本与最后一个 prompt token 三个 token 组的 $r$ 随 decoder 层变化的曲线，标出 full attention 层的位置，观察信息转移是否集中发生于 full attention 层。
3. **示例。** 选取典型配对，展示 clean 图、corrupted 图与指定实例，并列出各 token 组在关键层的 $r$。
4. **生成阶段与 prefill 的对照（2.5c）。** 生成阶段的“token 组 × site”恢复比例热图与 2.5a、2.5b 的 prefill 热图上下排列；另以曲线比较 std、direct、direct1 与 via 四种替换随 site 的变化。
5. **失败条件（2.5b）。** 各条件的 logit 位移热图，以及目标 token 与文本 token 上配对 probe 准确率随 site 的曲线。
6. **擦除效应。** 各擦除条件下的准确率变化以森林图展示，随机子空间对照并列；另以条形图展示擦除后的残余可读性。

## 6. 结果模式与解释

| 结果模式 | 解释 |
| --- | --- |
| 目标实例 token 在浅层 $r \approx 1$，最后一个 prompt token 在中层 $r$ 上升 | 异常信息停留在目标实例位置，并在中层被读取到回答位置，读取路径完整 |
| 目标实例 token 的 $r$ 在全部层均较低，全部视觉 token 的 $r$ 较高 | 异常信息分布于多个视觉 token，不局限于目标实例 |
| 高范数 token 组的 $r$ 显著高于随机视觉 token | 高范数 token 汇集了与答案相关的全局信息，支持实验 2.2 中的 register 假设 |
| 失败条件中目标 token 的 $r$ 高，但最后一个 prompt token 的 $r$ 始终接近 0 | 信息存在于视觉 token 上而未被读取，支持问题 5 中重新输入局部特征的方向 |
| 转移只发生在 full attention 层之后 | 跨 token 的信息读取主要由 full attention 层完成，Gated DeltaNet 层的递推状态作用较小 |
| LEACE 擦除使 T3、T5 准确率显著下降，随机擦除无影响 | 模型使用 2.3 识别的线性属性子空间 |
| LEACE 擦除后线性 probe 降至随机水平，而准确率不变 | 模型不依赖该线性子空间，可能使用非线性编码或其他信息 |

## 7. 运行与产物

1. 代码：2.5a 为 `analysis_tools/representation/activation_patching.py`；2.5b 与 2.5c 共用 `analysis_tools/representation/patching_extension.py` 与 `patching_extension_analysis.py`，答案生成、visual embedding 与 teacher forcing 序列通过导入 `analysis_tools/position/coordinate_readout.py` 的 prepare、embeds、teacher_forced 与 greedy 构造；2.5d 为 `analysis_tools/representation/concept_erasure.py`，hook 工具与实验 2.2d 共用。
2. Bash 入口：`scripts/analysis/representation/activation_patching.sh`（2.5a）、`scripts/analysis/representation/patching_extension.sh`（2.5b、2.5c）与 `scripts/analysis/representation/concept_erasure.sh`（2.5d），任务条件、提取位置、配对数量、batch size 与随机种子集中于脚本顶部。
3. 产物保存于 WORK_DIR 的 `analysis/representation/<日期>_activation_patching/`、`analysis/representation/<日期>_patching_extension/` 与 `analysis/representation/<日期>_concept_erasure/`，包括配对规格、生成的答案与计分 token、clean 与 corrupted 的判别分数、逐配对逐条件的替换记录、被排除配对及原因、配对差分（2.5b）、擦除函数参数与擦除前后的模型输出。

## 8. 解释范围

1. activation patching 测量的是在给定配对与问题下，替换某组 hidden states 对答案的作用；结论适用于所选任务条件，不能直接推广到模型答错的条件。模型答错条件中的信息中断位置，需要以答对条件的恢复比例热图为参照比较。
2. 替换 hidden states 会同时改变该 token 上的全部信息，恢复比例不能区分其中哪一种属性起作用；属性层面的因果检验由 2.5d 给出。
3. 概念擦除只去除线性可读的信息。擦除无效不能排除模型使用该属性的非线性编码；擦除有效也可能部分来自对其他相关信息的损伤，因此需以随机子空间擦除与 caption loss 变化为参照。
4. 配对图像只在一个实例上不同，模型的回答受问题措辞影响；两种指定方式的结果一致时，结论才不依赖于具体问法。
