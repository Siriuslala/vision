# 实验 2.5：Activation patching 与概念擦除

本实验属于[视觉细节表征](plan.md)，为实验 2.2 至 2.4 的读出结果提供因果检验。probe 与特征反演只能说明信息可以从表征中恢复，不能说明模型在生成答案时使用了这些信息。本实验在完整模型的前向计算中直接替换或擦除表征，测量答案的变化，回答模型是否依赖目标实例 token 上的信息、依赖发生在哪些层，以及是否依赖 2.3 识别的属性子空间。

## 1. 研究问题与动机

2.1 表明配对差分可以在 ViT 与 merger 上定位局部改动；2.3 将测量单图中属性的可读性。若属性可读而模型仍答错，需要确定信息在何处中断：是目标实例 token 上的信息没有被问题文本读取，还是读取后没有转化为答案。本实验包含两个子实验：

1. **2.5a activation patching。** 在严格配对的正常图与异常图之间替换特定层、特定 token 组的 hidden states，测量答案 logit 差的恢复比例，确定承载异常信息的 token 与层，以及信息从视觉 token 转移到文本位置的层。
2. **2.5b 概念擦除。** 在视觉 token 中擦除 2.3 probe 识别的属性线性子空间，测量完整模型准确率的变化，检验模型是否使用该子空间。

2.5a 只需要已有的配对图像与 forward hooks，不依赖其他子实验，预计 1–2 天完成，安排在第一阶段；2.5b 依赖 2.3 的 probe，安排在第三阶段。

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

### 3.6 概念擦除（2.5b）

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
| 准确率与 AUROC 变化 | 擦除后相对未擦除的变化，减去随机子空间擦除的变化 | 2.5b |
| 残余可读性 | 擦除后线性与 MLP probe 的 AUROC | 2.5b |
| caption loss 变化 | 擦除前后 COCO caption 的平均 token loss 之差 | 2.5b |

比例使用 Wilson 95% 区间，连续指标使用以配对或图像为单位的 bootstrap 95% 区间。

## 5. 可视化

1. **恢复比例热图。** 行为 token 组，列为提取位置（ViT、merger 与 decoder 各层），颜色表示 $r$ 的中位数；恢复与破坏方向各一张，按变化类型与指定方式分面。
2. **转移曲线。** 目标实例、问题文本与最后一个 prompt token 三个 token 组的 $r$ 随 decoder 层变化的曲线，标出 full attention 层的位置，观察信息转移是否集中发生于 full attention 层。
3. **示例。** 选取典型配对，展示 clean 图、corrupted 图与指定实例，并列出各 token 组在关键层的 $r$。
4. **擦除效应。** 各擦除条件下的准确率变化以森林图展示，随机子空间对照并列；另以条形图展示擦除后的残余可读性。

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

1. 代码实现为 `analysis_tools/representation/activation_patching.py`（2.5a）与 `analysis_tools/representation/concept_erasure.py`（2.5b），hook 工具与实验 2.2d 共用。
2. Bash 入口为 `scripts/analysis/representation/activation_patching.sh` 与 `scripts/analysis/representation/concept_erasure.sh`，任务条件、提取位置、token 组、配对数量与随机种子集中于脚本顶部。
3. 产物保存于 WORK_DIR 的 `analysis/representation/<日期>_activation_patching/` 与 `analysis/representation/<日期>_concept_erasure/`，包括配对图像与 metadata、全部 prompt、clean 与 corrupted 的判别分数、逐配对逐条件的 $r$、被排除配对的记录、擦除函数参数与擦除前后的模型输出。

## 8. 解释范围

1. activation patching 测量的是在给定配对与问题下，替换某组 hidden states 对答案的作用；结论适用于所选任务条件，不能直接推广到模型答错的条件。模型答错条件中的信息中断位置，需要以答对条件的恢复比例热图为参照比较。
2. 替换 hidden states 会同时改变该 token 上的全部信息，恢复比例不能区分其中哪一种属性起作用；属性层面的因果检验由 2.5b 给出。
3. 概念擦除只去除线性可读的信息。擦除无效不能排除模型使用该属性的非线性编码；擦除有效也可能部分来自对其他相关信息的损伤，因此需以随机子空间擦除与 caption loss 变化为参照。
4. 配对图像只在一个实例上不同，模型的回答受问题措辞影响；两种指定方式的结果一致时，结论才不依赖于具体问法。
