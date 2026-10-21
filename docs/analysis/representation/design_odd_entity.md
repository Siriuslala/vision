# 实验 2.6 设计：异常实体的比较机制

本文属于[视觉细节表征](plan.md)研究，给出实验 2.6 的问题、假设、刺激、指标与操作。结果见[实验 2.6 结果](results_2026-10-20_odd_entity.md)。主模型为 Qwen3.5-9B，冻结、bfloat16、SDPA，主评测为 thinking off。

## 1. 研究问题

找不同类任务（BabyVision 的 Find the different、实验 1.3 的 T5）要求模型在许多实体中指出与众不同的一个。已有实验回答了“异常信息经由哪些 token 进入答案”：[实验 2.5a](results_2026-10-13_activation_patching.md) 表明在指定实例判断中异常信息由目标实例的 visual tokens 进入，在 full attention 第 8–16 层转入文本 token；[实验 2.5c](results_2026-10-15_patching_extension.md) 表明生成行列数字时，答案前的格式 token 在第 16–20 层直接读取目标 visual tokens。这些实验都没有回答一个更基本的问题：**模型如何知道哪一个实体是异常的。** 人在完成该任务时通常先形成“多数实体长什么样”的模板，再寻找偏离模板的个体；模型是否存在类似的比较过程，或者以其他方式实现，尚无证据。

本实验检验三类假设：

1. **H1（多数模板）。** 模型在某处形成多数实体的模板表示，例如汇聚于问题文本的 token，或经 Gated DeltaNet 的递推状态累积于靠后的 visual tokens，再把每个实体与模板比较。
2. **H2（局部对比）。** 异常由局部对比产生，例如 ViT 或 decoder 浅层中目标与相邻实体的差异形成 pop-out 信号，不需要全局模板。
3. **H3（其他途径）。** 例如模型不进行实体间比较，只在图中搜索“带有某种显著属性”的实体；此时异常的判定依赖属性本身是否显著，与多数实体是什么无关。

实验目标是得到一个机制性结论：异常性（oddness）在哪一阶段、以何种形式出现，以及答案如何使用它。

## 2. 核心构造与符号

### 2.1 配对四元组

核心刺激为 6×6 网格（单元 64 pixels，画布 384×384 pixels，12×12 个 merged visual tokens，每个实体覆盖 2×2 个 token），取自[共享合成刺激集](../stimuli.md)的生成器。记模板为 $T$，经过一种属性变化的实例为 $X$。每个配对由同一模板、同一目标单元与同一缺口位置生成四张图像：

| 变体 | 目标实体 | 其余 35 个实体 | 异常实体 |
| --- | --- | --- | --- |
| std | $X$ | $T$ | $X$ |
| swap | $T$ | $X$ | $T$ |
| homT | $T$ | $T$ | 无 |
| homX | $X$ | $X$ | 无 |

std 与 swap 的关键性质是：两者的异常实体位于同一单元，但异常实体的属性相反。若某一表征只编码属性（“这是 $X$”），它在 std 与 swap 中对目标给出相反的信号；若它编码异常性（“与其余实体不同”），则在两者中给出相同的信号。homT 与 homX 不含异常实体，用作 patching 的供体与文本侧读出的参照。

### 2.2 变化类型

| 记号 | 变化 | 实验 1.3 中 6×6 T5 行列准确率 |
| --- | --- | --- |
| red | 笔画改为纯红色（显著颜色） | 1.00 |
| color20 | CIELAB 色差 20 的深色 | 0.75 |
| color10 | CIELAB 色差 10 的深色（低对比） | 0.15 |
| gap16 | 16-pixel 缺口（显著形状） | 0.85 |
| gap2 | 2-pixel 缺口（细微形状） | 0.81 |

每种变化 72 个配对，目标单元在 36 个单元中各出现两次，使每一行都有 12 个目标，以便按光栅顺序分析。模板在四个开发模板间循环。

### 2.3 符号

- $h^{\ell}_{e} \in \mathbb{R}^{d}$：提取位置 $\ell$ 处实体 $e$ 的状态，取该实体覆盖的全部 token（ViT 为 16 个 patch，merger 与 decoder 为 4 个 merged token）的均值。ViT 中 $d = 1152$，merger 输出与 decoder 中 $d = 4096$。提取位置记为 V$k$（ViT 第 $k$ 个 block 输出）、D0（merger 输出，即 decoder 的视觉输入）与 D$k$（decoder 第 $k$ 层输出），层号从 1 开始。分析前在每个提取位置对全部实体逐维标准化。
- $t$：目标实体的索引；$n = 36$ 为实体数。
- 目标对比 $D \in \mathbb{R}^{d}$：同一图像中目标状态与其余实体均值之差，

$$
D = h_{t} - \frac{1}{n-1} \sum_{e \neq t} h_{e}.
$$

  在图像内部做差消去了全图共有的成分（例如多数实体的属性所造成的全局偏移），只保留目标相对于其余实体的差异。

- $\bar{D}_{\mathrm{std}}$、$\bar{D}_{\mathrm{swap}}$：std 与 swap 图像上 $D$ 的配对均值。

### 2.4 异常性与属性的分解

把目标对比写成属性成分与异常性成分之和。设 $u_{\mathrm{feat}}$ 为“$X$ 相对 $T$”的属性方向，$u_{\mathrm{odd}}$ 为“与其余实体不同”的异常性方向，则在 std 中目标是 $X$ 且异常，在 swap 中目标是 $T$ 且异常，

$$
\bar{D}_{\mathrm{std}} \approx u_{\mathrm{odd}} + u_{\mathrm{feat}}, \qquad \bar{D}_{\mathrm{swap}} \approx u_{\mathrm{odd}} - u_{\mathrm{feat}}.
$$

由此得到两个方向的估计

$$
u_{\mathrm{odd}} = \tfrac{1}{2}\left(\bar{D}_{\mathrm{std}} + \bar{D}_{\mathrm{swap}}\right), \qquad u_{\mathrm{feat}} = \tfrac{1}{2}\left(\bar{D}_{\mathrm{std}} - \bar{D}_{\mathrm{swap}}\right),
$$

并定义两项汇总指标：

$$
c = \cos\left(\bar{D}_{\mathrm{std}}, \bar{D}_{\mathrm{swap}}\right), \qquad \rho = \frac{\lVert u_{\mathrm{odd}} \rVert^{2}}{\lVert u_{\mathrm{odd}} \rVert^{2} + \lVert u_{\mathrm{feat}} \rVert^{2}}.
$$

$c = -1$（$\rho = 0$）表示目标的独特性完全来自其属性，即纯属性编码；$c = +1$（$\rho = 1$）表示目标在 std 与 swap 中向同一方向偏离其余实体，即纯异常性编码。该分解成立的原因在于：若实体状态只是自身属性与全局上下文的加性组合，那么在图像内部做差后上下文项抵消，std 与 swap 的对比只差一个符号；只有当实体状态依赖“自身属性与上下文是否一致”这一交互项时，两者才会出现同号分量。因此 $u_{\mathrm{odd}}$ 的出现等价于该位置的实体表征已经完成了某种自身与上下文的比较。

**交叉验证的读出。** 将配对分为 4 折，在训练折上估计 $u_{\mathrm{odd}}$ 与 $u_{\mathrm{feat}}$，在测试折的每张图像上计算全部实体的投影 $s_{e} = u^{\top} h_{e}$，以 $\arg\max_{e} s_{e} = t$ 的比例为 top-1 准确率（随机水平 $1/36$）。$u_{\mathrm{odd}}$ 在 std 与 swap 上的 top-1 同时高，说明单一方向就能在两种图像中找出异常实体，与其属性无关。另报告目标的异常性 margin，即目标投影减去其余实体投影均值、再除以其余实体投影的标准差。作为不依赖训练的对照，另报告“与图像均值距离最大的实体是否为目标”的比例；实验 2.1 已表明这类手工距离读出受位置相关成分主导。

## 3. 子实验

### 3.1 行为：std 与 swap 的对称性（检验 H3）

在全部 std 与 swap 图像上以实验 1.3 的 T5 问法生成行列答案：

> One figure differs from all the others. Give its position. Return only {"row": r, "column": c}, counting rows from top to bottom and columns from left to right, starting at 1. Output only the JSON, with no explanation.

若模型按多数模板比较，std 与 swap 应同样容易，因为两者的异常实体与多数实体之间差异相同；若模型搜索显著属性，swap（在 35 个带缺口或带颜色的实体中找出唯一完整或黑色的实体）应明显更难。另以实验 1.3 的显式推理问法（thinking off，先描述再回答）在每种变化的前 16 个配对上生成，比较直接回答与先推理后回答。std 与 swap 的配对差异用 exact McNemar 检验。

### 3.2 表征：异常性在哪一层出现（检验 H1 与 H2 的位置）

对全部 std 与 swap 图像执行一次 prefill，提取 V1、V3、V6、…、V27，D0，以及 D2、D4、…、D16、D20、D24、D28 的实体状态，计算第 2.4 节的 $c$、$\rho$、交叉验证 top-1 与 margin。

decoder 中 visual tokens 位于问题文本之前，因此其状态与问法无关；直接回答与显式推理两种问法下的实体状态完全相同，差别只能出现在文本 token 与答案 token。

### 3.3 光栅顺序（检验 decoder 内的比较）

ViT 使用双向 attention，每个实体都能看到全部实体；decoder 是因果的，第 $r$ 行的实体只能看到光栅顺序在它之前的实体。若异常性在 decoder 中通过比较新产生，第 1 行的目标在 decoder 中几乎没有可比较的上下文，其异常性 margin 应明显低于第 6 行；若异常性在 ViT 中已经形成，margin 不随目标行变化。该分析使用第 3.2 节的交叉验证 margin，按目标行分组。

### 3.4 上下文替换（因果检验）

在答对的 std 与 swap 图像上（red、color20、gap16、gap2 的每种变体按配对顺序取前 24 张答对的图像），以模型自己的直接回答作 teacher forcing，记答案分数

$$
S = \log p(r^{*}) + \log p(c^{*} \mid r^{*}),
$$

其中 $r^{*}$、$c^{*}$ 为正确行列数字，条件概率取自同一次前向计算中列数字位置的输出。在提取位置 $\ell$ 处把一组实体的 token 状态替换为供体图像的对应值，之后的计算照常进行，得到 $S_{\mathrm{patch}}$。三种替换如下：

1. **ctx_same：上下文替换为与目标相同的属性。** std 的非目标实体取自 homX，swap 的取自 homT。替换后，在位置 $\ell$ 处目标与其余实体在属性上已无差别；若答案仍然保留，说明目标的 token 在位置 $\ell$ 之前已经携带了自足的异常性标记，无需此后再与上下文比较。实体组为全部 35 个、8 个相邻实体、其余较远实体、光栅顺序在目标之前的实体与之后的实体。
2. **ctx_clean：上下文替换为不含异常实体的图像。** std 的非目标实体取自 homT，swap 的取自 homX。属性不变，只去掉上下文 token 在前向计算中从目标处获得的信息；若答案下降，说明多数实体的 token 也承载关于异常实体的信息。
3. **tgt：目标替换为多数实体的模板。** 用于确认目标 token 的必要性，与实验 2.5a 对照。

保留比例定义为

$$
R = \frac{\sum_{i}\left(S_{\mathrm{patch}, i} - S_{\mathrm{base}, i}\right)}{\sum_{i}\left(S_{\mathrm{clean}, i} - S_{\mathrm{base}, i}\right)},
$$

其中 $S_{\mathrm{clean}}$ 为未替换输入的分数，$S_{\mathrm{base}}$ 为供体图像本身在同一 teacher-forced 答案上的分数（homX 或 homT 中目标单元并不异常，因此它代表无异常时的水平）。$R = 1$ 表示替换不影响答案，$R = 0$ 表示答案退回无异常水平。比例以先求和再相除的形式汇总，并报告以图像为单位的 bootstrap 95% 区间；$S_{\mathrm{clean}} - S_{\mathrm{base}} \le 1$ nat 的配对分母不稳定，计数并剔除。

提取位置为 V1、V3、V6、…、V27，D0，D2、D4、…、D20、D24。在 ViT 位置替换时，替换的是对应 patch 的 block 输出，此后 ViT 各层与 merger 重新计算；因此在 V1 替换上下文几乎等同于把上下文像素换成供体图像。

预测如下：

| 机制 | ctx_same 的 $R$ 随位置的变化 | 实体组 |
| --- | --- | --- |
| ViT 内已形成异常性标记 | V 后段起 $R$ 接近 1 | — |
| decoder 内与上下文比较 | 直到 decoder 某层 $R$ 才回升 | before 组的作用大于 after 组 |
| 局部对比（H2） | — | nbr 组的作用大于 far 组 |
| 全局模板（H1） | — | far 组的作用不小于 nbr 组，随替换数量增加 |

### 3.5 文本侧的多数模板（检验 H1 的文本形式）

若模型在文本 token 上形成多数模板，那么这些 token 应编码多数实体的属性。在最后一个 visual token、问题最后一个 token 与最后一个 prompt token 上，以 homX 与 homT 的状态均值之差为方向 $w$，把 std 与 swap 的状态投影到该方向，归一化为

$$
\tau = \frac{w^{\top} h - \tfrac{1}{2} w^{\top}(\mu_{X} + \mu_{T})}{\tfrac{1}{2} w^{\top}(\mu_{X} - \mu_{T})},
$$

其中 $\mu_{X}$、$\mu_{T}$ 为 homX 与 homT 的状态均值。$\tau = +1$ 表示与全 $X$ 图相同，$\tau = -1$ 表示与全 $T$ 图相同。定义多数指数

$$
m = \tfrac{1}{2}\left(\tau_{\mathrm{swap}} - \tau_{\mathrm{std}}\right).
$$

swap 的多数实体为 $X$，std 的多数实体为 $T$，因此 $m = +1$ 表示该位置编码多数实体的属性（模板），$m = -1$ 表示它编码异常实体的属性。

### 3.6 局部对比、数量与同质性

1. **半半分割图（局部对比）。** 6×6 网格左右（或上下）各一半为 $X$ 与 $T$，不存在唯一的异常实体。将第 3.2 节在 std/swap 上估计的 $u_{\mathrm{odd}}$ 投影到每个实体上：若异常性来自局部对比，分界线两侧的实体得分应高于远离分界线的实体；若来自全局多数比较，各实体得分应接近（两种属性各占一半）。
2. **实体数量。** 3×3（8 个干扰实体）与 10×10（99 个干扰实体）网格，各 36 个配对的 std 与 swap，测量行为与第 2.4 节指标。
3. **多数的同质性。** 6×6 网格中每个实体的模板从四个开发模板随机抽取，异常实体只在属性变化上与众不同。若检测依赖模板比较，异质多数会破坏比较；若依赖属性本身的显著性，影响应较小。

### 3.7 自然图像检查

从 RefCOCO testA（人）与 testB（按指代表达中的关键词归类的物体）裁剪物体，缩放后放入 3×3 网格（单元 128 pixels，画布 384×384 pixels）。std 为 8 个类别 A 的不同实例加 1 个类别 B 的实例，swap 为 8 个类别 B 加 1 个类别 A，问法改为“One picture shows a different kind of object from all the others.”。多数实体在外观上互不相同，只在类别上一致，因此该条件检验语义层面的异常检测，并用同一分解指标测量异常性编码。

## 4. 与已有工作的关系

1. 实验 2.5a、2.5c 确定了异常信息的入口（目标 visual tokens）与读出层（第 16–20 层），本实验确定入口处的信号是属性还是异常性，以及它由何种比较产生。
2. 实验 1.3 发现 T5 与已知属性搜索 T4 的准确率接近，说明找不同可能并不需要额外的比较步骤；第 3.1 节的 std/swap 对称性直接检验这一推测。
3. 视觉搜索的心理学研究区分并行的特征 pop-out 与需要逐项比较的结合搜索，并报告“搜索不对称”：在缺少某特征的背景中寻找带特征的目标较易，反之较难。第 3.1 节的 swap 条件正是这一不对称的检验，可以把模型行为与该经典现象对照。

## 5. 运行与产物

- 入口：`scripts/analysis/representation/odd_entity.sh`，阶段依次为 build、behave、spotcheck、capture、patch、analyze；参数集中在脚本顶部。
- 代码：`analysis_tools/representation/odd_entity.py`（刺激、评分、指标）、`odd_entity_run.py`（GPU 阶段）、`odd_entity_analysis.py`（CPU 分析与作图）；测试为 `tests/analysis/representation/test_odd_entity.py`。
- 产物：WORK_DIR 的 `analysis/representation/2026-10-20_odd_entity/`；文档用图位于 `docs/analysis/representation/figures/odd_entity/`。
- 生成使用 `qwen_runtime.generate_batch`（直接回答 batch size 32，显式推理 16，10×10 网格减半），另取 20 条直接回答与 batch size 1 的 `generate` 对照。
