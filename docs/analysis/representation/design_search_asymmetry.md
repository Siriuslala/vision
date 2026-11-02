# 实验 2.7 设计：视觉搜索不对称的机理

本文属于[视觉细节表征](plan.md)研究，给出实验 2.7 的问题、假设、刺激、指标与操作。结果见[实验 2.7 结果](results_2026-10-21_search_asymmetry.md)。主模型为 Qwen3.5-9B，冻结、bfloat16、SDPA，评测为 thinking off、直接回答。

## 1. 研究问题与动机

[实验 2.6](results_2026-10-20_odd_entity.md) 发现，找不同存在搜索不对称：在 6×6 网格中，带 2-pixel 缺口的一个图形混在完整图形中时准确率为 0.75，完整的一个混在带缺口的图形中时为 0.22。该现象与心理物理学中“特征出现优于特征缺失”（feature presence beats feature absence）的搜索不对称一致。实验 2.6 同时表明，对“拥有属性”的目标，ViT 第 12 个 block 之前即在目标 token 上形成自足的异常标记；对“缺少属性”的目标，比较延续到 decoder 第 1–8 层，并且只与光栅顺序在前的实体比较。

实验 2.6 的度量（std 与 swap 目标对比的余弦、上下文替换）说明了异常性在何处、何时出现，但这些度量的原理是实验者构造的对照，并不等同于模型自身完成定位的计算方式。本实验回答三个问题：

1. **不对称的范围。** 哪些特征维度上出现不对称，哪些维度上对称；不对称是否随实体数量变化，即模型的搜索更接近并行还是串行。
2. **不对称的来源。** ViT 中是否存在只在特征出现时激活的特征检测单元；“缺失”的目标在局部是否有任何信号。
3. **模型自身的定位机制。** 比较由哪类层完成（ViT 的双向 attention、decoder 的 Gated DeltaNet 层，或 full attention 第 4、8、12、16 层），信息如何从干扰实体的 token 流向目标 token，再流向答案。

## 2. 相关工作与定位

1. **心理物理学。** Treisman 与 Gelade（1980）的特征整合理论区分了特征搜索（并行，耗时不随干扰项数量增加）与合取搜索（串行）。Treisman 与 Souther（1985）发现 Q 混在 O 中容易找到，O 混在 Q 中困难，并据此提出搜索不对称是判断某一属性是否为前注意特征的诊断。Treisman 与 Gormican（1988）把不对称推广到倾斜与竖直、长与短、弯曲与笔直、开口与闭合、饱和与不饱和等维度，解释为特征检测器群体的“出现”信号可被并行检测，而“缺失”只能通过与参照的比较得到。Wolfe（1994，2001）的 Guided Search 把不对称归因于自下而上的激活图中目标与干扰项的激活差。Rosenholtz（2001）指出许多不对称可由显著性模型解释：目标的显著性等于其特征相对干扰项分布的 Mahalanobis 距离，干扰项分布的离散程度不同即会产生不对称。
2. **深度网络。** Gupta 等（2021）在 ImageNet 训练的网络上复现了多种人类搜索不对称，并把来源归于自然图像统计。Campbell 等（2024）发现 VLM 在合取搜索中准确率随实体数量下降，把它解释为绑定问题（binding problem）。
3. **本实验的定位。** 已有深度网络研究停留在行为层面。本实验在同一 VLM 中把行为不对称追溯到 ViT 神经元的激活符号结构，并用对称的神经元编辑检验因果性，同时定位 decoder 中完成剩余比较的层类型，回答“出现”信号与“缺失”信号在模型中分别经过哪条路径。

## 3. 刺激

### 3.1 特征对

每个特征对 $(A, B)$ 中，$A$ 为带有额外特征或在某一维度上“更多”的实体，$B$ 为标准实体。实体为 64-pixel 单元内的黑色线条图形（饱和度对为填充圆盘），笔画宽 3 pixels，以 4 倍超采样绘制后缩小；每个实体相对单元中心有 $\pm 3$ pixels 的随机偏移，偏移在同一组图像间共享。

| 记号 | $A$ | $B$ | subtle 参数 | salient 参数 |
| --- | --- | --- | --- | --- |
| qo | Q（O 加一条径向线段） | O（半径 14） | 线段长 6 px | 14 px |
| tilt | 倾斜线段（长 32） | 竖直线段 | 8° | 20° |
| length | 长竖直线段 | 短竖直线段（长 28） | 长 32 px | 40 px |
| curve | 弧线（弦长 32） | 直线段 | 拱高 3 px | 8 px |
| closure | 开口圆 | 闭合圆 | 缺口弧长 4 px | 12 px |
| sat | 饱和圆盘（$C^{*}=60$） | 不饱和圆盘 | 色度差 15 | 40 |
| size | 大圆 | 小圆（直径 28） | 直径 32 px | 40 px |

Q 线段的方向、缺口方向、倾斜方向与弧线凸向在每组图像内固定，在组间随机，因此 $A$ 与 $B$ 只在被操纵的特征上不同。实验 2.6 的 gap2、gap16、color20 与 red 作为补充特征对参与比较。

### 3.2 四元组

与实验 2.6 相同，每组图像共享目标单元、方向与偏移，包括四个变体：

| 变体 | 目标实体 | 其余实体 | 异常实体的性质 |
| --- | --- | --- | --- |
| std | $A$ | $B$ | 拥有特征（presence） |
| swap | $B$ | $A$ | 缺少特征（absence） |
| homA | $A$ | $A$ | 无异常 |
| homB | $B$ | $B$ | 无异常 |

std 与 swap 中目标与干扰项的像素差相同、方向相反，因此两者的准确率差只能来自模型对差异方向的处理。

### 3.3 规模

| 组别 | 网格 | 每个特征对的组数 | 变体 |
| --- | --- | --- | --- |
| 主组 subtle | 6×6 | 72（每个单元作目标两次） | std、swap；前 24 组另有 homA、homB |
| 主组 salient | 6×6 | 36 | std、swap |
| 实体数量 | 3×3、10×10 | 36 | std、swap（subtle） |
| 合取 | 3×3、6×6、10×10 | 36、72、36 | 目标为红色圆盘 |

合取组的干扰项为绿色圆盘与红色方块各半（conj），对照为只有绿色圆盘（feat_color）或只有红色方块（feat_shape）。

## 4. 指标与符号

### 4.1 行为

准确率要求行与列都正确；另记录行列各相差不超过 1 的近似命中，用于区分定位失败与大网格中的行列计数误差。std 与 swap 以组为单位配对，用精确 McNemar 检验，区间为 Wilson 95% 区间。

### 4.2 ViT 神经元的选择性

记 ViT 第 $k$ 个 block 的 MLP 中间激活（GELU 之后）为 $a^{k}_{p} \in \mathbb{R}^{4304}$，$p$ 为 patch。实体 $e$ 的激活取其覆盖的 16 个 patch 上的逐维最大值，

$$
z^{k}_{e,j} = \max_{p \in e} a^{k}_{p,j},
$$

取最大值是因为 Q 的线段、缺口等特征只占实体的少数 patch。神经元 $j$ 的选择性以 homA 与 homB 中全部实体的激活计算：

$$
d'_{j} = \frac{\mu_{A,j} - \mu_{B,j}}{\sqrt{\tfrac{1}{2}\left(\sigma^{2}_{A,j} + \sigma^{2}_{B,j}\right)}},
$$

其中 $\mu_{A,j}$、$\sigma_{A,j}$ 为 homA 实体上 $z_{\cdot,j}$ 的均值与标准差，$\mu_{B,j}$、$\sigma_{B,j}$ 同理。$d'_{j} > 0$ 的神经元在 $A$ 上更活跃，称为出现选择性（A-selective）神经元；$d'_{j} < 0$ 的神经元在 $B$ 上更活跃，称为 B 选择性神经元。若特征检测器只对出现的特征激活，A 选择性神经元的数量与强度应明显超过 B 选择性神经元。

目标的局部信号以目标相对同图干扰项的标准化偏差衡量：

$$
\delta_{j} = \frac{z_{t,j} - \operatorname{mean}_{e \neq t} z_{e,j}}{\operatorname{std}_{e \neq t} z_{e,j}}.
$$

该量在 std 与 swap 中的大小直接比较“出现”与“缺失”的局部信号。分母为干扰项的离散程度：若 $A$ 实体的激活本身离散，swap 中缺失的目标相对干扰项的偏差就会被稀释，这对应 Rosenholtz 的显著性解释。

### 4.3 目标显著性

在每个提取位置，以干扰项状态的分布估计目标的 Mahalanobis 距离 $m$。35 个干扰项不足以估计完整协方差，因此协方差取干扰项前 8 个主成分加各向同性 ridge（平均特征值的 0.1 倍）。$m$ 在 std 与 swap 中的比值衡量不对称是否已存在于该位置的表征中。

### 4.4 对称的神经元编辑

这是检验“出现检测器”假说的关键实验。在 homB 图像中，把某个实体 16 个 patch 上 A 选择性神经元（每个 block 选 $d'$ 最大的 $K=32$ 个）的激活设为 homA 在同一 patch 上的值，称为注入 A；在 homA 图像中把同一组神经元设为 homB 的值，称为删除 A。两种编辑改变的是同一组神经元、同样大小的激活差，只是方向相反。对 B 选择性神经元做同样的注入与删除；随机选取的 32 个神经元与全部神经元的替换作对照。编辑同时作用于 ViT 第 3、6、9、12 个 block。

读出为 teacher forcing 下答案 $\{\text{"row"}: r, \text{"column"}: c\}$ 中被编辑实体的行列数字：命中定义为行数字与列数字的 argmax 均为该实体（列数字在给定正确行数字的条件下读取），并记录对数概率之和 $S$。若“出现”信号本身就是异常的来源，注入 A 应能让模型指向被编辑实体，删除 A 的效果应弱得多；若注入 B 同样有效，则决定可检测性的是神经元是否激活，与哪个特征被称为“出现”无关。

### 4.5 token mixer 的输出替换

对答对的 std 与 swap 图像，以 homogeneous 图像为供体替换某一层 token mixer 在目标 token 上的输出。供体选择“与目标相同”的均匀图像（std 用 homA，swap 用 homB）：该图像在目标位置有相同的像素与位置，只是上下文与目标一致，因此替换去掉的正是该层 mixer 从上下文向目标写入的比较信息，目标自身的信息保留。替换范围：

1. decoder 第 1–16 层逐层替换，以及按层类型成组替换：DeltaNet 第 1–8 层（第 1、2、3、5、6、7 层）、full attention 第 4、8 层、DeltaNet 第 9–16 层、full attention 第 12、16 层。
2. ViT 的 attention 输出在目标 patch 上按 block 组替换（1–6、7–12、13–18、19–27、全部）。
3. 对照：把干扰项 token 上的 mixer 输出替换为无异常图像（与干扰项相同的均匀图像）的值，检验干扰项是否需要从目标处获得信息。

保留比例沿用实验 2.6 的定义：

$$
R = \frac{\sum_{i}\left(S_{\mathrm{patch}, i} - S_{\mathrm{base}, i}\right)}{\sum_{i}\left(S_{\mathrm{clean}, i} - S_{\mathrm{base}, i}\right)},
$$

其中 $S_{\mathrm{clean}}$ 为未干预图像的答案分数，$S_{\mathrm{base}}$ 为同一答案在“与干扰项相同”的均匀图像上的分数。$R=1$ 表示干预不影响答案，$R=0$ 表示答案退回无异常水平。

### 4.6 Gated DeltaNet 门控编辑

Gated DeltaNet 第 $t$ 个 token 的状态更新可写为

$$
S_{t} = \alpha_{t} S_{t-1}\left(I - \beta_{t} k_{t} k_{t}^{\top}\right) + \beta_{t} v_{t} k_{t}^{\top}, \qquad o_{t} = S_{t} q_{t},
$$

其中 $S_{t} \in \mathbb{R}^{d_v \times d_k}$ 为每个 head 的状态矩阵，$\alpha_{t} \in (0, 1)$ 为衰减门，$\beta_{t} \in (0, 1)$ 为写入强度，$k_{t}$、$v_{t}$、$q_{t}$ 为 key、value、query。目标 token 只能经由 $S_{t-1}$ 读取光栅顺序在前的 token。沿用实验 3.10 的门控编辑代码：

1. **目标处清零（$\alpha_t=0$）。** 在目标 token 上令 $\alpha_t = 0$，目标在该层只能读到自己的写入，切断它经 DeltaNet 读取之前 token 的通路。分别作用于 DeltaNet 第 1–8 层、第 9–16 层与第 1–16 层；对照为对一个光栅位置相近的干扰项做同样编辑。
2. **之前实体不写入（$\beta=0$）。** 在光栅顺序位于目标之前的全部实体 token 上令 $\beta_t = 0$，使它们不进入状态；对照为目标之后的实体。
3. **组合。** 目标处清零（DeltaNet 第 1–16 层）同时替换 full attention 第 4–16 层在目标上的输出，检验两类层合起来是否完成全部 decoder 比较。

### 4.7 attention 边

在 full attention 第 4、8、12、16、20 层重算 softmax 概率，统计以目标 token 为 query 时落在光栅顺序之前的实体、背景 token 与文本 token 上的质量，以及答案行列数字位置对目标 token 的质量，按 head 取平均与最大值，比较 std 与 swap。

## 5. 预期与判据

1. 若 ViT 的特征检测器以“出现”编码，A 选择性神经元多于 B 选择性神经元，std 中目标的局部偏差 $\delta$ 远大于 swap 中的偏差，且注入 A 的命中率远高于删除 A。
2. 若 decoder 中“缺失”目标的比较由 Gated DeltaNet 完成，目标处 $\alpha_t=0$ 与之前实体 $\beta=0$ 应主要降低 swap 的 $R$，对 std 影响小；若由 full attention 完成，第 4、8 层 mixer 输出替换应主要降低 swap 的 $R$。
3. 若合取搜索为串行式，合取条件的准确率随实体数量下降，特征搜索对照不下降。

## 6. 操作

1. 入口：`scripts/analysis/representation/search_asymmetry.sh`，阶段 build、behave、spotcheck、capture、select、edit、mech、analyze。
2. 代码：`analysis_tools/representation/search_asymmetry.py`（刺激与指标）、`search_asymmetry_run.py`（GPU 阶段）、`search_asymmetry_analysis.py`（分析与图）；测试为 `tests/analysis/representation/test_search_asymmetry.py`。
3. 产物：WORK_DIR 的 `analysis/representation/2026-10-21_search_asymmetry/`。
