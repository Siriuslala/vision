# 实验 2.2：位置相关成分与高范数 token

本实验属于[视觉细节表征](plan.md)，承接实验 2.1 中关于固定高分位置与高范数 token 的结果。2.1 发现，相同像素内容位于不同位置时得到不同的表征，这一位置相关成分足以主导手工单图距离读出；同时，少数 token 的范数远高于其余 token，且其位置在 decoder 末层与图像内容无关。本实验研究这一反常现象的三个方面：位置相关成分与高范数 token 来自何处（来源），包含什么信息（信息），对模型回答起什么作用（功能）。

## 1. 研究问题与动机

### 1.1 现有证据

现有证据来自 2026-09-30 与 2026-10-01 的两组实验（见[固定高分位置](results_2026-09-30_pilot.md)、[距离指标与高范数 token](results_2026-09-30.md)、[图像布局与高范数排除](results_2026-10-01.md)），可以归纳为四项：

1. **跨位置差异产生于位置编码与 attention 之后。** 在未加位置编码的 patch embedding 上，正常图各实例的分数全部为零；从 ViT 第 3 层起，正常图已经存在固定最高分位置（6×6 小人图的第 2 行第 6 列），异常图与正常图的实例分数相关系数为 0.986–0.996。该结果只能说明差异产生于加入 learned absolute position embedding 与若干层 attention 之后，无法区分 learned absolute position embedding、ViT attention 中的 2D RoPE 与有限画布边界造成的上下文差异三者各自的贡献。
2. **merger 上的左上偏好并非纯位置效应。** 标准网格与错列布局的 15 张正常图在 merger 输出上均以左上实例为最高分；环形布局中最高分实例随图案变化。若该成分只由二维位置决定，则环形布局中也应出现与图案无关的固定峰。
3. **decoder 末层的范数峰与内容无关。** decoder 第 32 层的全图范数峰在 18 张正常图中均位于第一行第 1 或第 2 个 token；在环形布局中这些位置为空白背景。该峰与图像内容无关，而与视觉 token 序列的起点重合。
4. **decoder 中两种位置因素混淆。** Qwen3.5 的 ViT 使用非因果 attention，按每张图像的 cu_seqlens 划分 attention 范围，token 按 2×2 block-major 顺序排列；decoder 使用 causal 计算，merger 输出的第一个 visual token 恰好对应图像左上角。因此，在 decoder 中，“序列中第一个 visual token”与“图像二维左上位置”是同一个 token，现有数据无法区分二者。

此外，高范数 token 对手工读出的影响依赖层：在 ViT 第 18 层排除高范数 token 将正确定位数从 15/288 提高到 89/288，而在 merger 上从 66/288 降至 61/288。这说明高范数 token 在不同层可能携带不同信息。现有实验均只修改外部分析器读取的特征，没有把修改送回网络，因此不涉及高范数 token 的功能。

### 1.2 相关文献与候选机制

已有文献报告了若干相关现象，但没有文献直接报告“固定左上角”这一现象：

1. Vision Transformers Need Registers（Darcet et al., 2024, arXiv:2309.16588）发现，DINOv2 等 ViT 中约 2% 的 patch token 范数约为其余 token 的 10 倍。这些 token 出现在冗余的背景 patch 上，其位置随图像而变；它们的局部信息较弱（位置预测与像素重建误差较大），全局信息较强（以其为输入的图像分类准确率较高）。加入可学习的 register tokens 后该现象消失。
2. Massive Activations in Large Language Models（Sun et al., 2024, arXiv:2402.17762）发现，LLM 中起始 token 与分隔符上存在固定维度的巨大激活，其数值几乎与输入无关，作用类似 bias，并吸引大量 attention。
3. Efficient Streaming Language Models with Attention Sinks（Xiao et al., 2024, arXiv:2309.17453）发现，序列起始 token 接收大量 attention，移除这些 token 会使模型性能显著下降。
4. Vision Transformers Don't Need Trained Registers（Jiang et al., 2025, arXiv:2506.08010）发现，少数 register neurons 产生高范数 token，在测试时把这些 neurons 的激活转移到额外添加的 token 上，可以在不训练的条件下消除高范数伪影。
5. Denoising Vision Transformers（Yang et al., 2024, arXiv:2401.02957）发现，ViT 特征中存在跨图像固定、与内容无关的伪影，并将其来源归于位置编码。
6. See What You Are Told: Visual Attention Sink in Large Multimodal Models（Kang et al., 2025, arXiv:2503.03321）发现，在 Qwen2-VL-7B 等模型的 LLM 层中，部分 visual tokens 与 BOS token 在相同的少数维度上具有 massive activation，接收大量 attention，而移除它们不影响模型性能。

据此，本实验检验三种候选机制，它们对干预与读出给出不同的预测：

| 机制 | 内容 | 适用部位 |
| --- | --- | --- |
| M1 位置伪影 | 位置编码或有限画布边界在相同内容上叠加随位置变化的成分；按来源分为 M1a learned absolute position embedding、M1b 2D RoPE、M1c 边界上下文 | ViT、merger，并可经 merger 传入 decoder |
| M2 序列起点 sink | 序列中某一段的第一个 token 在固定的少数通道上形成 massive activation，接收大量 attention | 仅 decoder |
| M3 register 挪用 | 低信息的背景 token 被网络挪用来存储全局信息，范数因此升高 | ViT、merger、decoder |

### 1.3 研究问题

1. **成分大小。** 在各提取位置，与内容无关的位置均值成分占特征总方差的比例是多少；高范数 token 出现在哪些位置，是否集中于少数固定通道。
2. **来源。** 对位置编码、画布边界、图像内容位置、序列顺序与 position IDs 分别施加干预后，固定最高分位置、高范数位置与位置成分比例是否随之改变。
3. **信息。** 与匹配的普通 token 相比，高范数 token 所含的局部信息、位置信息与全局信息是否不同。
4. **功能。** 在完整模型中替换高范数 token 的 hidden states 或阻断对它们的 attention，是否改变实验 1.3 子任务的准确率与一般任务的 loss。

## 2. 总体设计

本实验分为四个子实验，前三个只需要冻结模型的前向计算，第四个需要在完整模型中施加干预：

| 子实验 | 内容 | 回答的问题 | 阶段 |
| --- | --- | --- | --- |
| 2.2a 成分分解 | 在大量不同内容的图像上估计位置均值成分与高范数分布 | 成分大小 | 第二阶段 |
| 2.2b 来源干预 | 对 ViT 与 decoder 中的位置因素分别干预，观察峰与成分是否移动 | 来源 | 第二阶段 |
| 2.2c 信息内容 | 在高范数 token 与匹配的普通 token 上训练相同的 probe | 信息 | 第二阶段 |
| 2.2d 功能消融 | 测量 attention sink，并在完整模型中消融高范数 token | 功能 | 第三阶段 |

机制的判定以三类证据一致为前提：来源干预确定成分随哪一因素移动，信息读出确定其存储内容，功能消融确定模型是否依赖它。

## 3. 数据

1. **合成网格图。** 取自[共享合成刺激集](../stimuli.md)的正常图（不含异常实例），四个开发模板，网格规模 3×3、6×6、10×10、20×20，含相位偏移与抖动布局。网格图中实例内容随位置周期重复，用于与 2.1 衔接。
2. **随机放置图。** 在固定画布上随机放置随机数量（4 至 16 个）、随机模板的实例，使每个位置上的内容在图像之间独立变化，从而降低内容与位置的相关性。
3. **自然图像。** RefCOCO validation 图像按 SHA-256 去重后缩放并裁剪到固定尺寸（448×448 与 768×768 两组），每组至少 1000 张。
4. **无内容图像。** 纯白、纯黑、纯灰与 Gaussian 噪声画布，尺寸与上述各组一致。纯色图像中不同位置的 patch 像素完全相同，因此任何跨位置差异只能来自位置编码与边界。

decoder 特征使用与实验 2.3 相同的前缀模板：图像位于文字“Describe the image.”之前，只运行 prefill。成分分解要求同一组图像具有相同的 token 网格，因此所有统计按画布尺寸分组计算，不同尺寸的组之间只比较无量纲比例。

## 4. 方法

### 4.1 符号约定

对固定的提取位置 $\ell$（patch embedding、ViT 第 $k$ 层、merger 或 decoder 第 $k$ 层）与固定画布尺寸的一组图像，约定：

- $N$：组内图像数量；$P$：每张图像在该提取位置的 token 数量；$D$：特征维度（ViT 为 1152，merger 与 decoder 为 4096）。
- $x_{n,p} \in \mathbb{R}^{D}$：第 $n$ 张图像在空间位置 $p$ 的特征向量，$p$ 按二维栅格顺序编号，未合并层先从 2×2 block-major 顺序还原到空间网格。
- $\mu_p \in \mathbb{R}^{D}$：位置 $p$ 的跨图像均值；$\bar{x} \in \mathbb{R}^{D}$：组内全部 token 的均值。

### 4.2 位置均值分解（2.2a）

每个特征向量分解为位置均值与残差：

$$
x_{n,p} = \mu_p + r_{n,p}, \qquad \mu_p = \frac{1}{N}\sum_{n=1}^{N} x_{n,p}.
$$

其中，$\mu_p$ 表示在位置 $p$ 上与图像内容无关的平均成分，$r_{n,p} \in \mathbb{R}^{D}$ 是第 $n$ 张图像在该位置的残差。当组内图像内容在各位置上独立变化时，内容的影响在跨图像平均中相互抵消，$\mu_p$ 主要反映位置编码、边界与序列位置等固定因素。网格图的内容随位置周期重复，其 $\mu_p$ 同时包含内容成分，因此主要结论以随机放置图与自然图像为准。

位置成分的大小以位置可解释方差比例衡量。为避免有限样本下均值估计带有的正偏差，将图像随机分成两半 $A$ 与 $B$，用 $A$ 估计 $\mu_p^{A}$ 与 $\bar{x}^{A}$，在 $B$ 上计算：

$$
R^2_{\mathrm{pos}} = 1 - \frac{\sum_{n \in B}\sum_{p=1}^{P} \lVert x_{n,p} - \mu^{A}_p \rVert_2^2}{\sum_{n \in B}\sum_{p=1}^{P} \lVert x_{n,p} - \bar{x}^{A} \rVert_2^2}.
$$

分母是以全局均值预测每个 token 的误差，分子是以位置均值预测的误差。该比例对应单因素 ANOVA 中位置间方差占总方差的比例：若各位置均值相同，则 $R^2_{\mathrm{pos}}$ 在零附近；若特征几乎完全由位置决定，则 $R^2_{\mathrm{pos}}$ 接近 1。交叉估计使无位置效应时的期望值不高于零。报告两半交换后的均值，并以图像为单位做 bootstrap 95% 区间。

除逐层曲线外，另计算两项辅助统计：

1. **位置均值的空间结构。** 对 $\{\mu_p\}_{p=1}^{P}$ 做 PCA，将前三个主成分在空间网格上绘制为 RGB 图，观察其是否呈现左上到右下的梯度、边界环或单点峰。
2. **残差上的单图读出。** 在 2.1 的网格图上，以 $x_{n,p} - \mu_p$ 代替 $x_{n,p}$ 重新计算手工距离读出，其中 $\mu_p$ 由随机放置图估计。若定位准确率显著回升，说明 2.1 的失败主要来自本节估计的位置成分。

### 4.3 高范数 token 与 massive 维度（2.2a）

第 $n$ 张图像的位置 $p$ 被标记为高范数 token，当且仅当

$$
h_{n,p} = \mathbb{1}\left[ \lVert x_{n,p} \rVert_2 > \tau \cdot \operatorname{median}_{q} \lVert x_{n,q} \rVert_2 \right],
$$

其中 $\tau$ 取 3，与 2026-09-30 统计外部图像时使用的阈值一致，另以 $\tau = 5$ 检查结论对阈值的稳健性。以图像内中位数为基准，可以消除不同图像整体范数尺度的差异。各位置的高范数频率为

$$
f_p = \frac{1}{N}\sum_{n=1}^{N} h_{n,p},
$$

在空间网格上绘制为热图。若 $f_p$ 集中于少数固定位置，说明高范数与位置有关（M1 或 M2）；若 $f_p$ 分散且与背景区域重合，则与 M3 的预测一致。

massive 维度的检查方法为：对每个高范数 token，记录绝对值最大的 $k = 10$ 个通道，统计各通道在全部高范数 token 中出现的频率。若少数通道在大多数高范数 token 中都出现，且这些通道在普通 token 上的绝对值接近零，则与 Sun et al. 及 Kang et al. 报告的固定维度 massive activation 相符。decoder 中另将这些通道与 BOS token、chat template 中 vision start token 的 massive 通道比较。

### 4.4 来源干预（2.2b）

每项干预只改变一个位置因素。以下表格给出干预内容与三种机制的预测，“峰”指正常网格图上的固定最高分实例与全图范数峰。

| 编号 | 部位 | 干预 | M1 预测 | M2 预测 | M3 预测 |
| --- | --- | --- | --- | --- | --- |
| i | ViT | 将 learned absolute position embedding 网格水平翻转或循环平移后插值，或整体置零 | M1a：峰随网格变换移动；置零后跨位置差异显著减小 | ViT 中不适用 | 峰仍位于背景 token，位置随内容而定 |
| ii | ViT | 将 2D RoPE 的旋转置为恒等 | M1b：峰改变或消失 | ViT 中不适用 | 高范数仍出现于背景 token |
| iii | ViT | 将相同内容嵌入更大画布的不同偏移处 | M1a：峰停留在画布固定位置；M1c：峰随内容到边界的距离变化 | ViT 中不适用 | 峰位于背景区域，随空白分布变化 |
| iv | ViT | 将图像水平或垂直翻转，序列顺序与位置编码不变 | 峰停留在原二维位置 | decoder 范数峰停留在第一个 visual token | 峰随背景内容翻转 |
| v | decoder | 在图像之前插入不同长度的文本前缀 | 峰位置不变 | 若 sink 依附于视觉段起点，峰仍在第一个 visual token；若依附于全序列起点，峰转移到前缀 token | 峰位置不变 |
| vi | decoder | 将视觉 token 的 MRoPE position IDs 整体平移 | 峰位置不变 | 若 sink 由 position IDs 决定，峰改变；若由序列顺序决定，峰不变 | 峰位置不变 |
| vii | decoder | 反转视觉 token 的输入顺序，保持各 token 的 position IDs | 峰停留在左上 token | 若 sink 由序列顺序决定，峰转移到新的第一个 token（右下角） | 峰随内容而定 |

干预 iv 的作用是分离“内容所在的二维位置”与“token 所在的位置编码”：翻转后左上角的像素内容移到右下角，而左上角 token 仍接收相同的位置编码。干预 vii 将输入置于训练分布之外，Gated DeltaNet 层的递推顺序也随之反转，因此其结论只作为辅助证据，需要与 v、vi 一致才予采用。

实施干预时需要考虑 decoder 的结构。Qwen3.5-9B 的 32 个 decoder 层中，只有第 4、8、12、16、20、24、28、32 层这 8 个 full attention 层（从 0 起的 block 索引为 3、7、…、31）使用 MRoPE；每个 attention head 的 256 维中只有 64 维参与旋转（partial_rotary_factor 为 0.25，mrope_section 为 [11, 11, 10]，分别对应时间、高度与宽度三个轴的频率数）。其余 24 个 Gated DeltaNet 层没有位置编码，位置信息只能来自 causal convolution 与递推顺序。因此，干预 vi 只影响 8 个 full attention 层中 64 维的旋转，而干预 v 与 vii 同时影响全部层的序列顺序。

每项干预测量三个量：正常网格图上固定最高分实例的位置（沿用 2.1 的实例间平均 RMS 分数，仅作描述统计）、高范数频率热图 $f_p$、以及 $R^2_{\mathrm{pos}}$。干预前后的比较在同一批图像上进行。

### 4.5 信息内容（2.2c）

本节检验 M3 的核心预测：高范数 token 的局部信息较弱，全局信息较强。比较对象为高范数 token 与两类匹配的普通 token：

1. **内容匹配 token。** 同一图像中、不属于高范数的 token，其对应 32×32 像素区域的像素方差与高范数 token 最接近，通常同为背景。
2. **位置匹配 token。** 另一张同组图像中相同位置 $p$ 上的 token，且该 token 在该图像中不属于高范数。

内容匹配控制局部像素，位置匹配控制位置编码。三组 token 的数量相同，每张图像最多取相同数量的 token，避免少数图像主导结果。在三组 token 上分别训练相同结构的 probe（线性 probe 为主，一层隐藏层 MLP 作容量对照）：

| 信息类别 | 探测目标 | probe 输出与指标 |
| --- | --- | --- |
| 局部 | 该 token 对应 32×32 像素区域（下采样到 8×8 灰度） | 线性回归，测试集 $R^2$ |
| 局部 | 区域内是否含笔画、笔画颜色 | 分类，balanced accuracy |
| 位置 | token 的行与列 | 分类，exact match 与平均行列误差 |
| 全局 | 图像模板类别或 RefCOCO 图像中面积最大物体的类别 | 分类，macro-F1 |
| 全局 | 网格规模（实例数量） | 分类，balanced accuracy |
| 全局 | 图中是否含异常实例、异常属性类型 | 分类，AUROC 与 macro-F1 |

高范数 token 的大范数可能使线性 probe 的优化与正则化不同于普通 token。因此每个 probe 在两种输入上训练：原始特征与逐 token L2 归一化后的特征。M3 的预测为：高范数 token 的局部与位置读出低于两类匹配 token，全局读出高于两类匹配 token，且该差异在归一化后仍然存在。

### 4.6 功能（2.2d）

**Attention sink 测量。** 对每个 attention 层与 head，计算高范数 token 集合 $S$ 接收的 attention 富集比：

$$
e = \frac{\frac{1}{|Q|}\sum_{i \in Q}\sum_{j \in S} a_{ij}}{|S| \,/\, |K|},
$$

其中 $Q$ 是 query token 集合，$K$ 是可见 key token 集合，$a_{ij}$ 是 query $i$ 对 key $j$ 的 attention 权重。分子是集合 $S$ 平均获得的 attention 质量，分母是 $S$ 在可见 token 中所占的比例；$e = 1$ 表示 attention 与 token 数量成比例，$e \gg 1$ 表示 $S$ 是 attention sink。ViT 的全部层均可测量；decoder 只有 8 个 full attention 层具有显式 attention 矩阵，测量两类 query：其余 visual tokens，以及问题文本 token 与最后一个 prompt token。

**完整模型消融。** 在第 $\ell$ 层施加两种干预，并以相同数量、随机选择的普通 token 施加相同干预作为对照（5 个随机种子）：

1. **均值替换。** 把高范数 token 的 hidden state 替换为同一图像中普通 token 的均值，然后继续后续计算。
2. **attention knockout。** 在 ViT 各层与 decoder 的 full attention 层中，将后续 token 对高范数 token 的 attention logit 置为负无穷。Gated DeltaNet 层没有 attention 矩阵，不施加 knockout，因此该干预只阻断 full attention 层的读取路径，在文档中按此限定解释。

测量指标为：实验 1.3 的 T3、T5、T6 准确率与 AUROC（使用模型能够答对的条件），RefCOCO bbox 输出的 IoU≥0.5 比例，以及 COCO caption 在 teacher forcing 下的平均 token loss。若高范数 token 是与 Kang et al. 报告相同的无用 sink，消融不应改变任务指标；若其承担 register 功能，均值替换会损害依赖全局信息的任务。

**测试时 register 转移。** 参照 Jiang et al. 的方法，在 ViT 中定位激活与高范数位置共现的 register neurons，将其激活转移到额外附加的 token 上。该额外 token 需要在 merger 之前移除，以保持 2×2 合并的空间结构。比较转移前后的高范数分布、2.3 的属性 probe 读出与上述任务指标，可以检验移除伪影是否改善细节读出。

## 5. 指标

| 指标 | 定义 | 子实验 |
| --- | --- | --- |
| $R^2_{\mathrm{pos}}$ | 交叉估计的位置可解释方差比例 | 2.2a、2.2b |
| 高范数频率 $f_p$ | 各位置成为高范数 token 的图像比例 | 2.2a、2.2b |
| massive 通道重合度 | 高范数 token 的 top-10 通道集合在图像间的平均 Jaccard 系数 | 2.2a |
| 峰位置移动 | 干预前后固定最高分实例与范数峰的位置是否改变，以干预前后的行列差表示 | 2.2b |
| probe 读出差 | 高范数 token 与匹配 token 的 probe 指标之差 | 2.2c |
| 富集比 $e$ | 高范数 token 获得的 attention 质量与其数量占比之比 | 2.2d |
| 任务指标变化 | 消融相对未干预的准确率、AUROC、IoU 比例与 loss 变化，减去随机对照的变化 | 2.2d |

比例给出 Wilson 95% 区间，连续指标给出以图像为单位的 bootstrap 95% 区间。

## 6. 可视化

1. $R^2_{\mathrm{pos}}$ 随提取位置变化的曲线，每类数据（随机放置图、自然图像、纯色图像）一条。
2. 各提取位置的 $f_p$ 热图与 $\mu_p$ 的 PCA 伪彩色图，按层排成网格。
3. massive 通道频率条形图，decoder 中标出 BOS 与 vision start token 的 massive 通道。
4. 每项来源干预的前后对照：输入图、干预前与干预后的范数图与实例分数图并列，标出峰位置。
5. 2.2c 的 probe 指标分组条形图，三组 token 并列，按信息类别分面。
6. 2.2d 的富集比热图（层 × head）与消融效应的森林图（点估计与 95% 区间，随机对照并列）。

## 7. 结果模式与解释

| 结果模式 | 解释 |
| --- | --- |
| 纯色图像上 $R^2_{\mathrm{pos}}$ 很高，干预 i 使峰随位置编码网格移动 | 位置相关成分主要来自 learned absolute position embedding（M1a） |
| 干预 iii 中峰随内容到边界的距离移动，干预 i 无效 | 边界上下文是主要来源（M1c） |
| 干预 v 后 decoder 范数峰仍在第一个 visual token，干预 vii 后峰转到新的第一个 token，massive 通道与 vision start token 重合 | decoder 末层范数峰是视觉段起点上的 sink（M2） |
| 高范数 token 的 $f_p$ 分散于背景，局部读出弱、全局读出强，均值替换损害全局任务 | 高范数 token 承担 register 功能（M3） |
| 高范数 token 是强 attention sink，但消融不改变任务指标 | 与 Kang et al. 的报告一致，高范数 token 不承载任务所需信息 |
| 残差特征上的单图读出显著回升 | 2.1 中单图读出的失败主要由位置均值成分造成，位置成分可以作为可移除的偏置处理 |

不同部位可以对应不同机制，例如 ViT 中间层符合 M3、decoder 末层符合 M2。报告按提取位置分别给出结论。

## 8. 运行与产物

1. 代码实现为 `analysis_tools/positional_component.py`（2.2a–2.2c）与 `analysis_tools/positional_ablation.py`（2.2d），干预使用 PyTorch forward hooks 与对 position IDs、position embedding 的显式替换，2.2d 与实验 2.5 共用 hook 工具。
2. Bash 入口为 `scripts/analysis/positional_component.sh` 与 `scripts/analysis/positional_ablation.sh`，提取位置、图像组、阈值 $\tau$ 与随机种子集中于脚本顶部。
3. 产物保存于 WORK_DIR 的 `analysis/representation/<日期>_positional_component/` 与 `analysis/representation/<日期>_positional_ablation/`，包括图像 metadata、逐层统计、干预配置、probe 权重与指标、消融前后的模型输出。

## 9. 解释范围

1. 只有当来源干预、信息读出与功能消融三者给出一致结论时，才以 M1、M2 或 M3 命名某一部位的现象；单项证据只报告为与某机制相容。
2. 干预 i、ii、vii 使模型处于训练分布之外，其效应可能部分来自分布偏移本身，需以干预后模型在一般任务上的 loss 变化作为参照。
3. 高范数阈值是描述性定义，不能单独据此认定 register；register 的判定依赖 2.2c 与 2.2d 的结果。
4. $\mu_p$ 的估计依赖图像分布；在自然图像上，内容与位置存在统计相关（如天空常在上方），因此自然图像的 $R^2_{\mathrm{pos}}$ 包含这一部分，需与随机放置图和纯色图像的结果对照解释。
