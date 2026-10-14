# 多层视觉特征注入

本问题属于[视觉表征研究计划](../plan.md)的问题 4。Qwen3-VL 通过 DeepStack 将 ViT 中间层特征注入 LLM 前部各层，Qwen3.5 发布时不再使用这一机制。本问题检验 DeepStack 是否为细粒度视觉任务提供了 Qwen3.5 所缺少的局部信息，并从行为、因果消融和表征三个层面确定该机制的实际作用。

## 1. 研究问题与动机

ViT 浅层表征保留的局部细节通常多于末层表征，随着 ViT 与 LLM backbone 逐层处理，视觉表征中的 high-level 语义与全局信息增多，low-level 局部特征逐渐减少。DeepStack 的设计目的正是把这类细节重新送入 LLM。小物体识别、BabyVision 的找不同与细粒度辨别等任务依赖局部细节，因此需要明确 Qwen3.5 移除 DeepStack 是否削弱了这类能力。

已有观察为本问题提供了起点。在完全相同的 108 张异常网格图像上，merger 输出的未排除 RMS 单图异常定位中，Qwen3-VL-8B-Instruct 正确 86 张，Qwen3.5-9B 正确 53 张（见[图像布局与高范数排除](../representation/results_2026-10-01.md)第 5 节）。由于两模型的 ViT 权重、merger、训练数据与 decoder 结构同时不同，该差异不能单独归因于 DeepStack。

本问题依次回答以下四个子问题：

1. **背景（4.1）。** DeepStack 的原始设计、Qwen3-VL 中的实现与官方消融结果是什么；Qwen3.5 移除它的公开说明有哪些。
2. **端到端差异（4.2）。** 在相同刺激与问题上，具有 DeepStack 的 Qwen3-VL 与不具有 DeepStack 的 Qwen3.5 在细粒度任务上的差异有多大，差异是否随模型规模保持一致。
3. **因果依赖（4.3）。** 在 Qwen3-VL 内部关闭或替换 DeepStack 注入后，细粒度任务是否下降，而一般能力任务是否保持不变。
4. **表征来源（4.4）。** DeepStack 注入的特征是否携带主路径视觉 token 所缺少的细节；Qwen3.5 的 ViT 末层是否已经保留了与 Qwen3-VL 浅层相当的细节。

## 2. 背景：DeepStack 与 Qwen3.5 的取舍（实验 4.1，已完成）

### 2.1 原始 DeepStack

DeepStack 由 Meng et al.（2024，arXiv:2406.04334）提出。原设计将高分辨率图像切块得到的额外视觉 token 分组，以残差相加的方式逐层加入 LLM 前部的连续若干层，同时不增加输入序列长度。原论文的主要结论如下：

1. 在相同 context 长度下，7B 与 13B 模型的平均性能分别提升 2.7 与 2.9，增益集中于 TextVQA、DocVQA、InfoVQA 等需要高分辨率与细粒度信息的任务。
2. 注入起始层超过 LLM 深度的中点后，性能明显下降，说明注入位置需要靠近 LLM 前部。
3. 将原始低分辨率 token 重复堆叠到各层不带来增益，说明增益来自注入 token 所携带的新增细节信息。

原始 DeepStack 注入的特征来源为高分辨率切块 token；以多层 ViT 特征为注入来源的形式由 Qwen3-VL 引入。

### 2.2 Qwen3-VL 中的 DeepStack

Qwen3-VL Technical Report（arXiv:2511.21631）将 DeepStack 改为从 ViT 的三个层级抽取特征，各层特征经过独立的 merger 投影后加到 LLM 前三层的视觉 token 位置。报告中的消融使用内部 15B-A2B 模型，预训练 200B tokens，不做 post-training，结果如下表。

| 评测 | 无 DeepStack | 有 DeepStack |
| --- | ---: | ---: |
| 平均 | 74.7 | 76.0 |
| InfoVQA | 71.9 | 74.2 |
| DocVQA | 89.5 | 91.1 |
| OCRBench | 81.0 | 83.6 |
| ChartQA | 81.5 | 83.3 |
| AI2D | 81.8 | 83.2 |
| MMStar | 55.5 | 57.7 |
| MMMU | 52.9 | 54.1 |
| TextVQA | 80.6 | 80.5 |

增益集中于文档、图表与 OCR 类任务，TextVQA 没有变化。该消融不包含 post-training，也不包含 BabyVision 类的结构化视觉推理任务。

### 2.3 Qwen3.5 移除 DeepStack 的公开信息

Qwen3.5 的 model card 未提及 DeepStack，目前没有找到官方对移除原因的说明。sglang PR #33726 写明 Qwen3.5 的发布配置为 deepstack_visual_indexes 等于空列表，本地 Qwen3.5-9B 与 Qwen3.5-4B 的配置和 modeling 代码也不含 DeepStack 路径。社区中存在两类推测，均没有消融实验支撑：

1. Qwen3.5 采用 early fusion 的联合训练，LLM 的训练 loss 直接塑造 ViT 末层的编码，使末层已经保留足够细节，DeepStack 成为不必要的补偿机制。
2. 移除 DeepStack 出于参数量、实现复杂度或训练稳定性的考虑。

第一类推测给出了可检验的预测：若其成立，Qwen3.5 ViT 末层的细节可读性应当接近 Qwen3-VL ViT 浅层的细节可读性。实验 4.4 直接检验这一预测。

### 2.4 其他多层视觉特征融合工作

1. **Dense Connector**（Yao et al., NeurIPS 2024，arXiv:2405.13800）将多层 ViT 特征在 connector 中拼接或相加后输入 LLM，以较小代价提升多项评测。
2. **MMFuser**（Cao et al., 2024，arXiv:2410.11829）以深层特征为 query，从浅层特征中提取细粒度信息，并指出浅层特征包含丰富的细粒度信息。
3. **Lin et al.（CVPR 2025，arXiv:2503.06063）** 系统比较了多层视觉特征的选择与融合位置，发现来自不同 stage 的层均有益，且在输入端融合比在 LLM 中间层注入更稳定。

这些工作共同支持“浅层特征含有末层所缺少的细节”这一判断，但它们的评测以通用 VQA 为主，没有回答 LLM 是否在细粒度视觉推理中实际使用了注入的特征。

## 3. 总体设计

本问题的三个实验按证据强度互补：4.2 给出两类模型的行为差异，但混杂了训练数据、LLM 结构与 post-training 的差别；4.3 在同一个 Qwen3-VL 模型内部改变 DeepStack 注入，是归因最直接的证据；4.4 检验注入特征与主路径特征所含信息的差别，解释 4.3 中观察到的依赖来自何处。

| 编号 | 实验 | 比较对象 | 回答的问题 | 主要混杂 |
| --- | --- | --- | --- | --- |
| 4.2 | 端到端对照 | Qwen3-VL-8B 对 Qwen3.5-9B；Qwen3-VL-4B 对 Qwen3.5-4B | 两类模型在细粒度任务上的差异 | 训练数据、LLM 结构、post-training |
| 4.3 | DeepStack 消融 | Qwen3-VL 的完整模型与多种注入修改 | Qwen3-VL 的回答是否依赖注入特征 | 关闭注入属于分布外操作 |
| 4.4 | 表征对照 | 两模型的 ViT 各层、merger、decoder 前部各层 | 注入特征携带了哪些主路径缺少的细节 | 外部读出器只说明信息可恢复 |

所有实验使用冻结模型，thinking off。合成刺激来自[共享合成刺激集](../stimuli.md)，子任务定义沿用[能力分解](../behavior/design_decomposition.md)。

## 4. DeepStack 的形式化描述

本节约定后文使用的符号，并以 Qwen3-VL-8B-Instruct 的配置为例说明注入的计算过程。

### 符号约定

| 符号 | 形状 | 含义 |
| --- | --- | --- |
| $N_p$ | 标量 | 一张图像经 processor 得到的 patch 数，每个 patch 为 16×16 pixels |
| $N$ | 标量 | merger 输出的 visual token 数，$N = N_p / 4$ |
| $X^{(l)}$ | $N_p \times 1152$ | ViT 第 $l$ 个 block 的输出，$l$ 以 0 起编号 |
| $L_v$ | 标量 | ViT 的 block 数，8B 模型为 27，4B 模型为 24 |
| $\mathcal{S}$ | 集合 | DeepStack 抽取层的编号，8B 模型为 $\{8, 16, 24\}$，4B 模型为 $\{5, 11, 17\}$ |
| $M_0$ | 映射 | 主路径 merger，作用于 ViT 末层输出 $X^{(L_v - 1)}$ |
| $M_k$ | 映射 | 第 $k$ 个 DeepStack merger，$k = 1, 2, 3$，作用于第 $k$ 个抽取层 |
| $d$ | 标量 | LLM hidden size，8B 模型为 4096，4B 模型为 2560 |
| $V$ | $N \times d$ | 主路径视觉 token，作为 decoder 的输入 embedding |
| $D_k$ | $N \times d$ | 第 $k$ 个抽取层的 DeepStack 特征 |
| $H^{(j)}$ | $L \times d$ | decoder 第 $j$ 层输出的 hidden states，$L$ 为序列长度，$j$ 以 0 起编号 |
| $\Omega$ | 下标集合 | 序列中 visual token 所在的位置 |

层号约定：符号 $X^{(l)}$ 与 $H^{(j)}$ 的上标以 0 起编号，与模型配置中的索引一致，例如 deepstack_visual_indexes 中的 8 表示 ViT block 8；正文以“第 $n$ 层”表述 decoder 层时从 1 开始，与问题 2、3 的文档一致，即 $H^{(j)}$ 是 decoder 第 $j+1$ 层的输出，DeepStack 注入发生在 decoder 第 1–3 层之后。

### 注入计算

每个 merger 先将空间上相邻的 2×2 个 patch 特征拼接为一个 $4 \times 1152 = 4608$ 维向量，经 LayerNorm 与两层 MLP 投影到 $d$ 维。DeepStack merger 的 LayerNorm 作用于拼接后的 4608 维向量（配置项 use_postshuffle_norm 为 True）。记抽取层按升序为 $l_1 < l_2 < l_3$，主路径与 DeepStack 特征分别为

$$
V = M_0\left(X^{(L_v - 1)}\right), \qquad D_k = M_k\left(X^{(l_k)}\right), \quad k = 1, 2, 3.
$$

decoder 以 $V$ 替换图像占位 token 的 embedding。前三层 decoder 的输出在 visual token 位置上加入对应的 DeepStack 特征：

$$
H^{(j)}_{\Omega} \leftarrow H^{(j)}_{\Omega} + D_{j+1}, \qquad j = 0, 1, 2.
$$

其中 $H^{(j)}_{\Omega}$ 表示 $H^{(j)}$ 中位于 visual token 位置的各行。该操作对应 modeling 代码中 hidden_states 在 visual_pos_masks 位置上与 visual_embeds 的残差相加，不改变序列长度与 position IDs。

这一形式的原理如下。残差流中各层的输出以相加的方式累积，因此加入 $D_k$ 等价于在不改变 attention 结构的前提下，为每个 visual token 附加一项来自 ViT 较浅层的信息；浅层 $D_1$ 注入最早，深层 $D_3$ 注入最晚，使 LLM 前三层依次接收由浅到深的视觉特征。由于注入发生在 LLM 的最前部，后续所有层都可以通过 attention 读取这些信息，这与原始 DeepStack 中“注入起始层不宜超过中点”的发现一致。

Qwen3.5 的 ViT 规格与 Qwen3-VL 基本相同（8B 对 9B：27 个 block、hidden size 1152、patch 16、spatial merge 2、2304 个 position embedding），只保留主路径 $V$。Qwen3.5-9B 的 decoder 由“3 层 Gated DeltaNet 与 1 层 gated full attention”的单元重复 8 次构成，共 32 层；Qwen3-VL-8B 的 decoder 为 36 层 full attention。

## 5. 实验 4.2：端到端对照

### 5.1 模型与评测

模型分为两组：Qwen3-VL-8B-Instruct 对 Qwen3.5-9B，Qwen3-VL-4B-Instruct 对 Qwen3.5-4B。第二组用于检验结论是否随模型规模保持一致。评测包含三部分：

1. **BabyVision 全集。** 按子类型报告准确率，重点关注 Find the different、Fine-grained Discrimination 与 Visual Tracking。Qwen3.5-9B 在 thinking off 条件下已有结果：总体 17.3%，Find the different 1/16，Fine-grained Discrimination 19.0%，Visual Tracking 10.8%。其余三个模型需要新运行，解码参数采用各模型官方推荐的 thinking off 或 Instruct 设置，与评分规则一并记录。各模型在 BabyVision 上的完整回答均保存，Find the different 与 Fine-grained Discrimination 两个子类型的推理文本按[能力分解](../behavior/design_decomposition.md)第 6.3 节的编码表分析，比较两类模型的失败环节分布。
2. **能力分解子任务。** 使用实验 1.3 的全部子任务 T0–T6，刺激与问题完全相同。二值任务读取首 token 的 Yes 与 No logit，定位任务使用 greedy decoding。
3. **RefCOCO 小目标定位。** 以目标 bbox 面积占图像面积的比例 $a$ 分为三层：$a < 1\%$、$1\% \le a < 5\%$、$a \ge 5\%$。每层从 RefCOCO val 中抽取 300 条表达式，每张图像最多取一条；若某一层不足 300 条，则使用全部可用样本并报告分母。模型输出 0–1000 归一化 bbox，以 IoU≥0.5 的比例为主指标。

### 5.2 指标与统计

1. 比例指标报告分母与 Wilson 95% 区间。同一题目在两模型上的成对结果使用 McNemar 检验，比较两模型在相同题目上答对与答错的不一致对。
2. 实验 1.3 的子任务按网格规模与变化幅度分别报告，并比较二值任务的阈值 $\delta_{75}$，其定义见[能力分解](../behavior/design_decomposition.md)第 5.1 节。
3. RefCOCO 的三层结果用于检验两模型的差异是否随目标变小而增大。设模型 $m$ 在第 $s$ 层的 IoU≥0.5 比例为 $p_{m,s}$，关注的量为差值

$$
\Delta_s = p_{\mathrm{VL},s} - p_{3.5,s},
$$

其中下标 VL 与 3.5 分别表示 Qwen3-VL 与 Qwen3.5。若 DeepStack 主要补充小目标细节，$\Delta_s$ 应随目标面积减小而增大；若 $\Delta_s$ 在三层上相近，则差异更可能来自一般能力的差别。区间由以图像为单位的 bootstrap 给出。

### 5.3 解释范围

两类模型的训练数据、LLM 结构与 post-training 均不同，4.2 中的差异只说明“具有 DeepStack 的模型族在哪些任务上更强”，不能单独归因于 DeepStack。4.2 的作用是确定 4.3 与 4.4 应当集中分析的任务。

## 6. 实验 4.3：Qwen3-VL 内部的 DeepStack 消融

### 6.1 条件

本实验固定 Qwen3-VL-8B-Instruct 的全部权重，只修改第 4 节注入计算中加入 decoder 的 DeepStack 特征。记第 $k$ 个抽取层在条件 $c$ 下实际加入的特征为 $\tilde{D}_k$，各条件定义如下。

| 条件 | 记号 | $\tilde{D}_k$ 的取值 | 目的 |
| --- | --- | --- | --- |
| 完整模型 | Full | $D_k$ | 基线 |
| 全部关闭 | Off | 对全部 $k$ 取零矩阵 | 测量对注入的总体依赖 |
| 单层关闭 | Off-$k$ | 第 $k$ 层取零矩阵，其余保持 $D_k$ | 区分三个抽取层的贡献 |
| 均值替换 | Mean | 对全部 $k$ 取 $\mathbf{1}\bar{d}_k^{\top}$ | 降低分布外程度的对照 |
| 低通替换 | Blur | 对全部 $k$ 取由模糊图像计算的 $D_k$ | 保留注入的分布，移除高频细节 |

其中 $\bar{d}_k \in \mathbb{R}^{d}$ 是第 $k$ 个抽取层的 DeepStack 特征在参考图像集全部 visual token 上的均值，$\mathbf{1}$ 为长度 $N$ 的全 1 列向量，二者之积将同一均值向量放到每个 visual token 位置。参考图像集由 500 张 RefCOCO train 图像与 500 张合成刺激组成，不与评测样本重叠。低通替换中，DeepStack 特征由经过 Gaussian 模糊（标准差为 2 pixels）的同一图像计算，主路径 $V$ 仍由原图计算。

实现上，Off 可以将 model.model.visual.deepstack_visual_indexes 置为空列表，或将 _deepstack_process 替换为恒等映射；Mean 与 Blur 需要在 _deepstack_process 中替换加入的张量。若通过 inputs_embeds 调用 forward，必须显式传入 deepstack_visual_embeds 与 visual_pos_masks，否则 DeepStack 会被静默丢弃。每次运行在首个样本上核对各条件下前三层 decoder 输出的变化，确认修改确实生效。

### 6.2 分布外损伤的控制

Qwen3-VL 以 DeepStack 训练，关闭注入会使 decoder 前三层的 hidden states 偏离训练分布。因此，Off 条件下的性能下降同时包含两部分：模型对注入细节的依赖，以及分布偏移造成的一般性损伤。本实验用两类对照区分二者：

1. **均值替换与低通替换。** Mean 保留注入项的一阶统计量，Blur 进一步保留注入项的整体分布而只移除细节。若 Blur 与 Off 一样使细粒度任务下降，而一般能力任务保持不变，则下降可以归因于细节信息的缺失。
2. **一般能力任务。** RefCOCO 大目标层（$a \ge 5\%$）的定位准确率，以及 COCO caption 上参考描述的 per-token 交叉熵，用于测量一般性损伤。设条件 $c$ 下细粒度任务与一般能力任务相对 Full 的准确率变化分别为 $\Delta^{\mathrm{fine}}_c$ 与 $\Delta^{\mathrm{gen}}_c$，只有当 $\Delta^{\mathrm{fine}}_c$ 显著为负而 $\Delta^{\mathrm{gen}}_c$ 接近零时，才将结果解释为对细节的选择性依赖。

### 6.3 评测与统计

评测使用 4.2 的子集：BabyVision 中 Find the different、Fine-grained Discrimination 与 Visual Tracking 三个子类型；实验 1.3 的 T1、T2、T3、T5、T6，限于缺口与颜色两种变化及 6×6、10×10、20×20 三种网格；RefCOCO 三个面积层。所有条件在相同样本上运行，成对比较使用 McNemar 检验，预先指定的主要对比为 Off 对 Full、Blur 对 Full、Mean 对 Full，多重比较使用 Holm 校正。Qwen3-VL-4B-Instruct 在主要对比上重复，检验结论是否随规模保持一致。

## 7. 实验 4.4：表征对照

### 7.1 设计

本实验在相同合成刺激上，对两模型及 Qwen3-VL 的消融版本提取各层视觉表征，用实验 2.3 的属性 probe 与实验 2.4 的特征反演测量细节可读性。方法细节见[属性 probe 设计](../representation/design_attribute_probe.md)与[可视化设计](../representation/design_visualization.md)。提取位置如下：

| 模型 | 提取位置 |
| --- | --- |
| Qwen3-VL-8B | ViT block 8、16、24、26 的输出；$D_1, D_2, D_3$；$V$；decoder 第 1–4 层注入前与注入后的 hidden states；decoder 第 9、17、25、36 层 |
| Qwen3-VL-8B（Off） | decoder 第 1–4 层及上述较深层 |
| Qwen3.5-9B | ViT block 8、16、24、26 的输出；$V$；decoder 第 1–4 层；decoder 第 9、17、25、32 层 |

### 7.2 指标

1. **细节阈值。** 对每个提取位置拟合 probe 准确率随变化幅度 $\delta$ 的心理测量曲线，取准确率达到 75% 时的幅度 $\delta_{75}$。$\delta_{75}$ 越小，表示该位置可分辨的细节越细。
2. **重建质量。** 特征反演解码器由某一位置的 visual tokens 重建实例图像，以重建图与原图在实例区域内的 SSIM 衡量，并对异常实例单独报告笔画改动区域的重建误差。
3. **互补信息。** 为检验 DeepStack 特征是否携带主路径缺少的细节，比较两个 probe：一个只以 $V$ 为输入，另一个以 $V$ 与 $D_k$ 的拼接为输入。二者的准确率之差记为

$$
G_k = \mathrm{Acc}\left([V; D_k]\right) - \mathrm{Acc}(V),
$$

其中 $[V; D_k]$ 表示对每个 visual token 将两个 $d$ 维向量拼接为 $2d$ 维向量，$\mathrm{Acc}$ 为 probe 在测试图像上的准确率。两个 probe 使用相同的正则化选择流程。$G_k$ 显著大于零，说明 $D_k$ 中存在无法由 $V$ 线性读出的细节；为排除维数增加的影响，另以 $[V; V']$ 为对照，其中 $V'$ 为同一模型对另一张无关图像计算的主路径特征。

### 7.3 关键比较

1. **注入前后的跃升。** 比较 Qwen3-VL decoder 第 1–4 层在注入前与注入后的 $\delta_{75}$。若注入后 $\delta_{75}$ 明显下降，而 Off 版本在同一层没有下降，说明注入使 LLM 前部获得了新的可读细节。
2. **深层的保持。** 比较 Full 与 Off 在 decoder 较深层的 $\delta_{75}$，判断注入的细节能否在 LLM 中保持，还是在若干层后与 Off 版本趋同。
3. **ViT 末层的对照。** 比较 Qwen3.5 ViT 末层（block 26）与 Qwen3-VL ViT 各层（block 8、16、24、26）的 $\delta_{75}$ 与重建质量。该比较直接检验第 2.3 节的 early fusion 推测。

## 8. 结果模式与解释

| 结果模式 | 解释 | 后续 |
| --- | --- | --- |
| 4.3 中 Off 与 Blur 使细粒度任务显著下降，RefCOCO 大目标与 caption 交叉熵基本不变 | DeepStack 提供了被 Qwen3-VL 实际使用的细节信息 | 4.4 定位细节来自哪一抽取层 |
| 4.3 中 Off 使全部任务同等下降，Mean 与 Blur 基本恢复 | 下降主要来自分布外损伤，Qwen3-VL 对注入细节的依赖较弱 | 降低对 4.2 差异的结构归因 |
| 4.3 中 Off-1 的下降大于 Off-3 | 最浅抽取层提供的细节最为关键 | 问题 5 中优先尝试浅层特征重读 |
| 4.4 中 $G_k$ 显著大于零，且注入后 decoder 前部 $\delta_{75}$ 下降 | 注入特征携带主路径缺少的细节，并进入 LLM 表征 | 与 4.3 的行为依赖合并为完整证据 |
| 4.4 中 Qwen3.5 ViT 末层的 $\delta_{75}$ 与 Qwen3-VL 浅层相当 | 支持 early fusion 使 ViT 末层已保留细节的推测 | 4.2 的差异需要由其他因素解释 |
| 4.4 中 Qwen3.5 ViT 末层的 $\delta_{75}$ 明显大于 Qwen3-VL 浅层，且 4.2 中 Qwen3.5 在小目标与细粒度任务上更弱 | Qwen3.5 缺少 DeepStack 提供的细节，移除 DeepStack 带来细粒度能力的损失 | 方向二中考虑轻量的浅层特征注入 |
| 4.4 中细节在 ViT 与 $V$ 中可读，但在 decoder 中迅速下降 | 编码阶段保留了细节，瓶颈位于 LLM 对视觉细节的保持与使用 | 问题 2 的 2.5 因果分析与问题 5 |

上述模式可以同时出现，报告分别给出每一项的证据。

若证据支持 DeepStack 对细粒度任务的必要性，方向二可以考虑两种轻量方案：在 LLM 前部注入浅层特征，或在局部重读时以浅层特征替代主路径特征。后者与[局部重读](../reread/plan.md)中的条件 (g) 直接对应。

## 9. 运行与产物

1. 模型均使用 bfloat16，batch size 1，thinking off。BabyVision 评测沿用仓库现有的评测入口与评分规则，新增模型的解码参数写入运行配置。
2. 端到端对照的实验程序为 analysis_tools/deepstack/deepstack_comparison.py，Bash 入口为 scripts/analysis/deepstack/deepstack_comparison.sh；消融为 analysis_tools/deepstack/deepstack_ablation.py 与 scripts/analysis/deepstack/deepstack_ablation.sh；表征对照复用实验 2.3 与 2.4 的特征提取与 probe 代码，入口为 scripts/analysis/deepstack/deepstack_representation.sh。全部运行参数集中于 Bash 入口顶部。
3. 产物保存于 WORK_DIR 的 analysis/deepstack/ 下，目录命名为“日期_实验名称”，例如 analysis/deepstack/2026-10-20_end_to_end/、analysis/deepstack/2026-10-27_ablation/。每个目录保存模型版本与配置、解码参数、样本列表、全部 prompt 与原始回答、首 token logits、严格评分与诊断评分、均值替换所用参考集的 hash 与均值向量。文档用图复制到 docs/analysis/deepstack/figures/。

## 10. 解释范围与进度

1. 4.2 的模型族差异混杂训练数据、LLM 结构与 post-training，只用于确定分析重点。
2. 4.3 的消融修改的是已训练模型，结果说明 Qwen3-VL 对注入特征的依赖，不能直接推断“以不含 DeepStack 的方式重新训练”会得到什么结果；Qwen3.5 的训练流程可能已经以其他方式补偿了细节。
3. 4.4 的 probe 与反演结果说明信息可以被外部读出器恢复，不能说明 LLM 使用了这些信息；使用与否由 4.3 及问题 2 的因果实验判断。

| 编号 | 子实验 | 设计目的 | 状态 |
| --- | --- | --- | --- |
| 4.1 | 文献与官方说明调研 | 整理 DeepStack 原始设计、Qwen3-VL 消融与 Qwen3.5 的公开信息 | 已完成，见第 2 节 |
| 4.2 | 端到端对照 | 比较两类模型在 BabyVision、能力分解与小目标定位上的差异 | 待实施 |
| 4.3 | DeepStack 消融 | 在 Qwen3-VL 内部关闭或替换注入，测量对细粒度任务的选择性依赖 | 待实施 |
| 4.4 | 表征对照 | 比较注入特征、ViT 各层与 decoder 前部的细节可读性 | 待实施，依赖 2.3 与 2.4 的工具 |
