# 实验 2.2：位置相关成分与高范数 token

本实验属于[视觉细节表征](plan.md)，承接实验 2.1 中关于固定高分位置与高范数 token 的结果。2.1 发现，相同像素内容位于不同位置时得到不同的表征，这一位置相关成分足以主导手工单图距离读出；同时，少数 token 的范数远高于其余 token，其位置与图像内容关系很弱。本实验研究高范数 token 的三个方面：它们来自何处（来源），包含什么信息（信息），对模型回答起什么作用（功能），并以 Qwen3-VL-8B 作为对照。结果见[高范数 token 的来源、信息与功能](results_2026-10-16_high_norm.md)。

## 1. 研究问题与动机

### 1.1 现有证据

现有证据来自 2026-09-30 与 2026-10-01 的两组实验（见[固定高分位置](results_2026-09-30_pilot.md)、[距离指标与高范数 token](results_2026-09-30.md)、[图像布局与高范数排除](results_2026-10-01.md)），以及实验 3.7 与 2.5a，可以归纳为六项：

1. **跨位置差异产生于位置编码与 attention 之后。** 在未加位置编码的 patch embedding 上，正常图各实例的分数全部为零；从 ViT 第 3 层起，正常图已经存在固定最高分位置，异常图与正常图的实例分数相关系数为 0.986–0.996。
2. **merger 上的左上偏好并非纯位置效应。** 标准网格与错列布局的正常图在 merger 输出上均以左上实例为最高分；环形布局中最高分实例随图案变化。
3. **decoder 末层的范数峰与内容无关。** decoder 第 32 层的全图范数峰在 18 张正常图中均位于第一行第 1 或第 2 个 token；在环形布局中这些位置为空白背景。
4. **高范数 token 对手工读出的影响依赖层。** 在 ViT 第 18 层排除高范数 token 将正确定位数从 15/288 提高到 89/288，而在 merger 上从 66/288 降至 61/288。这些实验只修改外部分析器读取的特征，没有把修改送回网络。
5. **ViT 中存在由 learned absolute position embedding 播种的固定位置高范数 token（实验 3.7）。** 从 ViT block 10 起，左上角、右下角与第一行出现少数固定位置的高范数 token，它们承载 block 10 之后约 56% 的位置方差；将 learned absolute position embedding（下文简称 PE）置零后这些 token 消失。PE 表左右边缘第 9–11 行存在范数为中位数 5–7 倍的异常项，二者的关系尚未检验。
6. **decoder 中的高范数视觉 token 不承载答案所需的异常信息（实验 2.5a）。** 在指定实例判断的 activation patching 中，高范数视觉 token 组的恢复比例不超过 0.04，与随机视觉 token 相当。

### 1.2 相关文献与本实验的定位

1. **Vision Transformers Need Registers（Darcet et al., 2024, arXiv:2309.16588）。** DINOv2 等 ViT 中约 2% 的 patch token 范数约为其余 token 的 10 倍，出现在冗余的背景 patch 上，局部信息较弱、全局信息较强；加入可学习的 register tokens 后该现象消失。
2. **Massive Activations in Large Language Models（Sun et al., 2024, arXiv:2402.17762）。** LLM 中起始 token 与分隔符上存在固定维度的巨大激活，作用类似 bias，并吸引大量 attention。
3. **Efficient Streaming Language Models with Attention Sinks（Xiao et al., 2024, arXiv:2309.17453）。** 序列起始 token 接收大量 attention，移除后性能显著下降。
4. **Vision Transformers Don't Need Trained Registers（Jiang et al., 2025, arXiv:2506.08010）。** 少数 register neurons 产生高范数 token；在测试时把这些 neurons 的激活转移到额外添加的 token 上，可以在不训练的条件下得到与训练 register 相当的效果，并使 VLM 的 attention 归因更干净。
5. **Denoising Vision Transformers（Yang et al., 2024, arXiv:2401.02957）。** ViT 特征中存在跨图像固定、与内容无关的伪影，来源被归于位置编码。
6. **See What You Are Told: Visual Attention Sink in Large Multimodal Models（Kang et al., 2025, arXiv:2503.03321）。** LMM 的 LLM 层中，部分 visual tokens 在少数维度上具有 massive activation、接收大量 attention，移除它们不影响模型性能。

上述工作分别报告了高范数 token 的存在（1、4）、固定维度的 massive activation（2、6）、attention sink（3、6）与位置编码伪影（5），但没有回答三个对 VLM 视觉推理重要的问题。第一，高范数 token 出现在哪些位置由什么决定：Darcet et al. 认为位置随内容而定（背景 patch），Yang et al. 认为伪影由位置编码造成，两者在同一模型中如何共存未被检验。第二，ViT 中的高范数 token 是否穿过 modality connector 进入 LLM，以及它们与 LLM 中的 visual attention sink 是否为同一组 token。第三，在需要精确空间输出的任务（坐标、行列、找不同）上，消除这些 token 是否改变模型行为。本实验以这三个问题为主线：用 PE 的翻转、相位窗口、置零与内容放置分离位置编码与内容的作用（2.2b），沿 ViT、merger 与 decoder 全程追踪范数（2.2a），并在完整模型中消融高范数 token、测量坐标与行列任务的变化（2.2d）。

### 1.3 研究问题

1. **来源。** 高范数 token 首次出现在哪一个 ViT block 与哪一个 decoder 层；由 attention 还是 MLP 写入，是否由少数 register neurons 产生；出现位置由 PE、2D RoPE、图像内容（空白或有纹理）还是画布尺寸决定。
2. **信息。** 与普通 token 相比，高范数 token 所含的全局属性（实例数量、模板、画布尺寸、是否含异常实例）、位置与自身 patch 内容是否不同。
3. **功能。** 高范数 token 是否为 ViT 与 decoder 中的 attention sink；在完整模型中消融它们，坐标输出、T2、T3、行列回答与 RefCOCO 定位是否改变；它们是否介导此前距离读出中的位置相关成分；蓝框是否提高 token 范数，并由此使蓝框指定的 T3 更难。
4. **跨模型对照。** Qwen3-VL-8B 的 PE 范数约为 Qwen3.5-9B 的 2.76 倍（实验 3.7），其高范数 token 的来源、位置与作用是否不同。

## 2. 总体设计

| 子实验 | 内容 | 回答的问题 |
| --- | --- | --- |
| 2.2a 出现与分布 | 在空白、噪声、随机放置、网格、蓝框与自然图像上记录 ViT 全部 block、merger 与 decoder 全部层的范数图；统计首次出现层、位置频率、所在 patch 的内容与 massive 通道 | 来源（层与位置） |
| 2.2b 来源干预 | 分解产生高范数 token 的 block 中 attention 与 MLP 的更新，定位 register neurons；对 PE（置零、均值、打乱、翻转、缩放、去除异常项、相位窗口）、2D RoPE（置零、打乱）与图像内容（在固定位置放置纹理）分别干预 | 来源（组件与位置决定因素） |
| 2.2c 信息内容 | 在高范数 token、同位置普通 token 与随机普通 token 上训练相同的线性 probe，读出全局属性、位置与自身 patch 内容 | 信息 |
| 2.2d 功能 | 测量 ViT 与 decoder 的 attention 质量；在完整模型中施加三类消融与随机对照，测量任务指标、位置可解释方差与 2.1 的距离读出；比较蓝框图与无框图的范数 | 功能 |
| 2.2e 跨模型对照 | 在 Qwen3-VL-8B 上重复 2.2a、2.2b、2.2d 的主要部分 | 跨模型 |

机制的判定以三类证据一致为前提：来源干预确定成分随哪一因素移动，信息读出确定其存储内容，功能消融确定模型是否依赖它。

## 3. 数据

1. **空白与噪声画布。** 白、灰、黑三种纯色，边长 256、384、512、640、768、1024 pixels，另含 768×512、512×768、1024×512 三种非方形白色画布；均匀分布的 pixel 噪声画布（512 与 768，各 8 张）。纯色图像中不同位置的 patch 像素完全相同，跨位置差异只能来自位置输入与边界。
2. **随机放置合成图（probe 集）。** 画布边长 384、512、640 pixels，每种 160 张。每张图在 64-pixel 网格的随机格子上放置 2–16 个同模板实例（模板取自四个开发模板），整体偏移 0、16 或 32 pixels；一半图像中有一个实例为颜色异常（共享刺激集的 color 第 4 级）。实例数量、模板、画布尺寸与异常存在性相互独立。
3. **网格图与蓝框图。** 实验 1.3 的网格图（3×3、6×6、10×10、20×20 各 24 张）；T3 使用的蓝框正常图 48 张及其去掉蓝框后按相同 spec 重新渲染的无框图（重新渲染的蓝框图与原文件逐像素一致）。
4. **自然图像。** 实验 3.4 的 RefCOCO 测试集 512-pixel 副本 300 张（保持长宽比，token 网格随图像变化）；位置可解释方差的计算另用中心正方形裁剪并缩放到 512×512 的版本。

decoder 特征使用前缀模板：图像位于文字“Describe the image.”之前，只运行 prefill。

## 4. 方法

### 4.1 符号约定

对固定的提取位置（ViT 第 $k$ 个 block 的输出、merger 输出或 decoder 第 $l$ 层的输出）与一张图像，约定：

- $P$：该图像在该提取位置的 token 数量；$D$：特征维度（ViT 为 1152，merger 与 decoder 为 4096）。
- $x_p \in \mathbb{R}^{D}$：位置 $p$ 的特征向量；ViT 中的 patch 先从 2×2 block-major 顺序还原到空间栅格顺序。
- $\lVert x_p \rVert_2$：L2 范数；$m = \operatorname{median}_q \lVert x_q \rVert_2$：同一图像、同一提取位置的范数中位数。

### 4.2 高范数 token 与范数比

位置 $p$ 的范数比与高范数指示为

$$
\rho_p = \frac{\lVert x_p \rVert_2}{m}, \qquad h_p = \mathbb{1}\left[\rho_p > \tau\right], \qquad \tau = 3.
$$

以图像内中位数为基准可以消除不同图像、不同层整体范数尺度的差异；$\tau = 3$ 沿用 2026-09-30 的阈值，首次出现层另以 $\tau = 5$ 判定，以免把 ViT 浅层中 2–4 倍的弱峰误计为高范数 token。范数图即 $\rho_p$ 在空间网格上的热图。

对一组同一网格尺寸的 $N$ 张图像，位置 $p$ 的高范数频率为

$$
f_p = \frac{1}{N}\sum_{n=1}^{N} h_{n,p},
$$

其中 $h_{n,p}$ 为第 $n$ 张图像在位置 $p$ 的指示。若 $f_p$ 集中于少数固定位置，说明位置由与内容无关的因素决定；若 $f_p$ 分散且与背景区域重合，说明位置随内容而定。

**PE 表坐标。** Qwen3.5 的 PE 是 $48 \times 48$ 的可学习表，按 align-corners 双线性插值到 $h \times w$ 的 patch 网格。patch 网格中第 $r$ 行第 $c$ 列对应表坐标

$$
(r', c') = \left(\frac{47\,r}{h-1}, \frac{47\,c}{w-1}\right).
$$

若不同尺寸画布上的高范数位置换算到表坐标后重合，说明位置由 PE 表的某些项决定，而非由 patch 网格的行列编号或画布边界决定。

**内容统计。** 每个 patch（16×16 pixels）的灰度标准差小于 0.02（灰度归一化到 0–1）时记为空白 patch。比较高范数 patch 中空白 patch 的比例与随机 patch 中空白 patch 的比例。

**massive 通道。** 对每个高范数 token 记录绝对值最大的 10 个通道，统计各通道出现的频率。

### 4.3 写入组件与 register neurons（2.2b）

ViT block 的计算为

$$
y = x + \operatorname{Attn}(\operatorname{LN}_1(x)), \qquad z = y + \operatorname{MLP}(\operatorname{LN}_2(y)), \qquad \operatorname{MLP}(u) = W_2\,\phi(W_1 u + b_1) + b_2,
$$

其中 $x, y, z \in \mathbb{R}^{1152}$ 分别为 block 输入、attention 之后与 block 输出，$W_1 \in \mathbb{R}^{4304 \times 1152}$，$W_2 \in \mathbb{R}^{1152 \times 4304}$，$\phi$ 为 GELU（tanh 近似）。在高范数 token 首次出现的 block 中，分别比较高范数 token 与其余 token 的 $\lVert x \rVert$、attention 更新范数、MLP 更新范数与 $\lVert z \rVert$，确定大范数由哪一个子层写入。

MLP 中间激活 $a = \phi(W_1 u + b_1) \in \mathbb{R}^{4304}$ 的第 $i$ 个分量称为第 $i$ 个 neuron。在 49 张图像（空白、噪声、随机放置与自然图像）的全部 token 上，计算每个 neuron 在高范数 token 上的平均激活 $\bar{a}^{\mathrm{hn}}_i$ 与在其余 token 上的平均激活 $\bar{a}^{\mathrm{oth}}_i$。register neurons 定义为同时满足下式的 neurons：

$$
\bar{a}^{\mathrm{hn}}_i > 10, \qquad \left|\bar{a}^{\mathrm{oth}}_i\right| < 0.1 .
$$

第一个条件要求该 neuron 在高范数 token 上强烈激活，第二个条件要求它在其余 token 上几乎静默；满足两者的 neurons 是一个只对少数 token 开启的开关，与 Jiang et al. 描述的 register neurons 对应。随后计算这些 neurons 经 $W_2$ 写出的向量范数占 MLP 更新范数的比例，并检查 $W_2$ 对应列中权重最大的输出通道是否与 massive 通道一致。

### 4.4 位置来源干预（2.2b）

每项干预只改变一个位置因素，在同一批图像（白色与灰色画布、4 张噪声画布、16 张随机放置图、16 张自然图像）上与未干预条件比较 ViT block 12 的高范数集合。

| 干预 | 操作 | 若位置由 PE 决定，预测为 |
| --- | --- | --- |
| PE 置零 / 均值 | 每个 patch 的 PE 置零，或替换为该图像全部 patch PE 的均值 | 空白画布上的高范数 token 消失 |
| PE 打乱 | 插值后的 PE 在 patch 之间随机置换（2 个种子） | 高范数位置随置换移动 |
| PE 翻转 | 插值后的 PE 网格左右或上下翻转，2D RoPE 不变 | 高范数位置随之翻转 |
| PE 缩放 | PE 乘以 0.5 或 2 | 数量随 PE 幅值变化 |
| 去除 PE 异常项 | 将表中范数超过中位数 3 倍的项缩放到中位数范数后再插值 | 若异常项是种子，对应位置的高范数 token 消失 |
| PE 相位窗口 | 512 画布使用 640 画布（40×40 patch）中偏移 $(r_0, c_0)$ 处 32×32 窗口的 PE | 高范数图等于 640 画布高范数图的对应裁剪 |
| 2D RoPE 置零 / 打乱 | 全部 attention block 的 RoPE ids 置零或在 patch 之间置换 | 若位置只由 PE 决定，高范数位置不变 |
| 内容放置 | 白色 512 画布上，以 64×64 的噪声块覆盖某一固定高范数位置，或覆盖 3 个对照位置 | 若高范数 token 只出现在空白 patch，被覆盖的位置失去高范数 token |

比较指标为干预前后 block 12 高范数集合的 Jaccard 系数（翻转条件与相应翻转后的未干预集合比较）与高范数 token 数量。PE 与 RoPE 的替换复用实验 3.7 的 `vit_edit` 实现；翻转与异常项去除通过替换 PE 查表输出实现。

### 4.5 信息内容（2.2c）

在 probe 集的 480 张图像上，保存 ViT block 6、12、20、26 与 merger 输出的三组 token 特征：

1. **高范数 token**：该层判为高范数的 token（每张图像至多 16 个）。merger 上使用包含 block 12 高范数 patch 的 merged token。
2. **同位置普通 token**：白色画布上相同网格尺寸的高范数位置中，在本图像中未成为高范数 token 的 token。它们与高范数 token 共享位置输入。
3. **随机普通 token**：每张图像随机选取与高范数 token 数量相同的普通 token。

另以每张图像中普通 token 的均值作为图像级读出的参照。图像按 3:1 划分训练与测试，每组训练相同的线性读出：

| 信息类别 | 探测目标 | 读出与指标 |
| --- | --- | --- |
| 局部 | 该 token 对应 16×16 patch 的 4×4 平均池化灰度 | ridge 回归，测试集 $R^2$ |
| 位置 | token 的归一化行列 $(r/(h-1), c/(w-1))$ | ridge 回归，测试集 $R^2$ |
| 全局 | 模板（4 类）、画布尺寸（3 类）、是否含异常实例（2 类）、实例数量（2–5、6–10、11–16 三档） | ridge 分类器（one-hot 目标的 ridge 回归取最大值），balanced accuracy |

正则化系数在训练集的 20% 验证划分上选择。每个读出另在逐 token L2 归一化的特征上训练，以排除范数差异对读出的影响。合成图中的高范数 token 全部位于空白 patch，自身内容为常数，局部读出因此改在自然图像上比较高范数 token 与随机普通 token。

### 4.6 功能（2.2d）

**ViT attention 质量。** 对 ViT block $k$ 的第 $j$ 个 head，记 softmax attention 矩阵为 $A^{(k,j)} \in \mathbb{R}^{P \times P}$，由 block 输入经 qkv 投影与 2D RoPE 重新计算得到。高范数集合 $S$（block 12 判定）获得的 attention 质量为

$$
M_S^{(k,j)} = \frac{1}{P}\sum_{i=1}^{P}\sum_{q \in S} A^{(k,j)}_{iq}.
$$

$M_S$ 是全部 query 平均分给集合 $S$ 的 attention 比例；与随机选取的同样数量 token 的 $M_R$ 比较，$M_S \gg M_R$ 说明 $S$ 是 attention sink。富集比 $e = M_S / (|S|/P)$ 表示 $S$ 获得的 attention 是其数量占比的多少倍。

**decoder attention 质量。** 在 Qwen3.5 的 8 个 full attention 层上，由 prefill 的 pre-hook 重新计算 softmax attention（复用实验 3.5 的 `attention_readout`，重算输出与模块输出的相对误差均有记录），query 为最后一个 prompt token 与 16 个随机 visual tokens，统计其 attention 落在 4 个前缀 token、全部 visual tokens、包含 ViT 高范数 patch 的 visual tokens 与第一个 visual token 上的质量。Gated DeltaNet 层没有显式 attention 矩阵，不在测量范围内。

**完整模型消融。** 消融施加在高范数 token 首次出现的 ViT block（$k_0 = 10$）的输出上，并随前向计算向后传播。记该 block 输出中高范数集合为 $S$、其余 token 集合为 $\bar{S}$，三类消融与对照为：

$$
\text{均值替换：}\ x_p \leftarrow \frac{1}{|\bar{S}|}\sum_{q \in \bar{S}} x_q \ (p \in S); \qquad \text{置零：}\ x_p \leftarrow 0 \ (p \in S); \qquad \text{neuron 置零：}\ a_i \leftarrow 0 \ (i \in \mathcal{R}).
$$

其中 $\mathcal{R}$ 为 4.3 节定义的 register neurons，neuron 置零作用于 block $k_0$ 的 MLP 中全部 token。均值替换与置零删除高范数 token 在 block $k_0$ 之后携带的全部内容；neuron 置零只阻止 register neurons 写入，保留这些 token 的 attention 更新与此前的残差内容，近似 Jiang et al. 的测试时 register 去除（不添加额外 token，以保持 merger 的 2×2 合并结构）。随机对照在每张图像中随机选取 $|S|$ 个普通 token 施加同样的均值替换（2 个种子）。消融只修改 ViT 内部的计算，merger 与 decoder 不变；Qwen3-VL 中 DeepStack 从 block 9、17、25（0 起的 block 索引 8、16、24）取特征，消融同样影响后两者。

测量的任务与指标如下（thinking off）：

| 任务 | 数据 | 指标 |
| --- | --- | --- |
| 单圆点坐标（实验 3.5） | 白色 512 画布上 36 个网格位置 × 直径 32、64 pixels，point 与 bbox 两种格式 | 命中率，中心误差（0–1000 坐标），bbox IoU |
| T2 两实例比较 | 实验 1.3 主问法，240 条（变体与网格规模平衡） | 准确率，AUROC |
| T3 指定实例判断 | 实验 1.3 修订问法（计划第 6.8 条），蓝框与行列描述各 240 条 | 准确率，AUROC |
| T4 定位，直接回答 | 行列格式与 point 格式各 200 条（网格规模平衡） | 严格评分准确率 |
| T4 定位，先推理后回答 | 实验 1.3 显式推理问法的行列格式 72 条，max_new_tokens 为 1024 | 严格评分准确率 |
| RefCOCO | 实验 3.5 使用的 3.4 测试集 300 条，point 与 bbox 两种格式 | point 命中率、bbox IoU≥0.5 比例、中心误差 |

yes/no 任务以首个答案 token 的判别分数 $s = z_{\mathrm{Yes}} - z_{\mathrm{No}}$ 作答（$s > 0$ 判为 Yes），由左填充的批量 prefill 计算；生成任务使用 `generate_batch`。比例给出 Wilson 95% 区间，与未消融条件的差异以逐样本配对的 exact McNemar 检验，AUROC 差异以 bootstrap 区间表示。

**位置可解释方差。** 为检验高范数 token 是否介导此前距离读出中的位置相关成分，对同一网格尺寸的一组图像，以交叉估计的位置可解释方差衡量位置成分的比重。将图像按索引奇偶分为两半 $A$ 与 $B$，用 $A$ 估计位置均值 $\mu^{A}_p$ 与全局均值 $\bar{x}^{A}$，在 $B$ 上计算

$$
R^2_{\mathrm{pos}} = 1 - \frac{\sum_{n \in B}\sum_{p} \lVert x_{n,p} - \mu^{A}_p \rVert_2^2}{\sum_{n \in B}\sum_{p} \lVert x_{n,p} - \bar{x}^{A} \rVert_2^2}.
$$

分母是以全局均值预测每个 token 的误差，分子是以位置均值预测的误差；交叉估计使无位置效应时的期望值不高于零。报告两半交换后的均值，并另报告从分子与分母中同时去掉白色画布高范数位置（merger 与 decoder 上为包含这些位置的 merged token）后的结果。图像组为 160 张 512 随机放置图与 160 张 512×512 自然图像，条件为未消融、均值替换、随机对照、neuron 置零与 PE 置零。

**2.1 距离读出。** 在 2.1 的 6×6 小人图（384×384，缺口 4 与 8 pixels，36 个目标位置与 1 张正常图）上，以每个实例的特征到全部实例逐维中位数的 RMS 距离作为异常分数，记录分数最高的实例是否为目标，在 ViT block 18、merger 与 decoder 第 16 层上比较上述条件。

**蓝框效应。** 对蓝框图与无框图，取与蓝框边带（框线内外各 4 pixels）相交的 patch，计算各层这些 patch 的平均范数之比（蓝框图 / 无框图），并统计边带上的高范数 token 数量。若蓝框使这些 token 成为高范数 token，则可能吸收 attention 或干扰定位。

## 5. 指标汇总

| 指标 | 定义 | 子实验 |
| --- | --- | --- |
| 范数比 $\rho_p$、高范数频率 $f_p$ | 4.2 节 | 2.2a、2.2b |
| 首次出现 block | 第一个存在 $\rho_p > 5$ 的 block | 2.2a |
| 空白 patch 比例 | 高范数 patch 与随机 patch 中空白 patch 的比例，Wilson 区间 | 2.2a |
| 子层更新范数、register neurons | 4.3 节 | 2.2b |
| Jaccard 系数 | 干预前后 block 12 高范数集合的交并比 | 2.2b |
| probe 读出 | 测试集 $R^2$ 与 balanced accuracy | 2.2c |
| attention 质量 $M_S$、富集比 $e$ | 4.6 节 | 2.2d |
| 任务指标变化 | 准确率差、McNemar p 值、AUROC 差、中心误差差 | 2.2d |
| $R^2_{\mathrm{pos}}$ | 4.6 节 | 2.2d |

## 6. 可视化

1. 逐层范数图：ViT block、merger 与 decoder 层并列，统一 log 色标与色条，红色空心圆标出高范数 token。
2. 出现曲线：各数据组最大范数比与高范数数量随 ViT block 的变化，decoder 视觉 token 的最大范数比随层的变化。
3. 高范数频率热图与 PE 表坐标散点图（以 PE 表范数热图为底）。
4. 写入组件条形图与 register neuron 激活箱线图。
5. 来源干预的范数图网格、相位窗口与内容放置图。
6. ViT attention 质量曲线与 attention 接收图，decoder attention 质量曲线。
7. probe 分组条形图、$R^2_{\mathrm{pos}}$ 曲线、消融效应森林图与蓝框效应曲线。

## 7. 解释范围

1. 只有当来源干预、信息读出与功能消融三者给出一致结论时，才以 register 或 attention sink 命名某一部位的现象；单项证据只报告为与某机制相容。
2. PE 置零、打乱、翻转与 RoPE 干预使模型处于训练分布之外，其效应可能部分来自分布偏移本身，因此只用于判断高范数位置的决定因素，不用于任务指标。
3. 高范数阈值是描述性定义；不同阈值下的首次出现层与位置需一致才予采用。
4. 自然图像中内容与位置存在统计相关（如天空常在上方），自然图像的 $R^2_{\mathrm{pos}}$ 包含这一部分，需与随机放置图对照解释。

## 8. 运行与产物

1. 引擎为 `analysis_tools/representation/high_norm.py`（高范数检测、ViT 消融、PE 替换、attention 质量、批量判别分数与 $R^2_{\mathrm{pos}}$ 累加器），阶段驱动为 `high_norm_run.py`（neurons、survey、source、attention、posvar、ablation、spot），CPU 分析与作图为 `high_norm_analysis.py` 与 `high_norm_figures.py`。
2. Bash 入口为 `scripts/analysis/representation/high_norm.sh`，模型、阶段、样本量、消融条件与批大小集中于脚本顶部。
3. 产物保存于 WORK_DIR 的 `analysis/representation/2026-10-16_high_norm/<模型标签>/`，文档用图位于 `docs/analysis/representation/figures/high_norm/`。
