# 实验 3.5：坐标读取路径

本实验属于[位置信息与坐标读取](plan.md)。实验 3.1–3.3 说明 visual token 的槽位位置可以被外部 probe 部分读出，但没有回答模型在生成坐标时实际使用哪一种位置信息。实验 1.2 中，6×6 网格里红色目标的行列答案 3/3 正确而中心点 0/3 正确，RefCOCO 的 8 个 bbox 均达到 IoU≥0.5，说明坐标输出能力依赖场景与答案格式。本实验在完整模型生成 point、bbox 与行列答案时测量 attention，并对各位置来源进行因果干预，确定位置信息的来源以及它经过的层和 head。

## 1. 研究问题与动机

Qwen3.5-9B 中至少有四条路径可以把位置信息带到坐标 token：

| 来源 | 计算位置 | 作用方式 |
| --- | --- | --- |
| S1 ViT 位置 | ViT | learned absolute position embedding 在 patch embedding 后加入一次；ViT 每层 attention 使用 2D RoPE。二者的结果经 merger 写入 visual embedding |
| S2 decoder position IDs | decoder 的 8 个 full attention 层 | MRoPE 按 position IDs 旋转 query 与 key 的部分维度，使 attention score 依赖 token 之间的相对位置 |
| S3 视觉内容 | 图像本身 | 自然图像中内容与位置存在统计相关性，例如天空多位于上方、图像边缘的内容与中心不同 |
| S4 序列顺序 | decoder 的 24 个 linear attention 层与 causal mask | Gated DeltaNet 的 causal depthwise convolution 与递推更新对 token 顺序敏感；visual token 按 merged 网格的光栅顺序排列 |

本实验回答三个问题：

1. 生成每个坐标 token 时，哪些 full attention head 读取目标区域的 visual token；linear attention 层在目标区域上的归因如何随层变化。
2. 坐标输出由 S1、S2、S3、S4 中的哪些来源决定，各来源的权重是多少。
3. 行列序数答案与 point、bbox 答案是否使用相同的读取路径。

第 2 个问题的结论直接服务于问题 5 与方向二：局部重读时重新输入的窗口 token 应携带什么 position IDs，取决于模型的坐标是否由 position IDs 决定。

## 2. 总体设计

实验由六个部分组成，按顺序开展，后一部分使用前一部分的样本或候选 head：

| 部分 | 内容 | 输出 |
| --- | --- | --- |
| C0 基准任务 | 三类坐标任务的基线准确率，划分答对与答错样本 | 机制分析所用的样本集 |
| C1 attention 读取 | 记录生成坐标 token 时的 attention 与归因 | 候选“指向型 head”与层分布 |
| C2 位置来源干预 | 分别干预 S1、S2、S4，并以冲突设计估计来源权重 | 各来源对坐标输出的权重 |
| C3 内容交换 | 交换目标与空白区域的 visual embedding，position IDs 保持原位 | 位置随内容携带还是由槽位决定 |
| C4 head 消融与 patching | 对候选 head 做均值消融与输出 patching | 候选 head 对坐标的因果作用 |
| C5 行列答案 | 像素层面与 position IDs 层面的行插入 | 行列序数的读取方式 |

机制分析主要在答对样本上进行，因为只有在模型能够完成任务时，干预引起的输出变化才能解释为“破坏或移动了所使用的位置信息”。答错样本同样运行 C1 与 C2，用于比较失败时读取路径的差异。

## 3. 模型中的位置计算

### 3.1 MRoPE position IDs

设输入序列长度为 $L$，序列索引为 $n$。每个 token 的 position ID 为三维整数向量

$$
p_n = \big(p_n^{t},\ p_n^{h},\ p_n^{w}\big) \in \mathbb{Z}^{3},
$$

三个分量分别对应时间轴、行轴与列轴。设图像的 merged 网格为 $H$ 行 $W$ 列，图像之前的最后一个文本 token 的 position ID 为 $s-1$，则：

1. 文本 token 的三个分量相同，并逐 token 递增 1。
2. 第 $r$ 行第 $c$ 列（零起始）的 visual token 满足 $p^{t} = s$，$p^{h} = s + r$，$p^{w} = s + c$。
3. 图像之后的第一个文本 token 的三个分量均为 $s + \max(H, W)$，即图像的最大位置加 1。

Full attention 使用 interleaved MRoPE：每个 head 的维数为 256，只旋转前 64 维，即 32 个二维通道对，旋转频率为

$$
\theta_u = \left(10^{7}\right)^{-u/32}, \qquad u = 0, 1, \ldots, 31,
$$

其中 $u$ 为通道对索引。32 个通道对按 mrope_section 为 [11, 11, 10] 交错分配给 $t$、$h$、$w$ 三个轴。由于 rope_theta 为 $10^{7}$，频率随 $u$ 迅速下降：在 merged 网格内最大约 32 个单位的位移下，只有 $\theta_u \cdot 32 \geq 0.1$ 的通道对产生可分辨的旋转，即约 $u \leq 11$ 的 12 个通道对，每个空间轴约 4 个。因此 S2 只作用于 8 个 full attention 层、每个 head 1/4 的维数，且其中只有少数通道对区分图像内部的位置。通道对的具体分配在实现时按本地代码核对。

### 3.2 干预接口

1. **外部 position IDs。** forward 与 generate 均接受外部传入的 position_ids。本实验传入形状为 $(4, B, L)$ 的张量，第 0 行为原始一维文本位置，用于构造 attention mask，后三行为被干预的 $(t, h, w)$。传入后模型不再自行计算 MRoPE，生成阶段新 token 的 position IDs 由最后位置递增得到，实现时核对这一行为。
2. **替换 visual embedding。** 通过 model.model.get_image_features 取得 visual embedding，修改后按 image token 的位置写入 inputs_embeds，再执行前向；此时必须同时提供 position_ids。
3. **ViT 位置干预。** 在 ViT 前向中替换 learned position embedding 的插值结果，或修改 ViT 2D RoPE 的行列 ID。
4. **attention 权重。** 只有在 attn_implementation 为 eager 时返回，且只来自 full attention 层；返回值是 softmax 权重，head 的实际输出还要逐元素乘以 sigmoid gate。C1 使用 eager，其余部分使用 SDPA；实现时在 C0 的样本上核对两种实现的 greedy 输出一致。

## 4. C0：基准任务与样本

| 任务 | 图像 | 问题与答案格式 | 样本量 |
| --- | --- | --- | --- |
| 单目标定位 | 512×512 白色画布上的单个红色实心圆，直径 32、64、128 pixels，圆心位于 6×6 等距位置并加不超过 16 pixels 的随机抖动 | point 与 bbox | 3 × 36 = 108 张 |
| 平移序列 | 同上，圆心沿水平中线与竖直中线各取 15 个位置，步长 32 pixels | point | 3 × 30 = 90 张 |
| 自然图像 | RefCOCO 图像，取自实验 3.4 的测试划分，使用第一条 referring expression | point 与 bbox | 300 张 |
| 行列答案 | 实验 1.3 T6 的红色目标网格，6×6、10×10、20×20 | 行列、point、bbox | 每种规模 48 张 |

单目标画布的边长 512 pixels 对应 16×16 的 merged 网格，圆的位置与尺度完全受控，且画布背景不提供位置线索。平移序列用于绘制输出坐标随真实位置变化的连续曲线。问题文字与答案格式沿用实验 1.2 与 1.3：坐标为 0–1000 归一化，横坐标向右增大，纵坐标向下增大。

评分使用 point 命中率（预测点落在圆或目标框内）、中心误差（0–1000）、bbox IoU≥0.5 的比例与行列 exact match。全部样本使用 greedy decoding，输出上限为 64 tokens，thinking off。

**坐标 token。** Qwen 系列分词器可能把数字逐位切分，实现时先核对 0–1000 内全部整数的切分方式。若逐位切分，一个坐标值由多个 token 生成，例如 537 由 5、3、7 三个 token 组成；本实验对每个坐标 token 分别分析，并以每个坐标值的首位 token 为主要分析对象，因为首位决定坐标的百位区间，后续 token 只在首位确定后细化。

## 5. C1：attention 读取

### 5.1 符号

设 $\mathcal{V}$ 为全部 visual token 的序列索引集合，$\mathcal{T} \subset \mathcal{V}$ 为与目标 tight bbox 重叠的 visual token 集合。对生成步 $m$（即生成某个坐标 token 时的 query 位置），full attention 层 $\ell$ 的第 $k$ 个 head 对 token $j$ 的 softmax 权重记为 $a^{\ell,k}_{m,j}$，该 head 对 token $j$ 的 value 向量记为 $v^{\ell,k}_{j} \in \mathbb{R}^{256}$，生成步 $m$ 的 sigmoid gate 记为 $\gamma^{\ell,k}_{m} \in (0,1)^{256}$，output projection 中属于该 head 的子矩阵记为 $W^{\ell,k}_{O} \in \mathbb{R}^{4096 \times 256}$。

### 5.2 指标

**目标富集度。** 定义

$$
E^{\ell,k}_{m} = \frac{\sum_{j \in \mathcal{T}} a^{\ell,k}_{m,j} \big/ \sum_{j \in \mathcal{V}} a^{\ell,k}_{m,j}}{|\mathcal{T}| \,/\, |\mathcal{V}|}.
$$

分子是 visual token 上的 attention 中落在目标区域的比例，分母是目标区域占全部 visual token 的比例。$E = 1$ 表示该 head 对目标区域的 attention 与均匀分配相同，$E$ 远大于 1 表示该 head 集中读取目标区域。除以面积比例使不同尺度的目标可以比较。另记录 attention 最大的 visual token 是否属于 $\mathcal{T}$。

**有效贡献。** attention 权重只反映读取的分配，不反映读取的内容对 residual stream 的作用大小。Qwen3.5 的 full attention 在 output projection 前逐元素乘以 sigmoid gate，因此定义 token $j$ 经 head $(\ell, k)$ 写入 residual stream 的向量范数为

$$
c^{\ell,k}_{m,j} = \left\| W^{\ell,k}_{O} \left( \gamma^{\ell,k}_{m} \odot a^{\ell,k}_{m,j}\, v^{\ell,k}_{j} \right) \right\|_2 .
$$

以 $c$ 代替 $a$ 计算富集度，得到有效富集度。gate 接近 0 的 head 即使 attention 集中，其贡献也很小。

**attention 质心。** 以 $a^{\ell,k}_{m,j}$ 为权重对 visual token 的中心坐标加权平均，得到 head 的 attention 质心（0–1000）。在答对与答错样本上分别计算质心与模型输出坐标的相关性。若某 head 的质心在答错样本上跟随模型的错误输出而偏离真值，该 head 位于坐标的读取路径上；若质心始终指向真值而输出错误，失败发生在该 head 之后。

### 5.3 候选 head 的选择

把答对样本按图像随机分为选择集与验证集各半。在选择集上按坐标首位 token 的平均 $\log E$ 对 8 层 × 全部 head 排序，取前 10 个且“attention 最大位置属于目标”的比例不低于 0.5 的 head 作为候选“指向型 head”；在验证集上报告其富集度，避免选择偏差。

### 5.4 linear attention 层的归因

linear attention 层没有 attention 矩阵，使用梯度与激活的乘积作为替代度量。设 $z$ 为生成步 $m$ 的坐标 token 的 logit 减去其他数字 token 的平均 logit，$x^{\ell}_{j} \in \mathbb{R}^{4096}$ 为第 $\ell$ 层输入在 visual token $j$ 上的 residual，定义

$$
A^{\ell}_{j} = \left( \frac{\partial z}{\partial x^{\ell}_{j}} \right)^{\mathsf T} x^{\ell}_{j}, \qquad \rho^{\ell} = \frac{\sum_{j \in \mathcal{T}} |A^{\ell}_{j}|}{\sum_{j \in \mathcal{V}} |A^{\ell}_{j}|}.
$$

$A^{\ell}_{j}$ 是 $z$ 对该 residual 的一阶近似贡献，$\rho^{\ell}$ 是目标区域在全部 visual token 归因中的占比。$\rho^{\ell}$ 随层的变化指示目标区域的信息在哪些层之后已被搬运到其他位置：信息被读走后，后续层的 visual token residual 对输出的影响下降。该指标对全部 32 层计算，与 full attention 层的富集度对照。

### 5.5 比较对象

每个样本比较三类 query 位置：问题末端 token（生成开始前）、横坐标首位 token、纵坐标首位 token。若问题末端 token 已经集中读取目标，而坐标 token 的 attention 分散，说明位置在生成前已被汇集到文本位置；若坐标 token 各自重新读取目标，说明坐标在生成时逐个读取。

## 6. C2：位置来源的因果干预

### 6.1 位移权重

设基线输出坐标为 $\hat y_0 \in \mathbb{R}^{2}$，干预后输出为 $\hat y \in \mathbb{R}^{2}$，某一假设下干预应引起的位移为 $d \in \mathbb{R}^{2}$（均为 0–1000 坐标）。定义位移权重与正交残差

$$
w = \frac{(\hat y - \hat y_0)^{\mathsf T} d}{\| d \|^2}, \qquad e_{\perp} = \frac{\left\| (\hat y - \hat y_0) - w\, d \right\|}{\| d \|}.
$$

$w$ 是观测位移在预测位移方向上的投影与预测位移长度之比：$w = 1$ 表示输出完全跟随被干预的来源，$w = 0$ 表示输出不受该来源影响，$0 < w < 1$ 表示该来源与其他来源共同决定输出。$e_{\perp}$ 衡量与预测方向垂直的位移分量；$e_{\perp}$ 较大说明干预主要破坏了输出的稳定性，输出没有沿预测方向移动。bbox 输出以框中心计算 $w$，另报告框宽高的相对变化。干预样本的 $w$ 以图像为单位 bootstrap 给出区间。

### 6.2 S2：position IDs 干预

1. **全局平移。** 先构造带间隙的基线：图像前后各保留 $G = 8$ 个未使用的位置，即图像 token 的 position IDs 整体加 $G$，图像之后的文本从 $s + 2G + \max(H, W)$ 开始。间隙基线须先通过一致性检验：在全部 C0 答对样本上比较间隙基线与标准 position IDs 的 greedy 输出，报告坐标完全相同的比例与坐标差的分布，并比较首个坐标 token 的 logits。若二者一致，后续平移以间隙基线为参照；若存在系统差异，则单独报告该差异，平移效应只相对间隙基线计算，不与标准 position IDs 的结果直接比较。随后对图像 token 施加平移 $p^{h} \leftarrow p^{h} + \delta_h$、$p^{w} \leftarrow p^{w} + \delta_w$，$\delta_h, \delta_w \in \{-8, -4, -2, 0, 2, 4, 8\}$，文本 token 的 position IDs 保持不变。设置间隙是为了使平移后的图像 IDs 不与前后文本的 IDs 重叠，并使问题 token 的 position IDs 在全部条件下相同，从而只改变图像与问题之间的相对位置。若模型以“目标相对问题 token 的位置差”读取坐标，输出纵坐标的预测位移为 $1000\,\delta_h / H$，横坐标同理；若模型以目标相对图像首 token 或图像边界的位置差读取坐标，输出不变。对每个样本以输出位移对预测位移做过原点的线性回归，斜率即多个 $\delta$ 合并估计的 $w$。
2. **区域互换。** 选取与目标区域 $\mathcal{T}$ 等大、位于空白背景的区域 $\mathcal{T}'$，交换两区域 token 的 $(p^{h}, p^{w})$，visual embedding 与序列顺序保持不变。预测位移 $d$ 为 $\mathcal{T}'$ 中心减去 $\mathcal{T}$ 中心。若输出移向 $\mathcal{T}'$，坐标来自 S2。

### 6.3 S1：ViT 位置干预

1. **全局平移。** ViT 的 learned position embedding 由 48×48 的表经双线性插值（align_corners 为 True）得到。为在不外推的前提下平移，将图像视为一个更大虚拟画布的子窗口：按边长各多出 $2\mu$ 个 patch 的虚拟画布插值，再取偏移 $(\mu + \delta_r, \mu + \delta_c)$ 的窗口作为该图像的 position embedding。基线取 $\delta_r = \delta_c = 0$ 的居中窗口，使虚拟画布带来的尺度变化在基线与干预之间相同。若输出由 S1 决定，预测位移为 $1000\,\delta_c / W_p$ 与 $1000\,\delta_r / H_p$，其中 $H_p, W_p$ 为 patch 网格的行列数。
2. **ViT RoPE 的全局平移。** ViT 的 attention 是非因果的，2D RoPE 只依赖 token 之间的相对位置，因此全部 ViT RoPE ID 同时平移后输出应严格不变。该条件作为实现正确性的检查。
3. **区域互换。** 交换 $\mathcal{T}$ 与 $\mathcal{T}'$ 对应 patch 的 position embedding，分别在只交换 position embedding、只交换 ViT RoPE ID、二者同时交换三种条件下运行。decoder 的 position IDs 保持不变。

### 6.4 S4：序列置换

将 $\mathcal{T}$ 与 $\mathcal{T}'$ 的 token 在序列中的位置互换，每个 token 保留自己的 visual embedding 与 position IDs，只改变它们在序列中的顺序。full attention 在给定 position IDs 时只受 causal mask 影响，linear attention 的卷积与递推则直接感知新顺序。若输出移向 $\mathcal{T}'$，坐标依赖序列顺序。

### 6.5 冲突设计

在单目标画布上，目标像素固定于区域 $\mathcal{T}$，以 S1 区域互换与 S2 区域互换的 2×2 组合构造四个条件：

| 条件 | ViT 位置指向 | decoder position IDs 指向 | 估计量 |
| --- | --- | --- | --- |
| 基线 | $\mathcal{T}$ | $\mathcal{T}$ | $\hat y_0$ |
| 仅 S1 互换 | $\mathcal{T}'$ | $\mathcal{T}$ | $w_{1}$ |
| 仅 S2 互换 | $\mathcal{T}$ | $\mathcal{T}'$ | $w_{2}$ |
| 二者互换 | $\mathcal{T}'$ | $\mathcal{T}'$ | $w_{12}$ |

四个条件的预测位移 $d$ 相同，均为 $\mathcal{T}'$ 中心减去 $\mathcal{T}$ 中心。$w_{1}$ 与 $w_{2}$ 分别估计 S1 与 S2 的权重；$1 - w_{12}$ 估计像素内容与序列顺序共同承担的权重。比较 $w_{12}$ 与 $w_{1} + w_{2}$ 检验两个来源是否线性叠加：二者接近时来源独立贡献，$w_{12}$ 明显大于 $w_{1} + w_{2}$ 时说明模型在两个来源一致时才移动输出。$\mathcal{T}'$ 取与 $\mathcal{T}$ 距离为 2、4、6 个 merged 单元的三个位置，检验权重是否依赖位移大小。

### 6.6 自然图像

RefCOCO 样本只运行全局平移（S2 与 S1），因为自然图像中没有可以互换的空白区域。全局平移下像素内容保持不变，因此 $1 - w$ 包含了 S3 的贡献；与单目标画布的结果比较，可以估计自然图像中内容线索的额外作用。

## 7. C3：内容交换

把目标区域 $\mathcal{T}$ 的 visual embedding 与空白区域 $\mathcal{T}'$ 的 visual embedding 互换，position IDs 与序列槽位保持原位。互换后，目标内容位于槽位 $\mathcal{T}'$，该槽位的 position IDs 与序列顺序指向 $\mathcal{T}'$，而 visual embedding 内含的 S1 位置仍指向 $\mathcal{T}$。

1. 若输出移向 $\mathcal{T}'$，坐标由槽位决定（S2 或 S4），visual embedding 中的内容只决定选择哪个 token。
2. 若输出停留在 $\mathcal{T}$，位置信息随内容一起被 visual embedding 携带（S1）。

结合 6.2 的区域互换与 6.4 的序列置换，可以在槽位来源内部区分 S2 与 S4。另以同一问题询问“图中是否有红色圆”，读取 Yes 与 No 的 logit 差，确认交换后目标仍被识别，从而把“位置读取改变”与“目标识别失败”分开。该部分对应旧计划中的语义来源与槽位区分：交换后槽位坐标为 $\mathcal{T}'$、内容来源坐标为 $\mathcal{T}$，模型输出给出二者中实际起作用的一方。

## 8. C4：head 消融与 patching

1. **均值消融。** 将候选 head 的门控输出（gate 之后、output projection 之前）替换为该 head 在 C0 全部样本、同一 query 位置类型上的平均值，测量坐标中心误差与目标识别 logit 差的变化。坐标误差显著上升而识别不变，说明该 head 专门负责位置读取。对照为从相同层中随机抽取同数量的 head，重复 20 次。对 linear attention 层中 $\rho^{\ell}$ 最高的层，同样消融其 head 输出作为补充。
2. **输出 patching。** 取目标位于 $\mathcal{T}$ 的 run A 与目标位于 $\mathcal{T}'$ 的 run B（两张图尺寸相同，序列长度相同）。在 run A 的全部生成步中，用 run B 对应位置的候选 head 输出替换 run A 的输出，自由生成坐标，并以 $d$ 为 $\mathcal{T}'$ 中心减去 $\mathcal{T}$ 中心计算 $w$。逐个 head 与候选集合整体分别进行。若 patching 候选集合使 $w$ 接近 1，说明这些 head 的输出足以携带目标位置。

## 9. C5：行列答案

行列序数需要对实例计数，读取方式可能与 point 不同。本部分在 C2 之后开展，样本为 T6 中行列答对的网格。

1. **像素层面插行。** 在目标所在行的上方插入一条高度为一个实例单元的空白带，画布相应增高。目标的行序数不变，像素位置下移一个单元。
2. **position IDs 层面插行。** 像素不变，把目标上方插入点以下全部 token 的 $p^{h}$ 增加一个单元对应的 merged 行数（64-pixel 单元为 2），图像之后的文本 IDs 相应顺延。
3. **attention 记录。** 生成行序数 token 时，记录候选 head 对每行首个实例所在 token 的 attention，检查模型是否依次读取各行。

若模型对实例计数，两种插行均不改变行答案；若模型按目标的位置与单元尺度换算行号，像素插行使行答案加 1，position IDs 插行在 S2 起作用时使行答案加 1。同时比较同一图像在行列格式与 point 格式下候选 head 的 attention，检查实验 1.2 中“行列正确而中心点错误”是否对应两种格式读取了不同的位置信息。

## 10. 指标汇总

| 指标 | 定义 | 部分 |
| --- | --- | --- |
| 基线准确率 | point 命中率、中心误差、bbox IoU≥0.5 比例、行列 exact match | C0 |
| 目标富集度 $E$ 与有效富集度 | 见 5.2 | C1 |
| 最大位置命中率 | attention 最大的 visual token 属于目标的比例 | C1 |
| 归因占比 $\rho^{\ell}$ | 见 5.4 | C1 |
| 位移权重 $w$ 与正交残差 $e_{\perp}$ | 见 6.1 | C2–C4 |
| 识别 logit 差 | “是否有红色圆”的 Yes 与 No logit 差 | C3、C4 |
| 行答案变化率 | 插行后行答案改变的比例 | C5 |

比例给出分母与 Wilson 区间，连续指标给出以图像为单位的 bootstrap 95% 区间。

## 11. 可视化

1. **head 富集度热图。** 纵轴为 8 个 full attention 层，横轴为 head，颜色为坐标首位 token 上的平均 $\log E$，分别给出 point、bbox 与行列。
2. **attention 叠加图。** 对候选 head，把生成横、纵坐标首位 token 时的 attention 叠加在输入图像上，并标出真值与模型输出。
3. **归因曲线。** 横轴为层，纵轴为 $\rho^{\ell}$，答对与答错样本各一条曲线，竖线标出 full attention 层。
4. **平移响应曲线。** 横轴为预测位移，纵轴为观测位移，S1 与 S2 的全局平移各一组点，斜率为 $w$，并画出 $w = 1$ 与 $w = 0$ 的参考线。
5. **冲突设计条形图。** 四个条件的 $w$ 与区间，并标出 $w_{1} + w_{2}$。
6. **输出位移箭头图。** 在画布上以箭头从基线输出指向各干预条件下的输出，直观显示输出跟随哪一个来源。
7. **消融散点图。** 横轴为识别 logit 差的变化，纵轴为坐标误差的变化，候选 head 与随机 head 以不同颜色区分。

## 12. 结果模式与解释

下表给出各机制在每种干预下的预期输出。“A”表示输出留在原目标位置，“B”表示输出移向 $\mathcal{T}'$，“平移”表示输出随施加的平移量移动，“不变”表示全局平移不改变输出。

| 机制 | S2 全局平移 | S2 区域互换 | S1 全局平移 | S1 区域互换 | 内容交换 | 序列置换 |
| --- | --- | --- | --- | --- | --- | --- |
| M1 S2，以问题 token 为参照 | 平移 | B | 不变 | A | B | A |
| M2 S2，以图像首 token 或边界为参照 | 不变 | B | 不变 | A | B | A |
| M3 S1，位置由 visual embedding 携带 | 不变 | A | 平移 | B | A | A |
| M4 S4，位置来自序列顺序 | 不变 | A | 不变 | A | B | B |
| M5 S3，位置来自视觉内容（仅自然图像） | 不变 | A | 不变 | A | 不适用 | A |

实际结果可能是多种机制的混合，此时各干预的 $w$ 介于 0 与 1 之间，冲突设计给出各来源的相对权重。结果与后续工作的对应关系如下：

| 结果模式 | 解释 | 对问题 5 的含义 |
| --- | --- | --- |
| $w_{2}$ 接近 1，S2 全局平移不改变输出 | 坐标由图像内部的相对 position IDs 决定（M2） | 窗口 token 保留其在原图中的相对 position IDs，且窗口与整图的 IDs 处于同一坐标系 |
| S2 全局平移使输出平移 | 坐标依赖图像与问题之间的 ID 差（M1） | 重读窗口时需要保持图像与后续文本的 ID 关系，或在训练中重新校准 |
| $w_{1}$ 接近 1 | 位置由缓存的 visual embedding 携带（M3） | 缓存特征本身已携带位置，position IDs 的分配对坐标影响较小 |
| 内容交换与序列置换均使输出移动 | 序列顺序参与位置读取（M4） | 窗口 token 的排列顺序需要与原图保持一致 |
| 候选 head 的 attention 指向真值而输出错误 | 读取正确而后续表达失败 | 改进重点在坐标表达，局部重读的作用有限 |
| 候选 head 的 attention 在答错样本中偏离目标 | 读取阶段选错位置 | 局部重读可能通过提高目标区域的分辨率改善读取 |
| 行答案对像素插行敏感、对 ID 插行不敏感 | 行号由像素位置与单元尺度换算，未使用 position IDs | 行列答案与坐标答案分别设计窗口策略 |

## 13. 运行与产物

1. 实验程序为 `analysis_tools/position/coordinate_readout.py`，刺激生成复用 `analysis_tools/common/stimuli.py`，作图程序为 `analysis_tools/position/coordinate_readout_figures.py`；Bash 入口为 `scripts/analysis/position/coordinate_readout.sh`，模型、部分选择（C0–C5）、干预参数与随机种子集中在脚本顶部。
2. 产物保存于 WORK_DIR 的 `analysis/position/<日期>_coordinate_readout/`：`cases.json` 记录图像、真值与区域定义；`baseline/` 保存 C0 的 prompt、原始回答与评分；`attention/` 保存候选 head 的 attention 与归因；`interventions/` 保存每个干预条件的 position IDs 摘要、原始回答与解析坐标；`results.json` 保存全部指标与区间；`experiment.json` 保存模型版本、attention 实现、代码摘要与运行参数。
3. 每个干预条件保存完整的原始回答，解析失败计入分母并单独统计，不以诊断解析覆盖严格解析。

## 14. 解释范围

1. attention 权重描述读取的分配，不等于信息流；有效贡献与梯度归因也只是局部近似。关于位置来源的因果结论以 C2–C4 的干预结果为准，C1 只用于提出候选 head 与层。
2. position IDs 互换、ViT 位置互换和序列置换会产生训练中未出现的输入，干预后的输出变化可能部分来自分布外效应。每种干预都与同等幅度的自然变化（真实移动目标）对照，并报告 $e_{\perp}$；$e_{\perp}$ 较大的条件不用于估计来源权重。
3. 单目标画布排除了内容线索，结论能否推广到自然图像由 6.6 检验；行列答案的结论只适用于规则网格。
4. 本实验只分析冻结的 Qwen3.5-9B。Qwen3-VL-8B-Instruct 具有 DeepStack，其位置路径可能不同，在主模型结论确定后再以相同设计对照。
