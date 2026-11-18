# 异常实体定位的计算过程：设计（实验 2.7 第三轮）

实验日期：2026-11-05。本文属于[视觉细节表征](plan.md)研究，承接[实验 2.7 第二轮](results_2026-11-03_search_asymmetry_r2.md)。主模型为 Qwen3.5-9B，冻结、bfloat16、SDPA，thinking off，问法与前两轮相同（直接回答行列坐标）。

## 1. 问题与动机

第二轮用一个人工构造的方向（oddness 方向 $u$）描述“目标与众不同”这一信号。该方向由两张图像的平均对比相加得到，模型在一次前向中只看到一张图像，也不计算“其余实体残差的平均”，因此 $u$ 只能作为线索，不能作为机理。本轮要求把“定位异常实体”这一功能对应到模型在单张图像上真实执行的运算：注意力 logit、Gated DeltaNet 的状态更新、残差相加。具体回答四个问题：

1. **读取。** 答案位置的哪些 head 从目标处读取位置？它们的注意力 logit 在目标上为什么更高？这个更高的 logit 由哪些上游组件写入（ViT 与 merger、某几层 DeltaNet、某几层 full attention、某几层 MLP）？
2. **比较。** 对没有 ViT 专属检测器的特征，decoder 中的比较由哪一种运算完成：Gated DeltaNet 的 delta rule 预测误差（当前 token 的 value 与状态对它的预测之差），还是 full attention 对前文实体的直接读取？比较是否与特征无关？
3. **光栅顺序。** 光栅在前实体太少导致的失败来自 DeltaNet 的递归、full attention 的因果 mask，还是读取 head 的位置先验？若 ViT 已经给出强信号，decoder 的因果方向是否就不再重要？decoder 的开头效应位于哪里？
4. **两种模式。** 按是否存在 ViT 专属检测器把特征分成两类，展示两类目标在差异幅度与前文实体数量上的剂量—反应关系，并给出机制解释。

## 2. 符号与模型运算

以下所有层号从 1 开始。decoder 第 $4, 8, \dots, 32$ 层为 full attention 层，其余为 Gated DeltaNet 层。

### 2.1 读取 head 的注意力 logit 及其精确分解

记答案文本中预测行数字的位置为行查询位置 $p$（预测列数字的位置同理）。读取 head $(l, h)$ 对图像 token $j$ 的注意力 logit 为

$$
z_{j} = \frac{1}{\sqrt{d}} \, q_p^{\top} \, \mathrm{RoPE}_j\!\left( \mathrm{N}_k\!\left( W_K \, \mathrm{N}_{\mathrm{in}}(x_j) \right) \right),
$$

其中 $x_j \in \mathbb{R}^{4096}$ 为第 $l$ 层输入处 token $j$ 的残差，$\mathrm{N}_{\mathrm{in}}$ 为该层输入 RMSNorm，$W_K$ 为该 head 所在 key 组的投影（$256 \times 4096$），$\mathrm{N}_k$ 为 key 的 RMSNorm，$\mathrm{RoPE}_j$ 为位置 $j$ 的旋转，$q_p \in \mathbb{R}^{256}$ 为查询位置的 query，$d = 256$。残差是若干组件输出之和：

$$
x_j = x_j^{(0)} + \sum_{m < l} \left( a_j^{(m)} + f_j^{(m)} \right),
$$

其中 $x_j^{(0)}$ 为 decoder 输入（图像 token 处即 merger 输出），$a_j^{(m)}$ 为第 $m$ 层序列混合子层（DeltaNet 或 full attention）的输出，$f_j^{(m)}$ 为第 $m$ 层 MLP 的输出。RMSNorm 只对向量按其均方根缩放，若把缩放系数固定为它在实际前向中的值，$\mathrm{N}_{\mathrm{in}}$ 与 $\mathrm{N}_k$ 都变成线性映射；RoPE 本身是线性旋转。于是 $z_j$ 可以精确写成各组件贡献之和：

$$
z_j = \sum_{c} z_j^{(c)}, \qquad c \in \left\{ x^{(0)}, a^{(1)}, f^{(1)}, \dots, a^{(l-1)}, f^{(l-1)} \right\}.
$$

这一分解只把模型的实际运算按残差加法拆开，没有引入任何人为方向。对实体 $e$（4 个 token）取 $z^{(c)}$ 的平均，记为 $z_e^{(c)}$。目标 $t$ 的读取优势定义为

$$
\Delta z^{(c)} = z_t^{(c)} - \operatorname{mean}_{e \ne t} z_e^{(c)},
$$

即组件 $c$ 让读取 head 对目标的 logit 比对其他实体高出多少。所有组件的 $\Delta z^{(c)}$ 之和就是读取 head 偏好目标的全部原因。

### 2.2 Gated DeltaNet 的预测误差

Gated DeltaNet 的每个 value head 维护状态 $S_t \in \mathbb{R}^{d_k \times d_v}$（$d_k = d_v = 128$），按光栅顺序更新：

$$
\hat{v}_t = \left(\alpha_t S_{t-1}\right)^{\top} k_t, \qquad e_t = v_t - \hat{v}_t, \qquad S_t = \alpha_t S_{t-1} + \beta_t \, k_t e_t^{\top}, \qquad o_t = S_t^{\top} q_t ,
$$

其中 $k_t, q_t$ 为 L2 归一化后的 key 与 query，$v_t$ 为 value，$\alpha_t \in (0,1)$ 为衰减门，$\beta_t \in (0,1)$ 为写入强度。$\hat{v}_t$ 是状态以当前 key 为线索对 value 的预测，$e_t$ 是预测误差，状态只写入误差（delta rule）。这一更新是对目标 $\lVert S^{\top} k - v \rVert^2$ 做一步学习率为 $\beta_t$ 的在线最小二乘，因此同一对 $(k, v)$ 重复出现时误差按 $(1 - \beta)$ 的比例几何衰减。这是一个“与前文比较”的内建运算：多数实体在重复几次后被状态预测准确，误差变小；与前文不同的实体得不到准确预测，误差保持较大；序列最开头的实体没有任何可依赖的记忆，误差同样较大。

展开递推可得预测的来源分解：

$$
\hat{v}_t = \sum_{j < t} w_{tj} \, e_j, \qquad w_{tj} = \exp\left(G_t - G_j\right) \beta_j \left(k_j^{\top} k_t\right),
$$

其中 $G_t = \sum_{m \le t} \log \alpha_m$ 为累计对数衰减。$w_{tj}$ 是 token $j$ 写入的误差在 $t$ 处被取回的权重，它同时含有“key 相似”（$k_j^{\top} k_t$）与“未被遗忘”（衰减乘积）两个因素。本轮记录：

1. 相对预测误差 $\rho_t = \lVert e_t \rVert / \lVert v_t \rVert$，按实体与 head 平均；
2. 预测对 value 的覆盖 $\pi_t = \hat{v}_t^{\top} v_t / \lVert v_t \rVert^2$；
3. 目标 token 的预测来源份额 $w_{tj} \, e_j^{\top} \hat{v}_t / \lVert \hat{v}_t \rVert^2$，对每个实体的 token 求和，所有来源之和为 1。

### 2.3 单图反事实

每个异常图像都配一张反事实图像：把目标重绘为多数实体，抖动、朝向、空白格完全相同，称为“去异常图像”。两者只在目标格的像素上不同，因此任何差异都由目标引起。特征存在目标（$A$ 在 $B$ 中）的去异常图像全部为 $B$，特征缺失目标（$B$ 在 $A$ 中）的去异常图像全部为 $A$。

## 3. 实验

### 3.1 刺激

| 集合 | 内容 | 规模 |
| --- | --- | --- |
| atlas | 6 × 6，11 对特征（Q 与 O、开口与闭合圆、倾斜与竖直线、弯曲与笔直线、饱和与不饱和色、大与小圆、端点短横、四臂与三臂十字、方框内点、圆角方框与圆、旋转与直立十字），subtle 水平，两种目标；Q 与 O、开口与闭合圆另有 salient 水平 | 26 条件 × 24 对 |
| prev | 6 × 6，目标固定在第 4 行第 4 列（光栅序号 22，前面有 21 个格），前面的格随机留空，使光栅在前实体数 $n_{\mathrm{prev}} \in \{0, 2, 5, 10, 21\}$；目标的序列位置、行与列不变 | 12 条件 × 5 × 12 对 |
| mag | 6 × 6，Q 的尾长 3、6、10、14 像素，开口圆缺口 2、4、8、12 像素，两种目标，每格一个目标 | 16 条件 × 36 对 |
| shift | 6 × 6 均匀图像，前 $k \in \{0, 6, 12, 18\}$ 个格留空 | 2 对 × 2 种多数实体 × 4 × 6 张 |

### 3.2 阶段

1. **trace（相关性测量，单图）。** 对每张异常图像与去异常图像做一次 teacher forcing 前向。DeltaNet 层改由逐 token 的精确递推执行（与库实现的数值一致性在 selftest 中检验），记录 2.2 节的量；读取 head 的 logit 按 2.1 节分解；同时记录所有 full attention head 在两个查询位置的逐实体注意力，以及 full attention 第 4–16 层中图像 token 对第一个图像 token、文本前缀、本实体与前文实体的注意力。
2. **knock（因果分离）。** 在单张图像上执行四类干预：
   1. DeltaNet 记忆清除：第 1–15 层所有 DeltaNet 层在全部图像 token 上令 $\alpha_t = 0$，每个图像 token 的 DeltaNet 输出只依赖它自己（以及长度为 4 的短卷积窗口）；另有第 1–7 层、第 9–15 层两个子集。
   2. full attention 跨实体阻断：第 4、8、12、16 层中图像 token 作为 query 时不能注意其他实体的 token（可以注意本实体与文本前缀）；另有单层与子集版本。答案位置的 query 不受影响。
   3. 两者同时执行。
   4. DeltaNet 状态预热：在第一个图像 token 处，把每个 DeltaNet 层的状态替换为同一提示下另一张图像读完后的状态。供体分三种：同一特征对的去异常图像（多数实体相同）、另一特征对的均匀图像、空白图像。若开头失败来自 DeltaNet 记忆尚未建立，前者应恢复开头位置的定位，后两者不应恢复或恢复较少。
   读出为 teacher forcing 下正确答案的分数 $S$、行列数字的 argmax 是否都指向目标（命中），以及读取 head 从行查询位置落在每个实体上的注意力。
3. **patch（必要性检验）。** 在异常图像的前向中，把某一个组件在目标 token 上的输出替换为去异常图像中的值（噪声化），逐层逐类型执行，测量命中与读取 head 对目标的注意力。它与 2.1 节的直接贡献互补：直接贡献只统计组件对读取 key 的直接写入，噪声化同时包括经由后续层的间接作用。
4. **circuit（head 与神经元）。** 在答对且目标位于第 7 格以后的 80 对比较类与 40 对检测器类图像上，于目标 token 处逐个阻断 full attention 第 4、8、12 层每个 head 的跨格读取、逐个替换其输出、逐个清除 DeltaNet 第 9–15 层每个 value head 的记忆；按神经元经残差到读取 key 的直接贡献对 MLP 8–15 的全部神经元排序，对前 $k$ 个神经元与等量随机神经元做必要性（异常图像中换成去异常值）与充分性（去异常图像中换成异常值）检验；同时记录第 8 层 head 的 QK 与读取 head 的 query 与目标 key 变化。具体协议见结果文档第 2.7 节。

### 3.3 预测

| 假说 | 预测 |
| --- | --- |
| 特征存在目标的读取优势主要由 decoder 输入（ViT 与 merger）写入 | $\Delta z^{(x^{(0)})}$ 占总优势的大部分；DeltaNet 记忆清除与跨实体阻断对命中几乎没有影响 |
| 特征缺失目标的读取优势由 DeltaNet 的预测误差产生 | 目标的 $\rho$ 高于同图多数实体，且依赖 $n_{\mathrm{prev}}$；DeltaNet 记忆清除使命中下降，跨实体阻断影响较小；同类预热恢复开头位置 |
| 开头失败来自读取 head 的位置先验 | 均匀图像中读取 head 对前几个实体的注意力低于均匀分布；该先验随第一批实体移动（shift 集），而不是固定在第一个图像 token 或第一行 |

### 3.4 与第二轮的衔接

第二轮的 6 个读取 head（第 16 层 head 6、11、2，第 20 层 head 2、11、10）在第二轮由加方向实验选出。本轮在不使用 $u$ 的前提下重新选择：按真实异常图像中行查询位置落在目标上的注意力排序（第二轮 capture 数据，528 张），前 6 名与上述 head 完全一致，因此本轮沿用这 6 个 head，选择依据改为模型在真实输入上的注意力。
