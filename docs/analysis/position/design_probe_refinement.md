# 实验 3.4：位置 probe 完善

本实验属于[位置信息与坐标读取](plan.md)，承接实验 3.3。实验 3.3 在 360 张 RefCOCO 图像上发现，decoder 的位置预测误差在第 14 层最低（0–1000 坐标下 MAE 为 92.33），随后升至第 32 层的 185.29；但该 probe 使用 256 维固定随机投影与 ridge 回归，只有 514 个参数，训练图像只有 240 张，最佳层的精确网格准确率仅为 5.59%。本实验提高 probe 容量、扩大数据，并加入相对位置 probe，确定位置可读性的上限与形式。

## 1. 研究问题与动机

实验 3.3 的结果存在三种尚未区分的解释：

1. **读出容量不足。** 随机投影把 4096 维 hidden state 压缩到 256 维，会丢失大量线性可分的方向；即使深层保留了同样多的位置信息，若位置方向在深层分布于更多维度，固定投影后的读出误差也会上升。
2. **训练数据不足。** 240 张训练图像对应 11,520 个训练 token，对于更高容量的 probe 不足以稳定估计参数。
3. **位置以相对形式保留。** Decoder 的 full attention 通过 MRoPE 在 query 与 key 的内积中临时注入相对位置，residual stream 本身不需要写入绝对坐标。若深层 hidden state 只保留与相对位置相容的结构，绝对坐标的读出误差会上升，而两个 token 之间位移的读出误差保持较低。

因此本实验回答三个问题：

1. 在足够容量与数据量下，各层 visual token 的绝对槽位位置能被读出到什么精度；实验 3.3 中深层误差上升的趋势是否保持。
2. 各层的绝对位置读出误差与相对位移读出误差如何变化，二者的差异是否在深层增大。
3. 位置读出有多少来自视觉内容本身提供的线索；在内容无法提供位置线索的图像（均匀画布与重复图案）中，位置是否仍可读出。

## 2. 总体设计

实验由四组读出组成，共用同一套图像、特征提取与数据划分：

| 读出组 | 内容 | 回答的问题 |
| --- | --- | --- |
| A 容量序列 | 随机投影 ridge、全维 ridge 与三种宽度的 MLP probe 在全部层上的绝对位置读出 | 问题 1 |
| B 学习曲线 | 在选定层上改变训练图像数，测量误差是否饱和 | 问题 1 |
| C 相对位置 | 由同一图像中两个 token 的特征预测二者位移 | 问题 2 |
| D 内容对照 | patch content 基线、均匀画布与重复图案上的读出 | 问题 3 |

实验只执行 prefill，不生成回答。输入格式沿用实验 3.3：图像在前，随后为 `Describe the image briefly.`，thinking off。图像位于请求文字之前，因此 visual token 的 hidden state 不读取图像之后的文本。

## 3. 数据

### 3.1 自然图像

1. **来源与去重。** 从本地 RefCOCO 的全部 split 中读取图像，按原始图像字节的 SHA-256 去重；不足 3000 张时从 COCO val2017 补充，并与 RefCOCO 图像再次去重。记录每张图像与实验 3.3 所用 360 张图像的重合情况。
2. **划分。** 以固定种子划分为 2400 张训练图、300 张验证图和 300 张测试图。同一图像的全部 token 只出现在一个划分中。
3. **预处理。** 最长边缩放到 512 pixels，两边取至 32 的整数倍，与实验 3.3 一致，使两次实验的误差可以直接比较。另将 300 张测试图按最长边 768 pixels 预处理，作为未见分辨率的测试条件，probe 只用 512 pixels 的训练图拟合。
4. **token 采样。** Ridge probe 使用每张图的全部 visual token；MLP probe 与相对位置 probe 使用每张图均匀无放回采样的 64 个 token。采样方式的差异由存储量决定，见 5.2。

### 3.2 合成图像

合成图像来自[共享合成刺激集](../stimuli.md)，只用于测试与对照：

1. **重复图案。** 不含异常实例的标准网格，网格规模为 6×6、10×10、20×20（前两者每个实例占 64×64 pixels，20×20 每个实例占 32×32 pixels），开发模板（person、ring、square、arrow）在每种网格规模下各 16 张。16 张图由 4 种位置偏移条件与 4 种画布留白组合而成：位置偏移条件为网格相对 token 边界平移 0、8、16 pixels，以及每个实例独立随机平移不超过 4 pixels；画布留白为在网格外额外增加 0、64、128、192 pixels，使网格在画布中的相对范围随之变化。重复图案中每个实例的内容相同，内容只提供实例内部的相对位置，不提供实例在整图中的位置。
2. **均匀画布。** 纯白与纯灰画布，尺寸覆盖上述重复图案的画布尺寸，分为三个互不重叠的尺寸组：正方形小尺寸（边长 384–576 pixels）、正方形大尺寸（边长 640–832 pixels）与 8 种非正方形尺寸，每个尺寸各 1 张白色与 1 张灰色画布，共 32 张。均匀画布中所有 patch 的像素相同，因此 ViT 输入端 token 之间的差异只来自 learned absolute position embedding。
3. **保留模板。** 星形与房屋网格在每种网格规模下各 16 张，生成方式与开发模板相同，只用于合成图内部训练后的最终测试。

## 4. 符号约定

| 符号 | 形状 | 含义 |
| --- | --- | --- |
| $i$ | 标量 | 图像索引 |
| $H_i, W_i$ | 标量 | 图像 $i$ 在提取位置上的网格行数与列数（ViT 为 patch 网格，merger 与 decoder 为 merged 网格） |
| $(r, c)$ | 标量对 | token 的零起始行列索引 |
| $\ell$ | 标量 | 提取位置，包括 patch content、ViT 各层、merger 与 decoder 各层 |
| $x^{\ell}_{i,r,c}$ | $\mathbb{R}^{D}$ | 提取位置 $\ell$ 上 token $(r,c)$ 的 hidden state，ViT 的 $D=1152$，merger 与 decoder 的 $D=4096$ |
| $y_{i,r,c}$ | $\mathbb{R}^{2}$ | token 中心的图像坐标（0–1000），横坐标在前 |
| $N$ | 标量 | 训练 token 数 |

图像坐标标签沿用实验 3.3 的定义：

$$
y_{i,r,c} = 1000\left(\frac{c + 1/2}{W_i},\ \frac{r + 1/2}{H_i}\right).
$$

该标签把 token 中心映射到图像宽高的千分比，使不同尺寸图像的误差使用同一单位，并与模型输出 point 与 bbox 所用的 0–1000 坐标一致。

## 5. 方法

### 5.1 容量序列

容量序列包含六种 probe，按参数量递增：

| probe | 输入 | 映射 | 参数量（$D=4096$） |
| --- | --- | --- | --- |
| RP256 | 256 维固定高斯随机投影 | ridge | 514 |
| RP1024 | 1024 维固定高斯随机投影 | ridge | 2,050 |
| Full ridge | 全部 $D$ 维 | ridge | 8,194 |
| MLP64 | 全部 $D$ 维 | 单隐藏层 MLP，宽度 64 | 约 0.26M |
| MLP256 | 全部 $D$ 维 | 单隐藏层 MLP，宽度 256 | 约 1.05M |
| MLP1024 | 全部 $D$ 维 | 单隐藏层 MLP，宽度 1024 | 约 4.2M |

RP256 复现实验 3.3 的设置，作为两次实验之间的衔接。全部输入先按训练集各维均值与标准差标准化。

**全维 ridge。** 设标准化后的训练特征矩阵为 $X \in \mathbb{R}^{N \times D}$，标签矩阵为 $Y \in \mathbb{R}^{N \times 2}$，权重为 $B \in \mathbb{R}^{D \times 2}$，偏置为 $b \in \mathbb{R}^{2}$。优化目标与实验 3.3 相同：

$$
\min_{B, b}\ \left\| X B + \mathbf{1} b^{\mathsf T} - Y \right\|_F^2 + \lambda \left\| B \right\|_F^2 .
$$

其中 $\mathbf{1} \in \mathbb{R}^{N}$ 为全 1 向量，$\lambda$ 为正则化系数，偏置不受正则化。将标签中心化后，闭式解只依赖两个充分统计量：

$$
C = X^{\mathsf T} X \in \mathbb{R}^{D \times D}, \qquad U = X^{\mathsf T} Y \in \mathbb{R}^{D \times 2}, \qquad B_\lambda = Q (\Lambda + \lambda I)^{-1} Q^{\mathsf T} U .
$$

其中 $C = Q \Lambda Q^{\mathsf T}$ 为特征分解，$Q \in \mathbb{R}^{D \times D}$ 为正交矩阵，$\Lambda$ 为特征值对角阵。采用这一形式有两点原因：第一，$C$ 与 $U$ 可以在前向传播时逐图累加，不需要保存全部 token 的特征，从而使每张图的全部 token 都能参与训练；第二，一次特征分解即可得到全部 $\lambda$ 的解，$\lambda$ 在 $10^{-2}$ 至 $10^{6}$ 之间按对数等距取 17 个值，依据验证集均方误差选择。均值与标准差同样由逐图累加的一阶、二阶矩得到。

**MLP probe。** 单隐藏层 MLP 的形式为

$$
\hat y = W_2\, \operatorname{GELU}(W_1 x + b_1) + b_2 ,
$$

其中 $W_1 \in \mathbb{R}^{m \times D}$，$b_1 \in \mathbb{R}^{m}$，$W_2 \in \mathbb{R}^{2 \times m}$，$b_2 \in \mathbb{R}^{2}$，$m$ 为隐藏层宽度。训练使用 AdamW 与均方误差，batch size 为 1024，最多 25 个 epoch，并以验证集误差 early stopping（patience 为 4 个 epoch）。MLP256 的学习率与 weight decay 在验证集上从 $\{3\times10^{-4}, 10^{-3}, 3\times10^{-3}\} \times \{0.01, 0.1, 1\}$ 的 3×3 网格中选择；MLP64 与 MLP1024 沿用 MLP256 选出的 weight decay，只在验证集上选择学习率。MLP 用于检验位置是否以非线性方式编码：若 MLP 显著优于全维 ridge，说明位置信息存在但不在线性可读的方向上。

更高容量的 probe 也更容易利用视觉内容与位置的统计相关性（例如天空多位于图像上方），因此每种 probe 都在 patch content 上拟合同样的对照（见 5.5），以 probe 族为单位比较位置增益。

### 5.2 特征提取与存储

Ridge probe 对全部 62 个提取位置逐图累加 $C$ 与 $U$，每个 4096 维提取位置的 $C$ 以 float64 存储约需 134 MB，不保存 token 特征。前向传播时，每张图的统计量先在 GPU 上以 float32 累加：特征减去该提取位置第一张图的 token 均值，标签减去 500，以避免大均值造成的舍入误差；每 25 张图将平移后的统计量换算回原始矩并加入 CPU 上的 float64 累加器。测试图像的 ridge 预测在第二次前向传播中由已拟合的 probe 直接计算。MLP probe 与相对位置 probe 需要保存特征，只在以下提取位置进行：patch content、加入 position embedding 后的 ViT 输入、ViT 第 9、18、27 层、merger，以及 decoder 第 3、4、5、7、8、9、11、12、13、14、15、16、17、20、24、28、32 层；其中第 4、8、12、16 层 full attention 两侧的 Gated DeltaNet 层用于比较两类层的读出精度。每个 decoder 提取位置的训练特征约 1.3 GB（2400 张图 × 64 个 token × 4096 维 × float16）。

### 5.3 学习曲线

学习曲线在 merger 与 decoder 第 8、14、24、32 层上进行。训练图像按固定顺序分为 32 组，每组 75 张；训练集大小取 75、150、300、600、1200、2400 张，由连续的组构成，因而相邻大小的训练集互相嵌套。每个大小以 3 种不同的组顺序重复，验证集与测试集固定。Ridge 只需对相应组的充分统计量求和，MLP256 在每个子集上重新训练。

饱和判据为：训练图像数从 1200 增至 2400 时，测试 MAE 的相对下降小于 5%。若最佳 probe 在主要层上未饱和，本实验的误差只作为可读性的上界估计，并在报告中注明。

### 5.4 相对位置 probe

相对位置 probe 由同一图像中两个 token 的特征预测二者的位移。设一对 token 为 $a$ 与 $b$，其特征为 $x_a, x_b \in \mathbb{R}^{D}$，位移标签为

$$
\Delta = y_b - y_a \in \mathbb{R}^{2} .
$$

每张图从 64 个采样 token 中抽取 256 对，按位移长度分为 8 个等宽区间分层抽样，使远近位移的样本数平衡。比较三种读出，前两种如下，第三种单独说明：

- **R1 差分 ridge。** $\hat\Delta = B^{\mathsf T}(x_b - x_a)$，其中 $B \in \mathbb{R}^{D \times 2}$。若 hidden state 中存在线性可读的绝对坐标 $B^{\mathsf T} x$，则 R1 可以读出位移，因此 R1 测量的是“由线性绝对位置推出的位移精度”，作为相对位置读出的基线。
- **R2 成对 MLP。** 输入为 $[P x_a;\ P x_b;\ P x_a \odot P x_b]$，其中 $P \in \mathbb{R}^{512 \times D}$ 为训练集上的 PCA 投影，$\odot$ 为逐元素乘积，经宽度 512 的单隐藏层 MLP 输出 $\hat\Delta$。R2 可以近似任意成对函数，作为相对位置可读性的高容量估计。

**R3 Fourier 双线性读出。** 设目标为位移的 Fourier 特征

$$
\phi(\Delta) = \big[\cos(\omega_k \Delta_x),\ \sin(\omega_k \Delta_x),\ \cos(\omega_k \Delta_y),\ \sin(\omega_k \Delta_y)\big]_{k=1}^{K} \in \mathbb{R}^{4K},
$$

其中 $\omega_k = \pi k / 1000$，$K = 8$，最低频率的周期为 2000 个坐标单位，大于位移的取值范围，因此 $\phi$ 在位移范围内可以唯一解码。$\phi$ 的第 $j$ 个分量由一个低秩双线性形式预测：

$$
\hat\phi_j = (A_j P x_a)^{\mathsf T} (G_j P x_b) + c_j, \qquad A_j, G_j \in \mathbb{R}^{\rho \times 512},\quad \rho = 16 .
$$

其中 $P$ 与 R2 相同，为白化后的 PCA 投影，$c_j$ 为标量偏置。等价地，特征空间中的双线性矩阵为 $P^{\mathsf T} A_j^{\mathsf T} G_j P$，秩不超过 16；先投影到 512 维使参数量从约 4.2M 降至约 0.5M，在 2400 张训练图上可以稳定训练。$A_j$ 与 $G_j$ 以小方差初始化，使初始输出接近 0。

预测位移取候选位移集合中使 $\phi(\Delta)^{\mathsf T} \hat\phi$ 最大的值，候选集合为步长 5 个坐标单位的二维网格。

R3 的形式来自 RoPE 的计算方式。单个 head 中，旋转后 query 与 key 的内积可以写成

$$
\tilde q_m^{\mathsf T} \tilde k_n = \sum_{u} \Big[ \alpha_u(q_m, k_n) \cos\big(\theta_u (n - m)\big) + \beta_u(q_m, k_n) \sin\big(\theta_u (n - m)\big) \Big],
$$

其中 $q_m, k_n$ 为位置 $m$ 的 query 与位置 $n$ 的 key（旋转前），$\tilde q_m, \tilde k_n$ 为旋转后的向量，$u$ 遍历被旋转的二维通道对，$\theta_u$ 为第 $u$ 个通道对的旋转频率，$\alpha_u$ 与 $\beta_u$ 分别是 query 与 key 在第 $u$ 个通道对上的点积与叉积，均为 query 与 key 的双线性函数。若 decoder 在 residual stream 中以相位形式保存位置，坐标本身是相位的非线性函数，线性绝对读出会失败；而两个 token 的双线性组合可以直接给出相位差的余弦与正弦。R3 因此检验 hidden state 是否保存了可被 attention 内积使用的相对位置结构。

相对位置的指标为相对优势

$$
\Gamma(\ell) = \mathrm{MAE}_{\mathrm{R1}}(\ell) - \min\big(\mathrm{MAE}_{\mathrm{R2}}(\ell),\ \mathrm{MAE}_{\mathrm{R3}}(\ell)\big),
$$

其中三项 MAE 均为位移预测的平均绝对误差。$\Gamma(\ell)$ 接近 0 表示位移可以完全由线性绝对位置解释；$\Gamma(\ell)$ 显著大于 0 表示存在线性绝对读出无法利用的相对位置信息。

### 5.5 内容对照

**patch content 基线。** 在 ViT patch embedding 之后、加入 position embedding 之前提取特征，以每种 probe 族分别拟合。定义位置增益

$$
g(\ell) = \mathrm{MAE}_{\mathrm{content}} - \mathrm{MAE}(\ell),
$$

其中两项使用同一 probe 族。$g(\ell)$ 扣除了视觉内容本身提供的位置线索，用于比较不同容量 probe 的结果。

**均匀画布与重复图案。** 以自然图像训练的 probe 直接测试合成图像（迁移测试）；另在重复图案上按整图划分训练与测试，并以保留模板作为最终测试（合成内部测试）。迁移测试检验读出方向是否依赖自然图像的内容分布，合成内部测试检验内容无法提供位置线索时位置是否仍可读出。

## 6. 指标

| 指标 | 定义 | 用途 |
| --- | --- | --- |
| 坐标 MAE | 横纵两个坐标绝对误差的平均，0–1000 单位 | 主指标，与实验 3.3 可比 |
| 横、纵 MAE | 分别计算两个坐标的 MAE | 检查两个方向是否同步变化 |
| $R^2$ | 见下式 | 与图像尺寸无关的拟合优度 |
| 精确网格准确率 | 网格坐标 probe 的预测取最近整数后行列均正确的比例 | 衡量能否精确定位到单个 token |
| 一格内准确率 | 行列误差均不超过一格的比例 | 衡量粗定位能力 |
| 位移 MAE 与 $\Gamma(\ell)$ | 见 5.4 | 相对位置可读性 |

决定系数定义为

$$
R^2 = 1 - \frac{\sum_{n} \left\| y_n - \hat y_n \right\|^2}{\sum_{n} \left\| y_n - \bar y \right\|^2},
$$

其中求和遍历全部测试 token，$\bar y \in \mathbb{R}^{2}$ 为训练标签均值。$R^2$ 等于 probe 解释的标签方差比例：常数基线为 0，完美预测为 1。网格坐标 probe 沿用实验 3.3 的网格标签单独拟合。全部连续指标报告以测试图像为单位的 2000 次 bootstrap 95% 区间；常数基线与打乱训练标签的基线对每种 probe 族分别报告。

## 7. 可视化

1. **逐层曲线。** 横轴为提取位置，纵轴为坐标 MAE，每种 probe 一条曲线并附 bootstrap 区间，ViT 与 decoder 分两栏，以竖线标出 full attention 层，以水平虚线标出 patch content 基线。
2. **容量—误差曲线。** 在 merger 与 decoder 第 8、14、24、32 层上，横轴为 probe 参数量的对数，纵轴为测试 MAE，同时画出同一 probe 族在 patch content 上的误差。
3. **学习曲线。** 横轴为训练图像数的对数，纵轴为测试 MAE，每层一条曲线，阴影为 3 次重复的范围。
4. **空间误差图。** 将测试 token 投到 8×8 归一化网格，绘制每格的平均误差，比较最佳 probe 在 merger、第 14 层与第 32 层的结果，色标与实验 3.3 一致。
5. **绝对与相对读出对照。** 同一坐标系中画出各层的绝对坐标 MAE、R1、R2、R3 的位移 MAE，并在下方画出 $\Gamma(\ell)$。
6. **合成图预测图。** 在均匀画布与重复图案上，以箭头从真实中心指向预测位置，展示 merger、第 14 层与第 32 层的读出。

## 8. 结果模式与解释

| 结果模式 | 解释 | 后续 |
| --- | --- | --- |
| 全维 ridge 或 MLP 使深层误差降到接近第 14 层的水平 | 实验 3.3 中深层误差上升主要来自随机投影的容量限制，深层仍保留绝对位置 | 3.5 中以深层 hidden state 作为坐标来源的候选 |
| 只有 MLP 在深层取得低误差 | 深层位置以非线性方式编码 | 3.5 的 head 分析关注 MLP 可读方向与 head 输出的关系 |
| 全部绝对读出在深层失败，$\Gamma(\ell)$ 在深层显著大于 0 | 深层以相对形式保留位置，与 MRoPE 的内积计算相容 | 3.5 重点检验 position IDs 的作用；问题 5 保留窗口的相对排列 |
| 学习曲线在 2400 张图时仍未饱和 | 误差只能作为可读性的上界 | 扩大数据或限定结论范围 |
| 高容量 probe 的 $g(\ell)$ 小于低容量 probe 的 $g(\ell)$ | 容量增益主要来自视觉内容线索 | 以合成图结果为主要依据 |
| 均匀画布上 decoder 各层仍能读出位置 | 位置信号不依赖视觉内容，来自 position embedding、序列顺序或二者的组合 | 3.5 的 C2 区分这些来源 |
| 迁移测试失败而合成内部测试成功 | 读出方向依赖内容分布，同一位置在不同内容下以不同方向编码 | 报告按图像类型分别给出 |
| 未见分辨率条件误差显著升高 | probe 依赖固定网格尺度 | 3.5 的样本保持与训练相同的分辨率 |

## 9. 运行与产物

1. 实验程序为 `analysis_tools/position/position_probe_refinement.py`，MLP、相对位置 probe、合成图内部 probe 与评估程序为 `analysis_tools/position/position_probe_refinement_probes.py`，作图程序为 `analysis_tools/position/position_probe_refinement_figures.py`，Bash 入口为 `scripts/analysis/position/position_probe_refinement.sh`，模型、数据规模、提取位置、probe 列表与随机种子集中在脚本顶部。
2. 产物保存于 WORK_DIR 的 `analysis/position/<日期>_probe_refinement/`，该路径链接到 DATA_DIR 下的同名存储目录（全部产物约 80 GB，超出 WORK_DIR 的用户配额）：`cases.json` 记录图像来源、hash、划分、尺寸与网格；`stats/` 保存 ridge 的充分统计量；`ridge/` 保存拟合后的 ridge 映射；`features/` 保存选定层的采样特征与合成图的全部 token 特征；`predictions/` 保存 ridge 的逐 token 测试预测；`probes/` 保存 MLP、相对位置与合成图内部 probe 的预测与日志；`results.json` 保存全部指标、区间与基线；`extract_trainval.json` 与 `extract_test.json` 保存模型版本、代码摘要与运行参数。
3. 实验 2.3 属性 probe 复用本实验的特征提取与数据划分代码。

## 10. 解释范围

本实验的全部结论描述外部 probe 的可读性。probe 读出低误差只说明位置信息可以从 hidden state 中恢复，不说明模型在生成坐标时使用该信息；高误差也不说明位置信息已经消失，只能说明在给定 probe 族与数据量下无法恢复。相对位置 probe 的成功说明 hidden state 中存在与位移相容的结构，该结构是否被 attention 使用需要由实验 3.5 的干预检验。自然图像与合成图像的结论分别报告，不合并为单一数值。
