# Qwen3.5 视觉表征的细节保留与可解释性分析

## 1. 研究问题与分析策略

本研究分析 Qwen3.5-9B 的 visual encoder 与 merger 对细粒度视觉信息的保留情况。核心问题是：当两幅图像只在一个局部细节上不同时，output of a layer in the visual encoder 与 visual embedding 是否仍然允许区分这一细节，以及该细节能否以正确的形状和位置被恢复。重复图案中的异常辨别提供了直接的检验场景，但研究范围同时包括细线、缺口、方向、连接关系与微小颜色差异。

分析按三种证据组织。首先，通过受控图像改动测量逐层特征响应，确定细节变化是否影响表征。其次，通过独立参考样本上的细节检索与 feature inversion，检验表征能否支持具体细节的恢复。最后，通过完整 VLM 的回答和 visual embedding 干预，检验 LLM backbone 是否使用了这些细节。第一阶段仅加载 visual encoder 与 merger，无需加载 LLM backbone 或训练生成模型。

“保留多少细节”需要由细节类型、像素尺度、位置、背景和读取方法共同定义。主要结果应是各阶段的细节恢复准确率与最小可辨尺度曲线，不能直接解释为原图信息量的百分比。表征对变化敏感、一个外部读取方法成功和完整 VLM 回答正确分别支持不同结论。

本文第 7 节报告已执行的合成图初步实验；其他实验为待验证的研究方案。

## 2. 架构与训练背景核查

### 2.1 本地 Qwen3.5-9B

核查对象为 checkpoint `c202236235762e1c871ad0ccb60c8ee5ba337b9a`，本地 Transformers 版本为 `5.17.0`。模型配置与实现给出以下结构。[1–2]

| 项目 | 本地配置或实现 |
| --- | --- |
| visual encoder 层数 | 27 个 Transformer blocks |
| visual encoder hidden size | 1152 |
| 空间 patch size | 16×16 pixels |
| temporal patch size | 2；单张静态图像在 patchify 时复制时间维度 |
| merger 输入 | 第 27 层输出，即 `blocks[26]` 的输出 |
| merger 空间分组 | 相邻 2×2 个 patches |
| merger 计算 | 每个 1152 维特征独立 LayerNorm，按固定空间顺序拼接，然后经过 Linear、GELU、Linear |
| 拼接宽度与输出宽度 | 4608 → 4608 → 4096 |
| visual embedding 的空间网格 | 输入 patch 网格的高和宽各减半 |
| DeepStack | 本地配置为空，当前 forward 没有中间层注入分支 |

设经过 processor 的静态 RGB 图像为：

$$
I\in\mathbb{R}^{H\times W\times3}.
$$

其中 H、W 是实际进入 visual encoder 的像素高和宽，均满足 processor 的空间对齐要求。定义 patch 数 N，以及第 l 个 Transformer block 的输出：

$$
N=\frac{HW}{16^2},\qquad
F_l(I)\in\mathbb{R}^{N\times1152},\qquad l\in\{1,\ldots,27\}.
$$

这里 F 的每一行对应一个输入 patch 位置，但经过 self-attention 后可以包含整图信息。以 G 表示 merger，第 27 层的输出被转换为：

$$
E(I)=G(F_{27}(I))\in\mathbb{R}^{(N/4)\times4096}.
$$

E 是输入 LLM backbone 的 visual embedding。它的每个网格位置与输入图像中的一个 32×32 区域对应，但这种对应关系不限定特征的 receptive field，也不意味着特征只能表达 32-pixel 以上的结构。

空间拼接是确定性的重排，保留四个 patch 的全部数值与相对顺序；可能改变可恢复信息的计算包括 LayerNorm 和后续 MLP。每组输入的标量数量为 4608，输出为 4096，因此不能由 token 数减少到四分之一推断信息量减少到四分之一。4608→4096 也不能直接推出丢失的是哪类细节：实际特征分布可能位于更低维的子空间或流形上。

### 2.2 与 Qwen3-VL 的区别

Qwen3-VL 技术报告明确说明 visual encoder 从 SigLIP-2 初始化，并以动态分辨率继续训练；8B 使用 SO-400M，较小的 2B/4B 使用 Large 变体。[3] 本地 Qwen3-VL-8B-Instruct 的主 merger 同样接收第 27 层输出，另外通过 DeepStack 使用代码索引 `[8,16,24]` 的 block 输出，即按从 1 开始计数的第 9、17、25 层，经各自的 merger 注入 LLM backbone。[4]

Qwen3.5-9B 的配置和代码能够确定运行时架构，但不足以独立证明其 visual encoder 的完整初始化与预训练来源。对 Qwen3-VL 已公开的 SigLIP-2 训练背景，应与 Qwen3.5 已核实的结构事实分别陈述。

图文对比预训练通常监督特定的 pooled representation 或 projection head 输出；它不保证每一层、每个 patch 的特征都能直接与文本 embedding 做有意义的相似度比较。经过 VLM 联合训练后，也不能默认原 SigLIP-2 的文本空间仍然与这些局部特征保持校准。merger 的输出维度与 LLM backbone 对齐，同样不保证 LM head 能直接解读任意单个 visual embedding。

## 3. 无需训练的第一阶段：逐层响应与异常定位

### 3.1 受控局部改动

构造一幅重复图案图像及其变体，只改变一个已知区域的细节，例如擦除手臂上的一小段、移动眼睛、改变线段方向或断开连接。两幅图像采用完全相同的尺寸、processor 配置和 patch 网格。优先采用无损 PNG，并保存实际变化的像素 mask。

以 I 表示基准图像，以 I′ 表示局部改动后的图像，二者形状相同。对一个分析阶段，用 Z 表示该阶段的特征矩阵，其形状为 n×d；n 是空间位置数，d 是通道数。位置 p 的响应定义为：

$$
R(p)=\sqrt{\frac{1}{d}\sum_{c=1}^{d}
\bigl(Z(I')_{p,c}-Z(I)_{p,c}\bigr)^2}.
$$

其中 c 是通道索引。R 将已知像素改动引起的特征变化映射回空间网格，直接显示改动在何处产生响应、响应是否扩散到其他图案，以及 merger 前后空间分布如何变化。一个局部改动也可能通过 self-attention 改变远处特征，因此远处响应本身不属于定位错误。

响应应同时报告相对于原特征范数的变化量、变化能量在目标区域的比例、响应最大区域的位置，以及完全相同输入重复 forward 的误差。各层量纲与特征几何不同，原始响应大小不直接代表细节保留量。独立归一化的热力图只适合显示空间分布。

该实验属于已知干预下的敏感性分析。比较时使用了基准图像，因此不能把响应最大位置正确报告为单图找不同准确率；非零响应也不足以证明缺口的形状、方向或连接关系可恢复。

### 3.2 单图异常定位

在单幅图像中，从每个小人覆盖的特征区域提取一个有空间顺序的向量。初步实验利用生成器提供的区域边界，保留区域内所有 patch 的特征，然后计算每个小人与所有小人的逐坐标中位数之间的 RMS 距离。距离最大的小人作为预测异常。

这个 baseline 没有使用异常位置或无异常图像作为输入，但使用了已知的区域边界，因此仍然是一个受控分析方法。它测试异常是否能由简单特征距离读出。失败可能来自位置编码、边界上下文、特征范数和相似度度量，也可能来自细节信号较弱，不能单独诊断为信息丢失。

正式实验应比较欧氏距离与 cosine distance，加入全正常图的误报分析，并系统改变目标位置。还应在独立的正常图像集合上估计位置差异，再检查位置校正是否改善读取。位置校正所用图像不得来自测试图像的配对答案，所有校正步骤均应单独报告。

### 3.3 merger 的公平比较

比较以下四个阶段：第 27 层输出中每组 2×2 特征的完整拼接、LayerNorm 后的完整拼接、MLP 第一层与 GELU 的输出、最终 visual embedding。对同一幅图像，各阶段使用相同的空间区域与检索任务。禁止先把 merger 前的四个特征平均成一个特征再作为唯一 baseline，因为平均操作本身可能消除待研究的差异。

本次代码已保存第 27 层、LayerNorm、两者的完整拼接与最终 visual embedding 的分析结果；MLP 内部两处输出尚未纳入初步实验。

## 4. 从响应推进到具体细节恢复

### 4.1 独立参考样本上的 nearest-neighbor retrieval

建立包含明确细节标签的参考图像库，例如完整手臂、左臂缺口、右臂缺口，以及不同长度的缺口。每个局部图案放入多种完整场景后执行 visual encoder，保存对应区域的特征与标签。测试图像使用不同位置、不同背景或未参与构建参考库的图案实例。

检索时仅使用测试图像的特征，返回特征最接近的参考图案及其细节标签。将参考图案按测试区域位置排列，可以得到直观的图像展示；同时输出缺口类型和尺度的 confusion matrix、top-1 accuracy 与定位误差。图像展示表达检索结果，其视觉细节来自参考库，不能称为原图的无条件重建。

参考库不得包含测试图像本身、其逐像素副本或仅为测试图像枚举的同场景候选。否则极小但不可泛化的特征差异也可能导致正确匹配。参考图案应在完整场景中编码，避免把独立 crop 的特征与整图中的上下文特征直接混合比较。所有读取参数与候选类别应预先固定。

该方法不更新模型权重，可以给出细节可读出的实证下界。若 nearest-neighbor retrieval 失败，可以用少量独立数据训练线性 probe，判断失败是否来自距离几何。线性 probe 仍需严格的图像级划分，不能把同一图像的 patches 随机分到训练集和测试集。

### 4.2 冻结网络的 feature inversion

Feature inversion 可直接对像素优化，无需额外训练生成模型。经典工作已经用这一方法分析深层视觉表征。[5] 设 X 为待优化的 RGB 图像，与 I 具有相同形状；Z 是选定阶段的冻结特征函数；目标特征仅在开始时从 I 提取并固定。优化问题为：

$$
X^*=\underset{X\in[0,1]^{H\times W\times3}}{\arg\min}
\frac{\|Z(X)-Z(I)\|_F^2}{\|Z(I)\|_F^2+\epsilon}
+\lambda\operatorname{TV}(X).
$$

其中 Frobenius norm 对全部位置与通道计算平方和；epsilon 是避免分母为零的小常数；TV 是相邻像素差异的总变分正则；lambda 控制图像平滑程度。特征匹配项要求重建图像产生相似表征，TV 减少高频优化噪声。TV 也可能抹去真实细线，因此必须报告不使用 TV 和较弱 TV 的对照。

实现时固定 processor 的目标分辨率，用可微的归一化和 patchify 代替 PIL 路径，并验证其输出与官方 processor 一致。图像从随机像素开始，分别匹配浅层、中层、深层以及 visual embedding；深层独立重建不能加入浅层特征损失，否则重建出的细节可能来自浅层监督。不得从原图或其模糊版本初始化后，将保留下来的细节解释为特征恢复结果。

图像比较应以目标区域的缺口存在性、方向、长度和连通性为主，辅以像素误差。整图 PSNR 对极少像素的错误不敏感。每个目标至少采用多个随机初始化，公开特征匹配误差与重建差异：成功且稳定地恢复细节提供正面证据；优化失败不足以证明信息消失。

进一步可从候选图案集合检索，或在受限的图像参数空间优化，寻找细节不同而表征接近的图像对。只有在预先定义的噪声、读取方法与距离阈值下，才能讨论这些细节的实际不可辨性。任意像素优化产生的对抗性碰撞应与自然或程序生成图像中的歧义分开报告。

Feature inversion 是少量案例的补充实验。它不需要训练 decoder，但每幅图需要多次 forward/backward，通常比第 3 节的一次特征提取更昂贵。

### 4.3 低优先级可视化方法

PCA、特征相似度图与聚类适合显示表征的主要变化方向和空间结构。DINO 的 dense descriptors 展示了无需训练的特征对应分析的可行性。[6] 但小差异可能位于低方差方向，前三个 principal components 中不可见并不意味着细节消失。跨层或跨图比较时必须明确 PCA 拟合样本、标准化方式与颜色对应，不能比较独立拟合后的 RGB 色彩含义。

Logit lens 对物体类别、部件和可语言化属性有直接价值。[7] 对同属“小人”这一类别的微小缺口，LM head 的词表和语言偏好可能隐藏差异。中间层特征输入仅针对深层分布训练的 merger 还会引入分布偏移，因此其结果适合作为语义读取的补充证据。

重建器属于后续备选。WACV 2026 的 Feature Inversion as a Lens on Vision Encoders 已使用轻量重建器比较 CLIP、SigLIP、SAM 与 InternViT，说明重建路线并不限于 diffusion。[8] 若使用生成模型，必须检验其能否恢复随机指定的微小细节，避免从训练先验补出“正常的小人”。重建清晰度与源图细节忠实度需要分别评分。

ACL 2024 的 Diffusion Lens 使用已有 text-to-image pipeline 的 diffusion model 解释其 text encoder 中间层。[9] 将 Qwen3.5 视觉表征接入图像生成模型需要另行解决接口与训练问题，不能直接假定具有同样的无需训练条件。

## 5. 对照实验与判断标准

### 5.1 尺度、位置和干扰

细节类型至少覆盖缺口长度、线宽、局部方向、接点、部件数量和微小色差。分别扫描图案尺寸、重复数量、目标位置以及相对于 16×16 patch 和 32×32 merger 网格的偏移。尺寸、重复数量和画布分辨率应独立控制，避免密度增加时同时缩小目标。

记录原图、processor 后的像素图、patch 数和 visual embedding 数。若差异已经在 resize 后消失，应归因于预处理。若 patch embedding 已经无法区分，再分析 patch projection；若浅层读取成功而深层读取下降，则进一步区分信息变化与读取方法的适配问题。Patch size 并不是硬性的最小可辨细节尺度，一个向量可以编码 patch 内部的亚 patch 结构。

本地权重的补充检查显示：静态图像复制两个时间位置后，patch projection 等效为将 16×16×3＝768 维像素向量映射到 1152 维的线性变换。将卷积权重沿时间维求和并展开，在 float32 下计算 singular values，使用约 0.0002103 的数值秩阈值，得到列秩 768；最小 singular value 约 0.001355，condition number 约 1130。这提供了该静态线性映射在数值秩判定下可逆的证据，但不保证经过低精度计算、后续非线性层或 processor resize 后仍可精确恢复像素。此检查结果保存于 `patch_projection.json`。

受控改动应与正常图、同图重复 forward、无关位置的等像素数量改动、亮度和 JPEG 等扰动对照。通过独立场景上的属性检索，检验特征响应是否对应目标属性而非一般扰动敏感性。统计时以独立基础图案或场景为单位，而非把同一图案的多个 patches 当作独立样本。

### 5.2 crop and resize 的作用分解

crop and resize 可以同时改变目标所占 patch 数、背景上下文、位置编码、token 预算和模型读取负担。现有分辨率下缺失的信息不能仅由插值补回，但原图细节如果在整图下采样时被丢弃，先裁剪再编码可能保留这些像素差异。因此，需要比较：原始整图、提高分辨率的整图、保持目标像素尺度的上下文替换、以及放大目标的局部图像。

比较时保存实际输入像素并核对目标尺度。保持目标尺度的上下文替换仍会改变位置和注意力上下文，不能称为完全隔离了单一机制。crop 只作为诊断对照；研究目标仍是通过一次整图编码保留细节。

### 5.3 与完整 VLM 行为的联系

对同一批图像分别询问局部属性和最终异常位置。局部属性识别与多目标比较的差异，可以帮助缩小失败范围，但视觉感知与推理不能简单地按模块一一划分：LLM backbone 同样参与视觉读取和属性辨别。

如果独立检索能读出 visual embedding 中的缺口，而完整 VLM 回答失败，应进一步检验读取和比较机制。可在保持问题和序列位置相同的条件下，把配对图像的 visual embedding 替换到另一条 forward 中，比较正确答案相对于错误答案的 log probability。先以全部 visual embeddings 替换核对干预链路，再分析局部区域替换；局部替换会生成混合表征，必须考虑其分布偏移以及细节在区域外的传播。随机区域和无关属性替换用于对照。

| 观察 | 可以支持的结论 | 仍需排查 |
| --- | --- | --- |
| processor 后目标差异消失 | 当前输入设置没有提供该差异 | resize 配置和源图质量 |
| 表征随细节变化 | 对该干预有敏感性 | 具体细节的可恢复性 |
| 独立参考库读取成功 | 该属性在当前读取方法下可读 | 对未见图案、位置与干扰的泛化 |
| 像素重建稳定恢复微小差异 | 当前约束下存在成功的恢复方法 | 初始化、先验和优化偏差 |
| merger 前后读取明显不同 | merger 改变了当前读取结果 | 度量、归一化与特征几何的影响 |
| 特征读取成功但回答失败 | 存在外部方法可利用而 VLM 未正确利用的信息 | LLM backbone 的读取、比较与答案映射 |

## 6. 最小研究闭环与优先级

第一阶段以第 3 节的方法在合成图案上扫描细节尺度，固定图像尺寸并保存完整特征空间顺序。随后加入独立场景的细节检索，取得具体属性可读出的证据，再针对少量代表性样本做 feature inversion。BabyVision 的真实失败样本用于检验这些机制是否能迁移到实际题图；人工编辑得到的配对图应公开修改 mask，并排查编辑边缘、压缩或纹理差异。

第二阶段在完整 Qwen3.5-9B 上比较局部属性识别与最终答案，利用 visual embedding 干预检验因果影响。只有当无需训练的读取方法与像素优化无法提供足够清晰的结果，且已明确目标属性与评价标准时，才考虑训练轻量重建器；diffusion reconstruction 的优先级更低。

潜在研究贡献是明确哪些细节在何种条件下仍可读取、它们在 visual encoder 与 merger 中如何变化、以及这种变化与 VLM 失败的关系。单独展示几张热力图或重建图不足以构成新的机制结论。FineSightBench 已系统研究小目标尺度与感知、推理表现的差异，相关评测设计应作为比较背景。[10]

## 7. 已执行的合成图初步实验

2026-09-20 使用本地 Qwen3.5-9B 的 visual encoder 与 merger，在单张 A800 上以 float32 和 SDPA 执行测试。权重通过 checkpoint 中的 `model.visual.*` 严格加载；LLM backbone 未参与。合成图包含 6×6 个小人，画布为 384×384 pixels，每个小人占据 64×64 区域。processor 输出网格严格核对为 24×24 patches，未改变空间尺寸；visual embedding 网格为 12×12。

在四个预定位置分别擦除一个小人右臂上的 1、2、4、8 pixels 长的线段，线宽为 2 pixels，对应改变 2、4、8、16 个像素，共 16 个变体。位置覆盖角落与内部。实验比较已知正常图与变体的特征响应，同时独立计算每幅变体中的小人特征异常分数。异常分数仅使用已知区域边界与该幅图像。

| 阶段 | 配对响应能量最大的区域为目标区域 | 单图异常分数 top-1 正确 |
| --- | --- | --- |
| patch embedding，未加位置编码 | 16/16 | 16/16 |
| 第 3 层输出 | 16/16 | 0/16 |
| 第 9 层输出 | 16/16 | 0/16 |
| 第 18 层输出 | 16/16 | 0/16 |
| 第 27 层输出 | 15/16 | 4/16 |
| merger 的 LayerNorm 输出 | 16/16 | 4/16 |
| visual embedding | 16/16 | 4/16 |

对 2×2 pixels 缺口这一子集，四个位置在第 27 层和 visual embedding 中的配对响应最大区域均正确；两阶段的单图异常定位各成功 1/4，但成功位置不相同。图中选定的第 4 行第 4 列案例，第 27 层约 56.0% 的特征变化能量在目标小人区域，visual embedding 约为 80.3%。这些是各自特征空间中的响应分布比例，不能解释为信息保留率。

同图两次 forward 在所有记录阶段的最大绝对差为零。模型与 processor 加载完成后，18 次编码及分析约用时 3.03 秒，不包括导入、模型加载和最终绘图。结果表明该分析可以低成本执行，但这组图案简单且重复，位置和缺口尺度也少；当前结果没有提供 BabyVision 准确率、属性恢复准确率或完整 VLM 的回答。

这组实验支持的直接结论是：极小的局部改动能够影响深层输出与 visual embedding，而最简单的单图特征距离无法稳定定位异常。位置与上下文是需要检验的解释因素；细节是否被完整编码、是否能跨场景读出，以及是否被 LLM backbone 使用仍待后续实验。

代码位于 `analysis_tools/visual_detail.py`，运行入口与输出说明见 README。产物包含完整配置、checkpoint revision、软件版本、代码 hash、原始合成图、逐位置响应、异常分数与热力图。没有新增依赖。

## 参考文献与实现来源

1. [Qwen3.5-9B 官方配置](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/config.json)。
2. [Transformers v5.17.0 Qwen3.5 实现](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py)。
3. [Qwen3-VL Technical Report](https://arxiv.org/abs/2511.21631)。
4. [Qwen3-VL-8B-Instruct 配置](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/main/config.json)。
5. [Understanding Deep Image Representations by Inverting Them, CVPR 2015](https://arxiv.org/abs/1412.0035)。
6. [Deep ViT Features as Dense Visual Descriptors](https://arxiv.org/abs/2112.05814)。
7. [Reading Images Like Texts: Sequential Image Understanding in Vision-Language Models, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/file/d8c653706a0009f187a99bd4e7ea42b8-Paper-Conference.pdf)。
8. [Feature Inversion as a Lens on Vision Encoders, WACV 2026](https://openaccess.thecvf.com/content/WACV2026/html/Allakhverdov_Feature_Inversion_as_a_Lens_on_Vision_Encoders_WACV_2026_paper.html)。
9. [Diffusion Lens: Interpreting Text Encoders in Text-to-Image Pipelines, ACL 2024](https://aclanthology.org/2024.acl-long.524/)。
10. [The Last Visible Pixel: Probing Fine-Scale Perception in Vision-Language Models](https://arxiv.org/abs/2606.07861)。
