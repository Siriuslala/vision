# 实验 3.7：ViT 位置与布局表征

本实验属于[位置信息与坐标读取](plan.md)研究。实验 3.5（[结果](results_2026-10-14_coordinate_readout.md)）在 Qwen3.5-9B 的单目标定位任务上测得：decoder 的 MRoPE position IDs 对坐标输出没有因果作用（位移权重 $w \approx 0$），ViT 的 2D RoPE 承担了大部分位置信息（只交换 RoPE ID 时 $w = 0.71$），learned absolute position embedding（下称 PE）的贡献接近 0（只交换 PE 时 $w = 0.01$），其余部分来自序列顺序。本实验把研究重点移到 ViT 内部，回答三个问题：PE 为什么几乎不起作用；位置与物体布局在 ViT 中以什么形式表示；这些结论对方向二中局部重读的位置输入有何含义。

## 1. 研究问题

1. **PE 的作用。** PE 在 patch embedding 之后加入一次，RoPE 作用于全部 27 个 ViT block 的 attention。需要确定：PE 在数值上相对于 patch embedding 有多大，经过 ViT 后在 token 中还占多少；PE 是否只在 48×48 表的原生分辨率（768 pixels）附近才有作用；去除、打乱、替换或放大 PE 后，坐标输出、ViT 各层的位置可读性和与位置无关的识别能力分别如何变化。
2. **布局的表示方式。** 一个 token 自身的位置以及多个物体之间的相对布局（“哪一个在左边、哪一个更大”）能否从该 token 本身、其他 token（例如远离物体的背景 token）或整张图的 pooled 特征中线性读出，这种可读性如何随 ViT 层数变化；ViT 输出中的位置信息是否集中于一个低维子空间，它与内容是否可以分离。
3. **局部重读的位置输入。** 若把原图的一个局部窗口裁剪后重新送入 ViT，哪一种位置输入（按原图坐标平移的 RoPE ID、取自原图对应位置的 PE 窗口、在原图画布上下文中编码）能使 decoder 输出原图坐标。

## 2. 相关工作与本实验的区别

1. **Beyond Semantics（Qi et al., 2025，arXiv 2503.17349）。** 该工作在 LLaVA-1.5-7B 上随机打乱进入 LLM 的 visual token 顺序，标准基准只下降 0.2%–2.7%；visual token 的范数比文本 token 大 1–2 个数量级，作者认为这压制了 LLM 的 RoPE，并提出对 visual embedding 做 RMS 归一化、拼接 ViT 中间层特征。该工作只研究 LLM 一侧的位置编码，没有干预视觉编码器内部的 PE 或 RoPE，也不涉及坐标输出。实验 3.5 在 Qwen3.5 的 MRoPE 上得到一致的结论（$w \approx 0$），本实验进一步追问位置信息在 ViT 内部由哪一种编码写入。
2. **From What to Which（Corradini et al., 2026，arXiv 2610.12305）。** 该工作在冻结的 Qwen2.5-VL-3B（附录另有 Qwen3-VL-2B 等）上训练一个约 55 万参数的 optimal transport probe（OTTER），把生成文本 token 在 LLM 中间层的 hidden state 映射到图像区域，并用“目标—干扰物”反事实对检验 left、yellow 等修饰词能否选中被指称的实例。主要发现是：名词 token 的区域图与修饰词 token 同样具有选择性，实例身份分布在上下文化的多个 token 状态之中；定位能力强烈依赖层数，Qwen3-VL-2B 在约一半深度处最好。该工作的读出对象是 LLM 中文本 token 的 hidden state，没有逐层分析 ViT，也没有区分物体 token、背景 token 与 pooled 特征。本实验借用其三项方法要素：以“目标—干扰物”成对物体构造布局标签；设置不含信息的对照（打乱标签、无位置编码的 ViT）；逐层扫描。与之不同，本实验把读出对象移到 ViT 的 27 个 block 与 merger，并以线性 probe 分别读取物体 token、背景 token、单个背景 token 与 pooled 特征，回答“布局信息在哪些 token 中”。
3. **Qwen3-VL 为什么同时使用 PE 与 2D RoPE。** Qwen3-VL 技术报告（arXiv 2511.21631，第 2 节 Vision Encoder）说明其视觉编码器采用 SigLIP-2 架构，从官方 SigLIP2-SO400M checkpoint 初始化后以动态分辨率继续训练，并“按 CoMP 的方法使用 2D-RoPE 并根据输入尺寸插值 absolute position embedding”。CoMP（arXiv 2503.18931，第 3.1 节）给出的理由是：先插值原有的 absolute PE、再加入 RoPE-2D，可以从固定分辨率的预训练 ViT 平滑过渡到任意分辨率；在其消融中（Table 6，SigLIP-400M + Qwen2-0.5B，768 pixels），去掉原有 PE 只用 RoPE-2D 时 ChartQA / DocVQA 为 8.2 / 11.9，同时使用二者为 32.2 / 33.2。Qwen2-VL 与 Qwen2.5-VL 的 ViT 从 DFN 初始化，并用 RoPE-2D 替换了原有的固定 PE。因此 PE 的存在主要是为了在继续训练时保持预训练权重的连续性，并非为承载空间信息而设计。公开的 SigLIP2-so400m-patch16-naflex 的 PE 表为 16×16，Qwen 的表为 48×48（对应 768 pixels，与 CoMP 最后一个训练阶段的分辨率一致），插值方式也不同（align_corners=True），推测 Qwen 把表放大到 48×48 后继续训练；官方没有说明 48 的来源。Qwen3.5-9B 的 ViT 结构与 Qwen3-VL-8B 相同（hidden size 1152、27 层、16 个 head），区别在于不使用 DeepStack。

## 3. 符号与位置输入

| 符号 | 形状 | 含义 |
| --- | --- | --- |
| $h_p, w_p$ | 标量 | 图像的 ViT patch 网格行数与列数，patch 边长 16 pixels；边长 $S$ 的方形画布有 $h_p = w_p = S/16$ |
| $N_p$ | 标量 | patch 数 $h_p w_p$ |
| $x_i$ | $\mathbb{R}^{1152}$ | 第 $i$ 个 patch 的 patch embedding（卷积输出，下称内容向量） |
| $T$ | $\mathbb{R}^{2304 \times 1152}$ | learned PE 表，按 $48 \times 48$ 网格排列，第 $(a, b)$ 项记为 $T_{a,b}$ |
| $e_i$ | $\mathbb{R}^{1152}$ | 第 $i$ 个 patch 的 PE |
| $(r_i, c_i)$ | $\mathbb{Z}^{2}$ | 第 $i$ 个 patch 的行列号，作为 ViT 2D RoPE 的 position ID |
| $z^{(\ell)}_i$ | $\mathbb{R}^{1152}$ | 第 $\ell$ 个 ViT block 输出的第 $i$ 个 token，$\ell = 1, \ldots, 27$ |
| $m_j$ | $\mathbb{R}^{4096}$ | merger 输出的第 $j$ 个 visual token（$2 \times 2$ 个 patch 合并为一个） |

ViT 的输入为

$$
z^{(0)}_i = x_i + e_i, \qquad e_i = \sum_{k=1}^{4} \alpha_{ik}\, T_{a_{ik}, b_{ik}}.
$$

其中 $(a_{ik}, b_{ik})$ 与 $\alpha_{ik}$ 是双线性插值（align_corners=True）的 4 个邻点与权重：patch $(r_i, c_i)$ 被映射到表上的连续坐标 $\big(47 r_i/(h_p-1),\ 47 c_i/(w_p-1)\big)$。因此任何尺寸的图像都使用整张 $48 \times 48$ 表：$S < 768$ 时表被下采样，$S > 768$ 时表被上采样，PE 编码的是 patch 在图像中的**相对**位置。$S = 768$ 时插值退化为逐项取表。

RoPE 在每个 block 的 attention 中按 $(r_i, c_i)$ 旋转 query 与 key，使第 $i$ 个 query 与第 $j$ 个 key 的内积只依赖二者的内容与相对位移 $(r_j - r_i, c_j - c_i)$。因此对全部 patch 的 RoPE ID 加同一常数不改变 ViT 输出（实验 3.5 已验证）。PE 是唯一以绝对形式进入 ViT 的位置信号。

这一结构给出一个可检验的推论：若一片区域中的 token 内容完全相同（例如白色背景）且没有 PE，则 attention 的 value 在这片区域内相同，无论 RoPE 如何分配 attention 权重，输出都相同，RoPE 无法在均匀区域内产生位置差异。PE 为每个 patch 提供互不相同的内容，使 RoPE 的相对位置计算有了作用对象。第 4.1 节的“均值 PE”“打乱 PE”与“噪声背景”条件专门检验这一推论。

## 4. 实验设计

### 4.1 Q1：PE 的作用

**位置输入编辑。** 对每张图像独立计算并替换 ViT 的两种位置输入（Qwen3.5 与 Qwen3-VL 的 ViT 前向均支持从外部传入 RoPE ID 与插值索引，DeepStack 特征在同一前向中产生，因此 Qwen3-VL 的编辑同时作用于 merger 与 DeepStack 输出）：

| 条件 | PE | RoPE ID | 检验的问题 |
| --- | --- | --- | --- |
| base | 默认插值 | 默认 | 基线 |
| pe_zero | $e_i = 0$ | 默认 | 去除 PE 的总效应 |
| pe_mean | $e_i = \bar e$（本图全部 PE 的均值） | 默认 | PE 的平均偏置与 patch 间差异各自的作用 |
| pe_shuffle | $e_i = e_{\pi(i)}$，$\pi$ 为随机置换 | 默认 | 保留 PE 的分布与 patch 间差异、去除其空间排列 |
| pe_native | $e_i = T_{a_0 + r_i,\, b_0 + c_i}$，居中取 $h_p \times w_p$ 的 1:1 子表 | 默认 | 插值（缩放）是否削弱 PE；仅 $S \le 768$ |
| pe_x2, pe_x4 | $2 e_i$、$4 e_i$ | 默认 | PE 是否因数值过小而被 RoPE 掩盖 |
| pe_x2.76（Qwen3.5）、pe_x0.36（Qwen3-VL） | $2.76\, e_i$、$0.36\, e_i$ | 默认 | 把一个模型的 PE 缩放到另一个模型的尺度（两表的平均范数比为 2.76） |
| rope_zero | 默认 | 全部为 0 | 去除 RoPE；ViT 只剩 PE 一个位置来源 |
| rope_zero_pe_x4 | $4 e_i$ | 全部为 0 | 只有 PE 时放大 PE 能否恢复位置 |
| rope_zero_pe_mean | $\bar e$ | 全部为 0 | ViT 不含任何位置差异 |
| rope_zero_pe_zero | 0 | 全部为 0 | ViT 对 patch 置换等变，坐标只可能来自序列顺序 |

**测量。**

1. **数值占比（norms 阶段）。** 统计 $\lVert e_i \rVert / \lVert x_i \rVert$ 随画布尺寸的变化；对每个 ViT block 计算编辑引起的相对变化

$$
\delta^{(\ell)} = \frac{1}{N_p}\sum_{i} \frac{\lVert z^{(\ell)}_i - \tilde z^{(\ell)}_i \rVert}{\lVert z^{(\ell)}_i \rVert},
$$

其中 $\tilde z^{(\ell)}_i$ 为编辑后的 block 输出。$\delta^{(\ell)}$ 衡量该位置输入对第 $\ell$ 层表示的总体影响，不区分位置信息与其他信息。另比较 Qwen3.5 与 Qwen3-VL 的 PE 表与其他 ViT 权重，确定两个模型 ViT 的继承关系。

2. **坐标输出（coord 阶段）。** 沿用实验 3.5 的单目标任务与问题：白色方形画布上一个红色实心圆，问题为 “There is one red circle in the image. Give the center of its bounding box. Return only [{"point_2d": [x, y]}]. All coordinates use 0-1000 normalization; x increases rightward and y increases downward. Output only the JSON, with no explanation.”。画布边长 $S \in \{256, 384, 512, 768, 1024, 1536\}$，即 16–96 个 patch（PE 表为 48），圆的直径为 $S/8$，圆心位于 $6 \times 6$ 等距位置并加 $\pm S/32$ 抖动，每个尺寸 36 张。另在 $S \in \{256, 512, 1024\}$ 上把背景换成逐像素灰度噪声（取值 170–255），检验第 3 节的“均匀区域”推论。指标为命中率（输出点落在圆内，Wilson 95% 区间）、中心误差（0–1000 坐标，bootstrap 95% 区间）、输出与真值的相关系数及回归斜率。

3. **位置来源互换（swap 阶段，仅 Qwen3.5）。** 沿用 3.5 的区域互换：把目标所在 token 区域 $\mathcal{T}$ 与平移 $k = \max(2, H/4)$ 个 merged token 后的区域 $\mathcal{T}'$ 之间的 PE、RoPE ID 或二者交换，按 3.5 的位移权重

$$
w = \frac{(\hat y - \hat y_0)^{\mathsf T} d}{\lVert d \rVert^{2}}
$$

计算输出跟随程度，其中 $\hat y_0, \hat y \in \mathbb{R}^{2}$ 为干预前后的输出坐标，$d \in \mathbb{R}^{2}$ 为互换位移（均为 0–1000 坐标）。在 $S \in \{256, 512, 768, 1024\}$ 上分别测量，并在“去除 RoPE”“PE 放大 4 倍”及二者同时成立的条件下重复 PE 互换。若 PE 被 RoPE 掩盖，则去除 RoPE 后 PE 互换的 $w$ 应明显上升。

4. **ViT 各层的位置可读性（probe 阶段）。** 使用实验 3.4 的 RefCOCO 图像（取中心方形区域并缩放到 512×512），训练 320 张、验证 80 张、测试 100 张。在内容向量、ViT 输入、27 个 block 输出与 merger 输出上，以全维 ridge 回归 token 自身的归一化坐标（沿用 3.4 的矩累积与闭式解，$\lambda$ 由验证集选择），报告测试 MAE。对 base、pe_zero、pe_mean、pe_shuffle、rope_zero 与二者同时为 0 的条件分别拟合。另计算位置方差占比

$$
\rho^{(\ell)} = \frac{\frac{1}{N_p}\sum_{i} \lVert \mu^{(\ell)}_i - \mu^{(\ell)} \rVert^{2}}{\operatorname{tr} \Sigma^{(\ell)}},
$$

其中 $\mu^{(\ell)}_i \in \mathbb{R}^{1152}$ 为第 $i$ 个槽位在训练图像上的均值，$\mu^{(\ell)}$ 为全部 token 的均值，$\Sigma^{(\ell)}$ 为全部 token 的协方差。$\rho^{(\ell)}$ 是 token 方差中“只由槽位决定”的部分，分子已扣除槽位均值的有限样本偏差。

5. **识别对照（t2 阶段）。** 采用实验 1.3 的 T2 两实例异同判断：两个并排的图形，一半完全相同，一半其中一个有 8 pixels（模板单位）缺口或 45° 旋转，问题为 “Are the two figures exactly identical? Answer with only Yes or No.”。画布边长 256–1536，图形边长 $S/4$，每个尺寸 48 对。指标为 Yes 与 No 首 token logit 差的 AUROC。若一种编辑破坏坐标而不破坏 T2，其效应可以归于位置；若二者同时被破坏，属于一般性损伤。

6. **自然图像 grounding（refcoco 阶段）。** 使用实验 4.2 的 RefCOCO val 751 条表达式与相同的 bbox 问题和评分（IoU≥0.5 的比例，以及成对的平均 IoU 差），比较 base 与 PE 缩放条件（Qwen3.5：pe_x2、pe_x2.76、pe_x4；Qwen3-VL：pe_x0.36、pe_x2），检验合成图像上 PE 缩放的效应能否推广到内容丰富的自然图像。

Qwen3-VL 常把同一个点重复输出两次，严格评分记为格式错误；另计诊断解析（全部点相同时取该点），严格结果同时保存。

### 4.2 Q2：布局的表示方式

**刺激。** 512×512 白色画布上一个红色实心圆与一个蓝色实心正方形，边长 40–160 pixels，位置随机且互不重叠，二者横、纵坐标之差均大于 24 pixels，尺寸差至少 12 pixels，共 1600 张，按 60% / 15% / 25% 分为训练、验证、测试。标签为红圆的横纵坐标、蓝方的横纵坐标、“红圆在蓝方左侧”“红圆在蓝方上方”“红圆更大”三个二值关系。

**读出。** 在内容向量、ViT 输入、16 个 block（1–4、6、8、…、26、27）与 merger 上提取五种特征：红圆所覆盖 patch 的均值（物体 token）、蓝方所覆盖 patch 的均值、与两个物体的外接框相距至少一个 patch 的背景 patch 均值、随机一个背景 patch、全部 patch 的均值（pooled）。以 ridge 回归读取坐标（报告测试 $R^2$），以 ridge 分类读取二值关系（报告准确率与 Wilson 区间）。条件为 base、pe_zero 与 rope_zero（Qwen3.5），以及 Qwen3-VL 的 base。

背景特征的标签泄漏需要单独控制：背景 patch 的位置分布与物体位置负相关，因此背景均值在 ViT 输入（内容 + PE）处已含有物体位置的线性线索。该层的读出结果作为泄漏基线，后续层相对于它的提高才归因于 attention 带来的信息混合。

**位置子空间与线性擦除（leace 阶段）。** 在 merger 输出上（512×512 输入，$16 \times 16 = 256$ 个 token），以 320 张自然图像与 320 张布局图像拟合 LEACE（Belrose et al., 2023）擦除器

$$
r(m) = m - P\,(m - \mu), \qquad P = W^{+} U U^{\mathsf T} W, \qquad W = \Sigma_{mm}^{-1/2},
$$

其中 $\Sigma_{mm} \in \mathbb{R}^{4096 \times 4096}$ 为 merger token 的协方差，$\mu$ 为其均值，$U$ 为 $W \Sigma_{mz}$ 列空间的标准正交基，$\Sigma_{mz}$ 为 token 与概念 $z$ 的交叉协方差。擦除后没有任何线性函数与 $z$ 相关，且在白化度量下改动最小。概念 $z$ 取两种：槽位的 one-hot 指示（256 维，擦除全部依赖槽位的线性信息，秩不超过 255）与槽位的归一化坐标 $(x, y)$（秩 2）。对照为同秩的随机子空间擦除，以及直接减去槽位均值（不白化）。擦除施加在 decoder 读取之前，测量单目标坐标输出（直径 32、64、128 各 36 张）、T2 识别（48 对）与擦除后重新拟合的线性位置 probe。同时报告槽位均值矩阵的主成分谱与前几个主成分的空间形状，用于判断位置信号的维数。

### 4.3 Q3：局部重读的位置输入（小规模预实验）

512×512 白色画布上一个直径 48 pixels 的红圆，裁取包含圆的 256×256 窗口（起点为 32 pixels 的整数倍），48 个样本。decoder 只看到窗口的 $8 \times 8$ 个 visual token 与单目标问题，比较以下窗口编码方式：

| 条件 | ViT 输入 |
| --- | --- |
| crop_local | 窗口作为独立图像，默认位置输入 |
| rope_offset | RoPE ID 加上窗口在原图中的 patch 偏移 |
| pe_window | PE 取原图 512 画布插值后窗口对应位置的值 |
| rope_offset_pe_window | 二者同时 |
| pe_window_x4 及其与 rope_offset 的组合 | 放大 4 倍的原图 PE 窗口 |
| canvas_context | 在原图上编码，再从 merger 输出中取出窗口内的 64 个 token |
| full_image | 参照：decoder 看到整张原图 |

记窗口坐标系下的真值为 $y_{\mathrm{loc}}$，原图坐标系下为 $y_{\mathrm{full}}$（均为 0–1000），输出为 $\hat y$，以 $w = (\hat y - y_{\mathrm{loc}})^{\mathsf T} d / \lVert d \rVert^{2}$、$d = y_{\mathrm{full}} - y_{\mathrm{loc}}$ 衡量输出偏向原图坐标的程度。由第 3 节，rope_offset 在数学上不改变 ViT 输出，作为实现检查。

## 5. 预期结果与解释

1. 若 PE 去除后坐标输出大幅下降、而打乱 PE 影响较小，说明 PE 的作用主要是为 RoPE 提供互不相同的 patch 内容，而非提供位置标签；若这种下降在噪声背景上消失，则支持第 3 节的均匀区域推论。
2. 若去除 RoPE 后 PE 互换的 $w$ 明显上升，说明 PE 携带可用的位置信息但在完整模型中被 RoPE 冗余掩盖；若仍接近 0，说明 decoder 无法利用 PE 形式的位置。
3. 若背景 token 或 pooled 特征在中后层能读出物体位置与关系，且明显高于 ViT 输入处的泄漏基线，说明布局信息在 ViT 中被分布式地写入非物体 token。
4. 若 one-hot 擦除显著破坏坐标输出而 T2 不变，同秩随机擦除两者都不变，说明位置信息位于一个可与内容分离的线性子空间。

## 6. 运行与产物

1. 入口为 `scripts/analysis/position/vit_layout.sh`，模型、阶段与全部参数在脚本顶部设置；程序为 `analysis_tools/position/vit_layout.py`（位置输入编辑、特征提取、刺激、LEACE）、`vit_layout_run.py`（各阶段）、`vit_layout_analysis.py`（指标）与 `vit_layout_figures.py`（图）。
2. 产物保存在 WORK_DIR 的 `analysis/position/2026-10-15_vit_layout/`，每个阶段写入 JSONL 或张量文件以及记录模型版本、代码摘要与参数的元数据；文档用图保存在本目录 `figures/vit_layout/`。
3. 全部实验使用 bfloat16、SDPA、greedy decoding 与 thinking off。批量生成使用 left padding 的 `generate_batch`（batch size 8），并随机抽取 20 个样本与逐条 `generate` 比较答案一致性。

## 7. 第二轮：合成图像与自然图像中的 ViT 位置

### 7.1 动机

第一轮在合成图像上建立了三项结论：Qwen3.5 的 ViT 位置主要由 RoPE 计算，PE 在其中只负责打破均匀区域的对称性；Qwen3-VL 的 PE 仍是可用的绝对位置标签；merger 输出中的位置是可与内容分离的线性子空间。实验 3.5 在 RefCOCO 上得到与合成图像不同的结果：只交换目标区域的 ViT RoPE ID 时输出只跟随 $w = 0.16$（单目标画布为 0.71），把目标的 visual embedding 移入另一个序列槽位（内容交换）时跟随 0.78。第二轮回答四个问题：

1. 自然图像上为什么由序列槽位决定坐标，而合成图像上由 ViT 写入的位置决定坐标。
2. “Qwen3-VL 依赖 PE、Qwen3.5 依赖 RoPE”这一表述在自然图像上是否成立。
3. merger 输出中位置子空间的维数、跨图像与跨图像类型的共享程度、与 RoPE 频率的关系，以及 decoder 中的指向型 head（实验 3.5 的 L20 H11）是否从中读取位置。
4. 一个实例只占一个 token 时，均匀背景的对称性破缺是否更重要（低优先级）。

一个待检验的假设是：自然图像的表示带有与语言对齐的先验（ViT 来自 SigLIP 式的对比学习）和场景布局先验（天空在上、地面在下），而由重复实例构成的合成图像需要原始的视觉原子感知，且在训练数据中少见。第二轮的设计把这一假设拆成可以分别检验的成分：内容本身是否携带绝对位置（场景先验），位置是否依赖目标与上下文之间的关系（上下文），以及 decoder 从哪一组 token 读取 ViT 写入的位置。

### 7.2 位置来源的分解

第一轮第 3 节说明，ViT 的 attention 只通过 RoPE 看到相对位移，PE 是唯一的绝对输入。由此可以得到一个精确的等价关系：设 $\mathcal{T}$ 为与目标 bbox 重叠的 merged token 集合，$\mathcal{T}'$ 为其平移副本（与 3.5 相同的规则：沿横向平移目标宽度加一个 token，不可行时沿纵向），在 $\mathcal{T}$ 与 $\mathcal{T}'$ 的全部 patch 之间同时交换 PE 与 RoPE ID（记为 vit_swap），ViT 的输出等于“先把两块像素互换、正常编码、再把输出 token 换回原来的槽位”。因此 vit_swap 让目标内容在 ViT 中“位于” $\mathcal{T}'$，而 decoder 仍在槽位 $\mathcal{T}$ 读到它；内容交换（content_swap，交换两组 merged token 的 embedding，序列槽位与 MRoPE ID 不动）正好是余下的一半。两者依次作用等于真实移动目标，故有近似关系

$$
w_{\mathrm{vit}} + w_{\mathrm{content}} \approx w_{\mathrm{move}},
$$

其中 $w_{\mathrm{vit}}$、$w_{\mathrm{content}}$、$w_{\mathrm{move}}$ 分别为三种干预下按设计第 4.1 节定义的位移权重 $w$（输出位移在互换位移 $d \in \mathbb{R}^2$ 上的投影除以 $\lVert d \rVert^2$，0–1000 坐标）。$w_{\mathrm{vit}}$ 衡量 decoder 依据 ViT 写入的位置定位的程度，$w_{\mathrm{content}}$ 衡量依据序列槽位（及与之一致的 MRoPE ID）定位的程度。

vit_swap 在 ViT 内同时改变两类 token：$\mathcal{T} \cup \mathcal{T}'$ 内的 token（目标自身及被换走的区域）与其余 token（上下文，它们通过 attention 得知目标位于何处，第一轮第 3.7 节）。为区分 decoder 从哪一类 token 读取 ViT 位置，增加两个 token 组拼接条件：

| 条件 | $\mathcal{T} \cup \mathcal{T}'$ 内的 token | 其余 token |
| --- | --- | --- |
| vit_swap_target | 取自 vit_swap 的前向 | 取自基线前向 |
| vit_swap_context | 取自基线前向 | 取自 vit_swap 的前向 |

若 decoder 主要从目标 token 自身读取位置，则 $w$ 集中在 vit_swap_target；若主要读取上下文中分布式的布局编码，则集中在 vit_swap_context。另保留只交换 RoPE ID（rope_swap，与 3.5 相同）与只交换 PE（pe_swap）。所有拼接与交换都在 vision tower 的输出上进行，对 Qwen3-VL 同时作用于 merger 输出与三层 DeepStack 特征，因此两种模型使用同一实现。

### 7.3 刺激阶梯（Q1）

为把“合成”与“自然”之间的差别拆成单个因素，每个刺激都由实验 3.4 测试划分中的同一张 RefCOCO 图像派生（长边 512 pixels，画布尺寸与 token 网格在阶梯内相同，因而样本在刺激之间成对）：

| 刺激 | 背景 | 目标 | 问题 | 检验的因素 |
| --- | --- | --- | --- | --- |
| disc_white | 白色 | 红色实心圆，圆心为目标 bbox 中心，直径为 bbox 短边的 0.8 倍（限制在 32–96 pixels） | 3.5 的 point 问题 | 第一轮与 3.5 的参照 |
| disc_checker | 灰色棋盘格，周期 16 pixels，每个 patch 的像素完全相同 | 同上 | 同上 | 像素层面不均匀、patch 层面均匀 |
| disc_noise | 逐像素灰度噪声（170–255） | 同上 | 同上 | patch 之间有差异但无结构 |
| disc_shuffled | 本图的 32×32 块随机置换 | 同上 | 同上 | 自然纹理统计，无场景布局 |
| disc_scene | 本图 | 同上 | 同上 | 完整自然上下文，目标为合成物体 |
| obj_white | 白色 | 本图的被指物体（RefCOCO 分割多边形内的像素），位于原位置 | RefCOCO 表达式 + point 格式 | 自然物体、无上下文 |
| obj_shuffled | 本图的块置换 | 同上 | 同上 | 自然物体、无场景布局的上下文 |
| natural | 本图 | 原图 | 同上 | 3.5 的参照 |
| natural_vflip | 本图上下翻转 | 原图上下翻转 | 同上（表达式含 top、below、front 等方位或远近词的样本剔除） | 物体处于非典型位置（天空在下） |

每个刺激运行 base、rope_swap、pe_swap、vit_swap、vit_swap_target、vit_swap_context、content_swap，可移动的刺激（前 7 个）另运行真实移动（目标按 $d$ 平移后重新绘制）。natural 另按 $\mathcal{T}'$ 的纹理选两个平移副本：沿同一轴、平移量不小于目标宽度加一的候选中，$\mathcal{T}'$ 内 token 像素灰度标准差均值最小（homog）与最大（textured）的一个，各运行 rope_swap、vit_swap、content_swap，检验“交换到均匀区域”与“交换到复杂内容”是否不同。实验 3.5 已有的 237 个 RefCOCO 互换也按同一纹理指标重新分析（不需要 GPU）。

**ViT 层面的读出。** 为区分“ViT 没有把目标的位置改到 $\mathcal{T}'$”与“ViT 改了但 decoder 没有使用”，在每个条件下记录 merger 输出在 $\mathcal{T}$、$\mathcal{T}'$ 与其余 token 上的均值，并用两个线性读出（按样本 5 折交叉验证）：

1. token 位置 probe：以基线图像的 merger token 回归其自身的归一化坐标，应用于 $\mathcal{T}$ 的均值 token，得到“目标 token 被编码在何处”，其位移权重记为 $w_{\mathrm{probe}}$；
2. 上下文 probe：以基线图像其余 token 的均值回归目标中心，应用于各条件，得到“上下文认为目标在何处”，记为 $w_{\mathrm{ctx}}$。

**内容是否携带绝对位置。** 在 ViT 中同时去除 PE 与 RoPE（第一轮的 rope_zero_pe_zero）后，ViT 对 patch 置换等变，任何可读的绝对位置只能来自内容（场景先验）。在 512×512（等比例拉伸以保留全部内容）的 RefCOCO 图像上，以 320 张训练图像拟合线性 probe，分别报告横、纵坐标的误差与相关，并测试上下翻转图像：若存在“天空在上”的先验，翻转图像的纵坐标预测应与真值负相关。

### 7.4 PE 与 RoPE 依赖（Q2）

1. **整图编辑。** 在 3.5 的 300 张 RefCOCO 测试图像与 obj_white 上运行 bbox 问题，条件为 base、pe_zero、pe_mean、pe_shuffle、PE 缩放到另一模型的尺度（Qwen3.5 ×2.76，Qwen3-VL ×0.36）、rope_zero、rope_shuffle（RoPE ID 在 patch 之间随机置换）与 rope_zero_pe_zero，两种模型相同。指标为 IoU≥0.5 的比例（Wilson 区间）、成对的平均 IoU 差（bootstrap 区间）与中心误差。
2. **区域互换。** 刺激阶梯在两种模型上运行，pe_swap 与 rope_swap 给出两种来源在每种刺激上的相对权重。
3. **布局分布。** 第一轮只在 Qwen3.5 上比较了 pe_zero 与 rope_zero 下的布局读出，第二轮对 Qwen3-VL 补齐这两个条件（同一批 1600 张两物体图像）。
4. **其他 checkpoint 的 PE 表。** 直接读取本地缓存的 Qwen3-VL-4B/8B 与 Qwen3.5-4B/9B/27B 的 PE 表，比较方向与范数，检验 PE 缩小是否为 Qwen3.5 系列的共同现象。
5. **文献与社区。** 检索 ViT 中 absolute PE 与 RoPE 并用的设计与消融（RoPE-ViT、CoMP、SigLIP 2 NaFlex、EVA-02、Perception Encoder、DINOv3 等）以及相关的社区讨论，在结果文档中定位本实验的结论。

### 7.5 merger 输出的位置子空间（Q3）

**数据。** 均为 512×512 输入（$16 \times 16 = 256$ 个 merged token）：RefCOCO 训练图像 320 张（nat_train）、测试图像 100 张（nat_test）及其上下翻转（nat_vflip）与块置换（nat_shuffled）版本、第一轮的两物体布局图像 320 张（syn_layout）与 108 张单目标圆（syn_disc）。

**符号。** 记某一图像集合中第 $n$ 张图像第 $j$ 个槽位的 merger 输出为 $m_{n,j} \in \mathbb{R}^{4096}$，槽位均值 $\bar m_j = \frac{1}{N}\sum_n m_{n,j}$，全体均值为 $\bar m$。槽位均值矩阵 $M \in \mathbb{R}^{256 \times 4096}$ 的第 $j$ 行为 $\bar m_j - \bar m$。$M$ 的右奇异向量张成“位置子空间”，其前 $k$ 个方向记为 $B_k \in \mathbb{R}^{k \times 4096}$（行正交）。

1. **维数谱。** $M$ 的奇异值平方占比与参与比 $(\sum_i \sigma_i^2)^2 / \sum_i \sigma_i^4$。第一轮发现前两个主成分由角落的高范数 token 决定，因此另在去除偏离范数大于中位数 3 倍的槽位后重算。
2. **共享程度。** 两个集合的子空间 $B_k$ 与 $B'_k$ 之间的重叠定义为主角余弦平方的均值

$$
O_k = \frac{1}{k}\big\lVert B_k B_k'^{\mathsf T} \big\rVert_F^2 \in [0, 1],
$$

1 表示两个子空间相同，随机子空间的期望为 $k / 4096$。比较 nat_train 的两半（可靠性上限）、nat_train 与 nat_test、syn_layout、syn_disc、nat_vflip、nat_shuffled。另以一个集合上拟合的线性位置 probe 测试其他集合（跨集合迁移）。
3. **与 RoPE 频率的关系。** Qwen ViT 的轴向 2D RoPE 在每个轴上有 18 个频率 $\omega_i = 10000^{-2i/36}$（弧度/patch，$i = 0, \ldots, 17$），换算为每个 merged token $\omega_i / \pi$ 个周期。把 $M$ 的前 10 个主成分得分排回 $16 \times 16$ 网格，计算二维功率谱随空间频率的分布，并与 RoPE 频率集合对照：若位置子空间主要由网格可分辨的高频 RoPE 分量构成，功率应集中在 $\omega_0/\pi \approx 0.32$ 与 $\omega_1/\pi \approx 0.11$ 周期/token 附近；若集中在最低频，说明 merger 输出编码的是类似 PE 表的平滑绝对坐标。
4. **L20 H11 是否读取位置成分。** 在 512×512 的 RefCOCO 测试图像（point 问题，300 张）与 108 张单目标圆上，以模型自己的回答做 teacher forcing，在生成横坐标首位数字的 query 位置重算 decoder 第 20 层 full attention 的 $q$（各 head）以及图像 token 的 $k$、$v$（4 个 KV head，已施加 MRoPE 旋转；图像位于序列开头，同一槽位在各图像中的旋转相同）。对每个槽位用留一法均值把 key 分解为位置成分与内容成分

$$
k_{n,j} = \underbrace{\tilde k_{j}^{(-n)}}_{\text{位置成分}} + \underbrace{\big(k_{n,j} - \tilde k_{j}^{(-n)}\big)}_{\text{内容成分}},
$$

其中 $\tilde k_j^{(-n)}$ 为除第 $n$ 张图像外同一槽位 key 的均值。value 作同样分解。由此计算：（a）key 与 value 方差中位置成分的占比；（b）attention logit $s_{n,j} = q_n^{\mathsf T} k_{n,j} / \sqrt{d_h}$ 在图像 token 之间的方差中两种成分各自的占比，以及只用一种成分做 softmax 时目标区域的富集度（目标 token 上的概率质量除以其面积占比）；（c）把 head 的输出 $o_n = g_n \odot \sum_j a_{n,j} v_{n,j}$（$a$ 为 attention 权重，$g$ 为门控）分成位置成分与内容成分两部分，分别以 5 折交叉验证的 ridge 回归读取目标中心，比较 $R^2$。最后在 merger 输出上施加 one-hot LEACE 擦除后重复以上测量，检验 L20 H11 读到的位置是否来自 merger 的位置子空间。
5. **自然图像上的擦除。** 以 nat_train、syn_layout 或二者合并拟合槽位 one-hot LEACE 擦除器，另有 $(x, y)$ 二维擦除器与同秩随机擦除器，测量 RefCOCO（300 张，point）与 108 张单目标圆的命中、中心误差与回归斜率。以自然图像拟合的擦除器能否破坏合成图像的定位（及反之）同时检验子空间的跨类型共享。

### 7.6 一个实例占一个 token（Q4，低优先级）

512×512 白色与噪声背景上的单个红圆，直径 16、24、32、64 pixels（16 与 24 pixels 的圆落在一个 merged token 内），6×6 位置加抖动，条件为 base、pe_zero、pe_shuffle、pe×2.76（Qwen3.5）。比较 pe_zero 造成的损失是否随实例变小而增大。

### 7.7 运行

入口为 `scripts/analysis/position/vit_layout_r2.sh`，程序为 `analysis_tools/position/vit_layout_r2.py`（刺激、区域、逐行的 ViT 位置输入与输出拼接）、`vit_layout_r2_run.py`、`vit_layout_r2_analysis.py` 与 `vit_layout_r2_figures.py`，测试为 `tests/analysis/position/test_vit_layout_r2.py`。产物保存在 WORK_DIR 的 `analysis/position/2026-10-16_vit_layout_r2/`，文档用图保存在 `figures/vit_layout_r2/`。生成使用 left padding 的 `generate_batch`，同一样本的全部条件位于同一个 batch（拼接条件只引用同一 batch 中的行）；实现检查包括：钩子实现的 RoPE 互换与 3.5 引擎的 merger 输出逐元素一致，以及随机 20 个基线行与逐条 `generate` 的答案比较。
