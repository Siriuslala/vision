# 序列顺序的位置作用：实验设计

本文属于[位置信息与坐标读取](plan.md)研究，为实验 3.8 的设计。结果见 [results_2026-10-15_sequence_order.md](results_2026-10-15_sequence_order.md)。模型为 Qwen3.5-9B，bfloat16、SDPA、greedy decoding，thinking off。

## 1. 问题与动机

实验 3.5 表明 visual token 在序列中的顺序是坐标答案的第二个位置来源：网格任务中，交换目标与相邻单元在序列中的槽位后，行号几乎完全跟随槽位（跟随比 0.9–1.0），列号部分跟随（0.4–0.6），而 decoder 的 MRoPE position IDs 没有因果作用。Qwen3.5-9B 的 decoder 中所有 token（包括图像 token）都受 causal mask 约束，32 层中 24 层为不使用位置编码的 Gated DeltaNet（带长度为 4 的 causal 卷积的线性递推），只有层 4, 8, …, 32 为 full attention。因此序列顺序可以经由两条途径转化为位置：causal mask 使每个 token 只能看到它之前的 token，DeltaNet 的递推本身对顺序敏感。

本实验回答三个问题：

1. 序列顺序以什么机制转化为位置，是否可以归结为对已读 token 的计数；
2. 为什么行号比列号更依赖序列顺序；
3. 这一机制在哪些层产生，以及它是否参与网格定位的失败（实验 1.3 的 20×20 网格列号多计 1）。

## 2. 候选机制

记图像的 merged token 网格为 $H \times W$（单位为 merged token，每个 token 覆盖 32×32 pixels）。原图中位于第 $r$ 行、第 $c$ 列的 token 称为 token $(r, c)$，其 raster 序号为 $rW + c$。序列中图像片段的第 $i$ 个位置称为槽位 $i$，槽位的 raster 坐标为

$$
(R_i, C_i) = \left( \lfloor i / W \rfloor,\; i \bmod W \right).
$$

一个槽位顺序记为排列 $\pi$，即槽位 $i$ 放入 token $\pi(i)$，token 携带自己的 visual embedding（因而携带 ViT 写入的位置）与自己的 MRoPE ID。对 $\pi$ 的干预只改变 token 在序列中的位置，不改变任何 token 的内容。

四个候选机制给出不同的预测：

| 记号 | 机制 | 目标 token 的位置估计 |
| --- | --- | --- |
| M0 | ViT 位置 | 目标 token 自身 embedding 中的 $(r, c)$ |
| M1 | 序号计数 | 目标所在槽位的 $(R_i, C_i)$，即由“之前有多少个 visual token”换算 |
| M2 | 行首计数 | 之前出现过多少个 ViT 列号为 0 的 token，即数已读的行首 |
| M3 | 序列邻域借用 | 目标在序列中相邻 token 的 ViT 位置，经 DeltaNet 递推或 causal attention 汇聚到目标 |

M3 能直接解释“行跟随、列部分跟随”：raster 顺序下一个 token 的序列邻居几乎都在同一行，沿序列汇聚邻居的 ViT 位置会保留行信息、平均掉列信息。M1 也能解释这一现象：行号是序号的粗粒度函数 $\lfloor n / W \rfloor$，列号是余数，难以由单调的计数信号线性读出。两者对下文的整体平移条件给出相反的预测。

## 3. 刺激与问题

1. **主刺激。** stimuli-v1 生成的 8×8 网格，单元 64 pixels，画布 512×512（$H = W = 16$），模板为 person 与 ring，目标为第 4 级颜色变化（纯红）。目标遍历全部 64 个单元，共 128 张图像。网格原点在画布左上角，因此单元 $(r, c)$ 恰好覆盖 token $(2r..2r{+}1,\, 2c..2c{+}1)$。
2. **非方形网格。** 8 行 4 列（画布 256×512，$H = 16, W = 8$）与 4 行 8 列（$H = 8, W = 16$），person 模板，目标遍历全部 32 个单元。
3. **20×20 失败样本。** 实验 1.3 的 T6 20×20 行列任务 48 张原图与原问题。
4. **问题。** 实验 1.3 的 T6 原问题：“One figure is drawn in red. Give its position. Return only {"row": r, "column": c}, counting rows from top to bottom and columns from left to right, starting at 1. Output only the JSON, with no explanation.”。8×8 与非方形网格的行列号均为一位数，便于读取数字分布。

## 4. 指标

1. **期望行列号。** 生成行号与列号的数字 token 时，取数字 1 至 $n$（$n$ 为行数或列数）的 logit 做 softmax，得到分布 $p_k$，定义期望行号（从 0 计）

$$
\hat r = \sum_{k=1}^{n} p_k \,(k - 1),
$$

期望列号 $\hat c$ 同理。列号在模型自己生成的行号之后读取。

2. **位移与跟随比。** 设标准顺序下的期望行列号为 $(\hat r_0, \hat c_0)$，干预后为 $(\hat r, \hat c)$，某一机制预测的目标位移为 $d$（单位为单元，沿干预轴）。跟随比定义为

$$
f = \frac{\hat r - \hat r_0}{d} \quad \text{或} \quad f = \frac{\hat c - \hat c_0}{d},
$$

取干预轴对应的分量。下文的 $d$ 一律取目标槽位的位移（即 M1 的预测）；$f = 1$ 表示答案完全跟随槽位，$f = 0$ 表示不受影响。区间为以图像为单位的 bootstrap 95% 区间，比例给出 Wilson 95% 区间。

## 5. 实验

### 5.1 E1：区分计数与邻域借用的槽位顺序

对每张 8×8 图像运行以下槽位顺序（全部保持 token 内容与 MRoPE ID 不变）：

| 条件 | 构造 | M0 | M1 | M2 | M3 |
| --- | --- | --- | --- | --- | --- |
| 块移动 | 目标的 2×2 token 与上、下、左、右相邻单元交换槽位（实验 3.5 的 S4） | 0 | 1 | 行 1 | 行约 1，列部分 |
| 整体行平移 | 所有 token 行循环下移或上移 2 行（一个单元），行内顺序不变 | 0 | 1 | 1 | 0 |
| 行内平移 | 每一行内部循环右移或左移 2 个 token | 0 | 1 | 0 | 约 0 |
| 打乱其余 | 目标槽位不变，其余 252 个 token 随机重排 | 0 | 0 | 约 0 | 行列失去依据 |
| 竖直携带邻域 | 目标块下移或上移一个单元，同时携带其所在 token 行中左侧 $a_L$、右侧 $a_R$ 个 token，$a_L, a_R \in \{0, 2, 4, 8, W\}$ | 0 | 1 | 1 | 随携带量减小 |
| 水平携带邻域 | 目标块右移或左移一个单元，同时携带其后方 2 或 4 个 token | 0 | 1 | 0 | 随携带量减小 |
| 全局重排 | 列优先读取（转置）、行倒序、行内倒序 | — | — | — | — |
| 插入空白 token | 在图像片段开头插入 32 个白画布中心 token、白画布的两整行 token，或 2 个白画布左边缘 token；以及在目标单元行之前插入两整行空白 token | 0 | 1、1、1/16、1 | 0、1、1、1 | 约 0 |
| 编辑图形行与单元 | 在目标单元行之前或图像开头复制一行图形；删除顶行或目标之上的一行；在目标所在 token 行内复制或删除目标左侧一个单元 | 0 | 按 token 数 | 按行首数 | 约 0 |


表中数字为对目标槽位位移的预测跟随比。编辑图形行与单元的条件在整体行平移与插入空白 token 的结果出现后加入，用于检验“对图形内容计数”这一第五种解释：若模型数的是已读的图形行，复制或删除一行图形应使行号恰好加 1 或减 1，而插入空白 token 不应产生影响。整体行平移是关键条件：它改变目标之前的 token 数与行首数，却保持每个 token 的全部序列邻居，因此 M1、M2 预测完全跟随，M3 预测不动。携带邻域条件给出 M3 的作用范围：若只需携带目标之前的少数 token 就能消除跟随，说明位置来自 causal 方向上的局部序列上下文。

非方形网格运行块移动、整体行平移、行内平移与打乱其余，检验结论是否依赖 $W$。

### 5.2 E2：产生槽位依赖的层

对块移动条件做 mixer 输出恢复。每层的 token mixer 指 DeltaNet 层的 linear attention 模块或 full attention 层的 self-attention 模块；MLP 逐 token 计算，不能产生顺序依赖。记标准顺序下第 $\ell$ 层 mixer 在 token $j$ 上的输出为 $m^{\ell}_{j} \in \mathbb{R}^{4096}$。恢复第 $\ell$ 层时，置换输入的前向中槽位 $i$ 的 mixer 输出被替换为

$$
\tilde m^{\ell}_{i} = m^{\ell}_{\pi(i)},
$$

即该槽位上的 token 在标准顺序下得到的输出。若恢复全部 32 层，图像片段的 residual stream 与标准顺序下的逐 token 相同（数值检验见结果文档），剩余的跟随只能来自文本 token 对图像片段的读取（DeltaNet 在文本位置上的递推状态依赖图像 token 的顺序；full attention 对其之前全部 key 的读取与顺序无关）。恢复的层组包括：全部层、全部 DeltaNet 层、全部 full attention 层、按深度划分的 DeltaNet 与 full attention 子组、只恢复目标槽位或只恢复其余槽位，以及逐层单独恢复（person 模板的下移与右移）。

### 5.3 E3：位置编码的逐层 probe

1. **平移输入。** 白画布与 8×8 网格各 40 个输入，每个输入的 token 顺序做随机的二维循环平移（行平移 $k_r$、行内平移 $k_c$），使槽位坐标 $(R, C)$ 与 token 坐标 $(r, c)$ 解耦。在 decoder 输入与 32 层输出的 visual 位置上训练 ridge probe，按输入划分训练集与测试集（每 4 个输入取 1 个测试），报告四个坐标的测试 $R^2$。
2. **相同 token 序列。** 图像片段的 256 个槽位全部放入白画布 visual embedding 的平均值，MRoPE ID 保留或全部设为同一值。此时 decoder 中唯一的位置来源是序列顺序。probe 序号 $n$、$\log(1 + n)$、$R$ 与 $C$，检验 causal mask 与 DeltaNet 是否在没有内容差异时产生序号编码。

### 5.4 E4：与 20×20 失败的关系

对 48 个 20×20 行列样本运行整体行平移、行内平移、左右块移动、水平携带邻域，以及“打乱目标所在 token 行”（目标槽位不变，同两行其余 token 随机重排）。若列号多计 1 来自序列邻域，打乱目标行会改变多计 1 的比例；若来自序号计数，行内平移会使列号整体偏移。行为层面的网格规模扫描由问题 1 承担，本实验只做机制干预。

## 6. 停止标准

1. 若整体行平移下答案完全跟随槽位，且插入空白 token 使答案按插入数量偏移，则序列顺序的作用归结为计数，本方向在补充 E2 的层定位后结束。
2. 若整体行平移不改变答案，而块移动与打乱其余改变答案，则序列顺序以邻域借用的形式起作用；E2 定位产生这一作用的层，E4 判断其是否参与失败。若 E4 显示其与失败无关，本方向结束，结论并入位置问题的总结。

## 7. 运行

程序为 `analysis_tools/position/sequence_order.py`（引擎）、`sequence_order_run.py`（各阶段）与 `sequence_order_analysis.py`（统计与作图），入口为 `scripts/analysis/position/sequence_order.sh`，产物位于 WORK_DIR 的 `analysis/position/2026-10-15_sequence_order/`。解码沿用实验 3.5 的手写 greedy 解码器（不含 padding 的同长 batch）。
