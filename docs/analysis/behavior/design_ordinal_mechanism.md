# 行列序号的机理：实验设计

本文属于[行为诊断](plan.md)研究，为实验 1.5 的设计。结果见 [results_2026-10-16_ordinal_mechanism.md](results_2026-10-16_ordinal_mechanism.md)。模型为 Qwen3.5-9B，bfloat16、SDPA、greedy decoding，thinking off。

## 1. 问题与动机

前序实验给出了行列序号答案的四条线索：

1. 实验 1.4 表明，长轴含至少 16 个实例、单元不超过 64 pixels 时，列号系统性多计 1（下称 +1 偏差），+1 的起始位置对应第 5–8 个实例，与 token 位置或 pixel 位置无关；单元 32 pixels 的密集二维网格则出现 −1 偏差。
2. 实验 3.8 表明，行号由 DeltaNet 第 9–16 层沿 causal 序列对图形行计数得到，计数以被识别为顶行的图形行为起点。
3. 实验 3.5 的 C5 表明，在目标之前插入空白带时 6×6 与 10×10 网格中只有 9%–15% 的答案改变，序号主要由目标相对其他实例的次序决定。
4. 实验 2.5c 表明，行号的读取需要第 16 与第 20 层，列号在第 20 层由格式 token 从目标单元的 visual token 读入。

这些结果说明序号不由 pixel 位置直接换算，但尚未回答序号如何产生。本实验回答四个问题：

1. **Q1 生成路径。** 序号由以下哪条路径产生：(a) 先在内部定位到连续坐标，再换算为序号；(b) 直接使用位置信息（目标在规则网格上的槽位）；(c) 对目标之前的实例计数或累积。分别在只要求答案的 prompt（下称 direct）与允许先推理再作答的 prompt（下称 think，thinking 仍为 off）上回答，并统计推理文本中出现的计数策略。
2. **Q2 序号表征。** hidden states 中是否存在坐标之外的序号信息，位于哪些 token 与哪些层。
3. **Q3 起始位置。** +1 偏差在第 5–8 个实例开始出现，能否由简单的认知模型解释。
4. **Q4 行偏差。** 现有数据中行号的 +1 与 −1 有何规律，行与列为何不对称。

## 2. 候选路径与可区分的预测

### 2.1 符号

| 符号 | 含义 |
| --- | --- |
| $S = 21$ | 长轴上的槽位线数（H 条件为列，V 条件为行） |
| $s$ | 单元边长，单位 pixel |
| $P \subset \{1, \dots, S\}$ | 有实例的槽位线集合，$N = \lvert P \rvert$ 为实例线数 |
| $g$ | 目标所在槽位线的序号（从 1 计），等于目标中心 pixel 位置除以 $s$ 后向上取整 |
| $k$ | 目标所在实例线在 $P$ 中的序号，即“目标及其之前的实例线数” |
| $u$ | 目标中心沿长轴的归一化坐标，取值 0–1000 |
| $\hat{o}$ | 模型回答的长轴序号 |

### 2.2 三条路径的预测

| 路径 | 预测的序号 | 含义 |
| --- | --- | --- |
| (a) 坐标换算 | $r = 1 + (N - 1)\dfrac{g - g_{\min}}{g_{\max} - g_{\min}}$ | 在实例占据的范围内按比例换算，$g_{\min}$ 与 $g_{\max}$ 为首末实例线的槽位 |
| (b) 槽位 | $g$ | 序号等于规则网格上的槽位 |
| (c) 计数 | $k$ | 序号等于目标之前（含目标）的实例线数 |

路径 (a) 使用实例数 $N$ 与目标在占据范围内的相对位置；当实例均匀分布时，三条路径给出相同的答案，因此需要让它们分离的刺激。

### 2.3 配对变体

每张基础图像（base）为 6 行 21 列（H 条件）或 21 行 6 列（V 条件）的槽位网格，前 20 条槽位线有实例、第 21 条为空白，红色目标位于第 $c$ 条实例线（$c = 3, \dots, 18$）。每张 base 派生 4 个变体，变体只移动实例线或删除一条实例线，目标的模板、短轴位置不变：

| 变体 | 实例线 | $k$ | $g$ | 目标 pixel 位置 | 检验的问题 |
| --- | --- | --- | --- | --- | --- |
| base | 1–20 | $c$ | $c$ | $x$ | 基线 |
| shift | 2–21（空白在近端边缘） | $c$ | $c+1$ | $x + s$ | 不改变计数而移动目标 |
| insL | 1–21 去掉 $b \in [2, c]$ | $c$ | $c+1$ | $x + s$ | 同上，空白位于目标之前的内部 |
| delL | 1–20 去掉 $b \in [2, c-1]$ | $c-1$ | $c$ | $x$ | 不移动目标而减少计数 |
| delR | 1–20 去掉 $b \in [c+1, 19]$ | $c$ | $c$ | $x$ | 控制：实例数减 1 而计数不变 |

在配对比较中，记 $\Delta = \hat{o}_{\text{variant}} - \hat{o}_{\text{base}}$。三条路径对 $\Delta$ 的预测为：

1. 计数路径：shift 与 insL 的 $\Delta = 0$，delL 的 $\Delta = -1$，delR 的 $\Delta = 0$。
2. 槽位路径：shift 与 insL 的 $\Delta = +1$，delL 与 delR 的 $\Delta = 0$。
3. 坐标换算路径：shift 的 $\Delta = 0$；insL 的预测值为 $c + 1 - c/20$，四舍五入后在 $c < 10$ 时为 $+1$、$c > 10$ 时为 0；delL 与 delR 的预测值均接近 $c$，即 $\Delta \approx 0$。

delR 与 delL 改变相同的实例数 $N$，二者的差异只能来自目标之前的计数。H 条件取 $s \in \{32, 64, 96\}$，V 条件取 $s \in \{32, 64\}$；1.4 中单元 96 pixels 几乎没有 +1 偏差，作为对照。

## 3. 测量

### 3.1 行为

1. **direct prompt** 与 1.3、1.4 的 T6 行列问题相同：“One figure is drawn in red. Give its position. Return only {"row": r, "column": c}, counting rows from top to bottom and columns from left to right, starting at 1. Output only the JSON, with no explanation.”。
2. **think prompt** 为“One figure is drawn in red. First describe step by step how you determine its row and its column. Then give the final answer on the last line as {"row": r, "column": c}, counting rows from top to bottom and columns from left to right, starting at 1.”。1.3 的 explicit prompt 要求说明“有多少行和列”，会引导模型先数总数，本实验改用不指定策略的措辞。
3. 每个条件 64 张 base（每个 $c$ 4 张），共 320 张图像；H32、H64、V64 中每个 $c$ 取 2 张 base（160 张图像）另用 think prompt 提问。合计 1600 个 direct 与 480 个 think 问题。
4. 推理文本以规则编码六类策略：逐个枚举（“1, 2, 3” 或依次列出第 1、2、3 列）、从远端计数、比例或坐标估计、陈述总行列数、陈述目标之前的实例数、提及空白，并人工阅读约 20 条回答核对。

### 3.2 探针

以模型自己的回答做 teacher forcing，在偶数层（第 0 层为输入 embedding）的四个位置读取 residual stream：

1. last：prompt 的最后一个 token；
2. row0 与 col0：预测行号、列号首位数字的状态（即首位数字之前的 token），实验 2.5c 中读取视觉信息的格式 token 位于此处；
3. vis：目标单元覆盖的 merged visual token 的平均。

探针与对照如下：

1. **ridge 探针。** 目标为 $u$、$g$、$k$、模型答案 $\hat o$ 与 $r$；特征与目标都在条件内去均值，防止探针读取画布尺寸；以 base 为单位做 5 折交叉验证，正则化系数在训练折内以闭式 leave-one-out 误差选择。
2. **坐标之外的计数信息。** 在每个条件内用 $u$ 线性回归 $k$，取残差 $k_{\perp}$。在本设计中 $k_{\perp}$ 只由“目标之前是否有空白实例线”决定，与目标位置正交；若 hidden state 能读出 $k_{\perp}$，说明其中存在坐标之外的计数信息。
3. **分类探针与控制任务。** 对 $k$ 训练线性（one-hot 目标上的 ridge 最小二乘）与 MLP（一层 128 个 ReLU 单元，Adam 训练）分类器，输入为前 64 个主成分并在训练折内标准化。控制任务按 Hewitt & Liang（2019）的思路，把每个条件中的每个槽位 $g$ 随机映射到一个序号标签；选择性定义为 $k$ 的准确率减去控制任务的准确率。控制任务只依赖位置类型，若探针只靠记忆位置达到高准确率，控制任务的准确率同样高。
4. **只用坐标的基线。** 用交叉验证得到的 $u$ 读出值与条件编码训练 MLP 分类 $k$，与直接用 hidden state 分类的准确率比较。

### 3.3 沿方向的干预

在预测长轴序号首位数字的状态上（H 为 col0，V 为 row0），于第 16、20、24、28 层分别加入一个向量，并从该位置贪心续写序号。干预只作用于 base 变体，方向按 base 序号的奇偶交叉拟合，被干预样本所在折不参与其方向的估计。记被干预层上该状态为 $h \in \mathbb{R}^{d}$，模型在该样本上的回答为 $\hat o$，另一折中样本的状态与回答为 $\{(h_i, \hat o_i)\}$。加入的向量有四类：

1. **答案的线性编码方向** $a \in \mathbb{R}^{d}$：以 $\hat o_i$ 对 $h_i$ 做线性回归（全部五种变体），取 $\hat o$ 增加一个序号时状态的平均变化。加入 $\pm 3a$ 与 $\pm 10a$。
2. **条件均值差方向** $m_{\delta}$：

$$
\bar h(v) = \frac{\sum_i w_i(v)\, h_i}{\sum_i w_i(v)}, \qquad w_i(v) = \exp\!\left(-\frac{(\hat o_i - v)^2}{2}\right), \qquad m_{\delta} = \bar h(\hat o + \delta) - \bar h(\hat o).
$$

其中 $\bar h(v)$ 是另一折中回答接近 $v$ 的状态的核加权平均（带宽为 1 个序号），$m_{\delta}$ 把状态从“回答为 $\hat o$ 的典型状态”移向“回答为 $\hat o + \delta$ 的典型状态”，取 $\delta \in \{\pm 1, \pm 3\}$。采用均值差而非回归系数，是因为序号在状态中的编码可以是弯曲的（结果文档第 3.3 节的状态流形图），线性方向在远离均值处会偏离数据所在的区域。
3. **实验 3.6 的 x 编码方向**（仅 H 条件）：在 point prompt 上得到，单位化后缩放到 $\lVert m_{+3} \rVert$，正负两个方向。
4. **随机方向**：各向同性高斯向量，缩放到 $\lVert m_{+3} \rVert$，正负两个方向。

若长轴序号由状态中的连续量读出，$m_{\pm 1}$ 与 $m_{\pm 3}$ 应使回答按 $\delta$ 移动；同范数的随机方向给出非特异扰动的基线；3.6 的 x 方向检验 point prompt 中的坐标表示是否被序号读出所使用。

### 3.4 起始位置的模型（Q3）

用 1.4 的 6 行、相位 0 的 20 个条件（3920 个回答）拟合四个只有 3–5 个参数的模型。每个模型给出答案 $v$ 的离散化高斯分布，均值 $\mu(c)$、标准差 $\sigma(c)$，加上均匀的失误概率 $\varepsilon$：

$$
P(v \mid c) = (1 - \varepsilon)\,\frac{\Phi\!\left(\frac{v + 0.5 - \mu}{\sigma}\right) - \Phi\!\left(\frac{v - 0.5 - \mu}{\sigma}\right)}{Z} + \frac{\varepsilon}{v_{\max}}.
$$

其中 $\Phi$ 为标准正态分布函数，$Z$ 把 $v \in \{1, \dots, v_{\max}\}$ 上的概率归一化，$v_{\max} = n + 3$。四个模型为：

1. **subitizing 加近似计数。** $c \le K$ 时 $\mu = c$；$c > K$ 时 $\mu = c + \beta (c - K)$，$\sigma = \sigma_0 + w (c - K)$。对应人类“小数目精确、大数目近似”的计数方式。
2. **带偏差的比例估计。** $\mu = (c - 0.5)(1 + \gamma) + 0.5$，$\sigma^2 = \sigma_0^2 + (w c)^2$，即以正确的相对位置乘以偏大 $\gamma$ 的实例数。
3. **两端锚定的比例判断。** 以 $p = (c - 0.5)/n$ 为相对位置，$\mu = n p^{a} + 0.5$，$\sigma = \sigma_0 + w\sqrt{p(1-p)}\,n$；该模型在两端没有偏差，在中部偏大（$a < 1$），对应比例判断文献中的两端锚定模型。
4. **数字先验。** $\mu = c$，$\sigma = \sigma_0 + w c$，再乘以一个在全部条件间共享的数值先验 $\exp(\pi_v)$，检验 +1 是否落在被偏好的数值上。

以总对数似然、AIC 与 BIC 比较，并检查各模型对“最后一列很少回答 $n + 1$”的预测。

### 3.5 行偏差（Q4）

不新增运行，汇总 1.3 正式运行的 T6（3×3 至 20×20）与第 4 级 T4/T5、1.3 补充的 8 种布局、1.4 的全部相位 0 条件中的行 +1、行 −1、列 +1、列 −1 比例，按目标行号与布局分析，并与本实验 V 条件（行处于含 20 个实例的长轴）比较。

## 4. 与已有工作的关系

1. 视觉计数的心理学研究区分 1–4 个对象的 subitizing 与更大数目的近似估计；比例判断研究中，对位置的估计常以两端为锚点，呈现中部偏移的系统偏差。本实验把这两类模型作为可证伪的候选解释，用于检验 +1 偏差的起始位置。
2. Hewitt & Liang（2019）提出用控制任务衡量探针的选择性，本实验沿用其思路区分“表征中存在序号”与“探针本身能够拟合位置的任意函数”。
3. 实验 3.8 已在序列层面定位行计数；本实验在图像层面以配对变体检验计数，并把同样的干预用于列，直接比较行与列的路径。

## 5. 产物

WORK_DIR 下的 analysis/behavior/2026-10-16_ordinal_mechanism/。程序为 analysis_tools/behavior/ordinal_mechanism.py（刺激、问题、评分、路径预测与策略编码）、ordinal_mechanism_run.py（生成、抽查、特征采集与方向干预）、ordinal_mechanism_analysis.py 与 ordinal_mechanism_figures.py（分析与作图），入口为 scripts/analysis/behavior/ordinal_mechanism.sh。
