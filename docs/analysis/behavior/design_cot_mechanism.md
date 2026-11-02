# 思维链的行为与机理：实验设计

实验编号 1.7，属于[行为诊断](plan.md)研究。模型为 Qwen3.5-9B（冻结、bfloat16、SDPA、greedy decoding），默认 thinking off；thinking on 只用于小规模对照。结果见 [results_2026-10-21_cot_mechanism.md](results_2026-10-21_cot_mechanism.md)。

## 1. 问题与动机

实验 3.10 发现，行号在 prefill 中由图像 token 上的 DeltaNet 累加器给出：在 prefill 中破坏这一递推后，先推理再作答（think-then-answer，thinking off）与 thinking on 都不能恢复行号，推理文本仍逐行列举，却报告错误的总行数。实验 2.6 发现，先描述多数图形再作答不能弥补找不同的搜索不对称。实验 1.3、1.5、1.6 积累了约 1900 条推理文本，其中出现了虚构细节、“看不到图像”的声明与断言式的序号复述。这些现象共同指向一个假设：**VLM 的思维链（chain-of-thought, CoT）在视觉定位类任务中多为对 prefill 结果的复述，生成 CoT 的过程本身不重新读取图像，也不承担计算。**

本实验从三个层面检验这一假设：

1. **行为。** 在网格行列、坐标、找不同、行列计数与部分 BabyVision 题目上，CoT 的常见失败模式是什么，各占多少比例。
2. **忠实性（faithfulness）。** 答案是否依赖 CoT 的内容：截断或编辑 CoT 后答案是否随之改变；先给答案再解释与先解释再给答案是否得到同一个答案。
3. **机理。** CoT 生成时信息从哪里来：生成的 token 是否注意图像 token；注意时是否真正读取了信息；若不读取，推理依赖的是问题 token 上 prefill 形成的摘要，还是 DeltaNet 的循环状态。

## 2. 符号

| 符号 | 形状 | 含义 |
| --- | --- | --- |
| $P$ | 标量 | prompt 的 token 数（含 chat template），同一任务内所有图像相同 |
| $\mathcal{I}$ | 区间 $[s_0, s_1)$ | 图像 token 在序列中的位置 |
| $\mathcal{Q}$ | 区间 $[s_1, P-1)$ | 图像之后的问题文本与模板 token 的位置 |
| $\mathcal{G}$ | 集合 | 已生成 token 的位置 |
| $\ell$ | 标量 | 全注意力层编号，取 $\{4, 8, \dots, 32\}$（1-based，共 8 层） |
| $h$ | 标量 | 注意力 head，$1 \le h \le 16$ |
| $a_{t,\ell,h,j}$ | 标量 | 生成第 $t$ 个 token 的前向中，层 $\ell$ head $h$ 对位置 $j$ 的注意力权重 |
| $K^{\mathcal{I}}, V^{\mathcal{I}}$ | 每层 $(4, \lvert\mathcal{I}\rvert, 256)$ | 全注意力层中图像位置的 key/value cache |
| $K^{\mathcal{Q}}, V^{\mathcal{Q}}$ | 每层 $(4, \lvert\mathcal{Q}\rvert, 256)$ | 全注意力层中问题文本位置的 key/value cache |
| $S^{\Delta}$ | 每层 $(32, 128, 128)$ 与卷积状态 | 24 个 Gated DeltaNet 层在 prefill 结束时的循环状态 |

生成第 $t$ 个 token 时，层 $\ell$ head $h$ 放在图像上的注意力质量定义为

$$
m_{t,\ell,h} = \sum_{j \in \mathcal{I}} a_{t,\ell,h,j}.
$$

$m$ 取值在 0 与 1 之间，表示该 head 在这一步读取的信息中来自图像 token 的份额；同理定义问题文本份额 $q_{t,\ell,h}$（求和范围为 $\mathcal{Q}$）与已生成文本份额 $g_{t,\ell,h}$（求和范围为 $\mathcal{G}$）。注意力质量只说明“看向哪里”，不说明读到的内容是否被使用，后者由第 4.3 节的因果干预检验。

在 decoding 阶段，生成 token 能接触 prefill 信息的途径只有三条：全注意力层中的图像 cache $(K^{\mathcal{I}}, V^{\mathcal{I}})$、问题文本 cache $(K^{\mathcal{Q}}, V^{\mathcal{Q}})$，以及 DeltaNet 循环状态 $S^{\Delta}$。DeltaNet 层在 decoding 时不访问任何历史 token，只读写 $S^{\Delta}$。因此，把这三条通道逐一替换为另一张图像的版本，就能把“CoT 的内容来自哪里”分解为三个可测的份额。

## 3. 刺激与问题

### 3.1 新生成的四个任务集

全部图像由 stimuli-v1 生成（`analysis_tools/common/stimuli.py`），单元 64 pixels，每个任务 64 张，模板各半。

| 任务 | 网格 | 目标 | 答案格式 | 与已有实验的关系 |
| --- | --- | --- | --- | --- |
| rowcol | 8×8，person/ring | 红色图形 | `{"row": r, "column": c}` | 与 3.10 的 8×8 条件同构，think 问法与 1.5/3.10 相同 |
| odd | 6×6，person/arrow | 16 pixels 缺口 | `{"row": r, "column": c}` | 1.3 T5 的显式推理问法 |
| point | 10×10，person/ring | 红色图形 | `[{"point_2d": [x, y]}]`，0–1000 | 3.11/3.12 的坐标读出，2.5e 中 10×10 点坐标准确率 0.24 |
| count | 行列各 3–12 随机，person/ring | 无 | `{"rows": R, "columns": C}` | 1.3 T0 |

每个任务使用三种问法：direct（只输出 JSON）、think（先逐步说明，再在最后一行给出 JSON）与 answer-first（先在第一行给出 JSON，再解释）。rowcol 与 point 另取 8 张做 thinking on。

### 3.2 已有语料

| 语料 | 来源实验 | 条数 | 说明 |
| --- | --- | --- | --- |
| dec13_explicit / dec13_thinking | 1.3 | 300 / 52 | T3 二值判断与 T4–T6 行列，6×6 与 20×20 |
| ord15_think | 1.5 | 480 | H32/H64/V64 实例行列，含配对变体 |
| pe16_explicit | 1.6 | 156 | PE ×1 条件的 T4–T6 显式推理 |
| rc310_think | 3.10 | 200 + 392 | 8×8 与 V64；无干预与 prefill 门控干预（$\alpha = 1$、$\beta = 0$） |
| odd26_think | 2.6 | 192 | 找不同的 std 与 swap，颜色、缺口与自然物体 |
| bv16 | 1.6 | 99 | BabyVision 六个子类，thinking off，参考运行的 sampling 解码 |

## 4. 实验

### 4.1 E1：失败模式分类（行为）

1. **规则编码。** 不使用 LLM，以正则规则从每条推理文本中抽取：最终答案；推理正文中最后一次陈述的目标行与列（排除“Row 3: …”这类逐行标签、范围表述与坐标系说明）；声明的网格行列数；逐行或逐列列举及其中提到红色的行标签；“看不到图像”类声明；“let's assume”类无依据假设；自我修正词；“前面有 $n$ 个，所以是第 $k$ 个”中 $k \ne n+1$ 的算术不一致。
2. **主失败模式。** 对答错的回答按以下优先级取第一个成立的类别：截断、看不到图像声明、答案与 CoT 不一致、网格规模错误、算术不一致、逐项列举但标签错位、断言错误位置、未陈述位置。对答对的回答，记录反向情形：CoT 陈述的位置错误而答案正确、带有看不到图像声明而答案正确、网格规模错误而答案正确。
3. **人工抽查。** 按类别分层抽取 40 条，人工判断陈述位置、一致性与主失败模式，报告与规则编码的一致率。

### 4.2 E2：忠实性检验

1. **截断后强制作答（early answering）。** 对每条 think 回答，取最终答案之前的正文 token，按比例 $f \in \{0, 0.2, 0.4, 0.6, 0.8, 1\}$ 截断，接上答案提示（例如 `Final answer: {"row": `），greedy 解码答案。若答案在 $f = 0$ 时已与完整 CoT 的答案一致，CoT 对答案没有必要性。
2. **编辑陈述的数值。** 找到正文中最后一次陈述的行号（count 任务为声明的总行数），改为相差 1 的值，截断在被改的数字之后，让模型继续生成到结束；同样处理第一次陈述的行号。对照条件在同一位置截断但不修改。答案随编辑改变的比例衡量答案对 CoT 的依赖；第一次陈述被改后模型是否在后文纠正，衡量 CoT 中是否存在重新读取图像的过程。
3. **先答后解释。** 比较 answer-first 的第一行答案、direct 答案与 think 答案。若三者一致，CoT 不改变答案；answer-first 的解释若始终支持已给出的答案，说明解释是事后合理化。

### 4.3 E3：CoT 生成中的信息流（机理）

每个机理任务（rowcol、odd、point）取 32 个配对 $(A, B)$：同一模板与网格，目标单元在行与列上都至少相差 2。两张图像的 token 布局完全相同，因此 cache 可以逐位置互换。decoding 由自写的 greedy 循环完成：prefill 只覆盖前 $P-1$ 个 token，最后一个 prompt token 作为第一个 decoding 步输入，所以全部生成 token 都由编辑后的 cache 计算。

1. **注意力时间线（问题 3a）。** 在无干预的 A 上，记录每个生成步、每个全注意力层、每个 head 的 $m$、$q$、$g$，以及 head 平均的图像注意力分布与层 20 head 11（3.11 确认的坐标读取 head）的图像注意力分布。按生成 token 的类别（数字、颜色词、“row/column”、图形名词、格式符号）汇总 $m$，并检验注意力是否落在目标单元或 CoT 正在陈述的行。
2. **只在 CoT 阶段改变图像（问题 3b）。** 条件如下表；“follow A”与“follow B”分别指答案或 CoT 首次陈述的位置等于 A 或 B 的真值（point 任务按答案点更接近哪个目标计）。

| 条件 | prefill | decoding 时的图像 cache | 问题文本 cache | DeltaNet 状态 |
| --- | --- | --- | --- | --- |
| clean_A | A | A | A | A |
| clean_B | B | B | B | B |
| mask_img | A | 不可见（注意力屏蔽） | A | A |
| img_from_B | A | B | A | A |
| B_prefill_img_from_A | B | A | B | B |
| txt_from_B | A | A | B | A |
| delta_from_B | A | A | A | B |
| txt_delta_from_B | A | A | B | B |
| all_from_B | A | B | B | B |

   若 CoT 生成时真正读取图像，img_from_B 的答案应转向 B，B_prefill_img_from_A 应转向 A；若 CoT 只复述 prefill 结果，两者分别保持 A 与 B，mask_img 也不改变答案。
3. **不看图像时的信息来源（问题 3c）。** txt_from_B、delta_from_B 与 txt_delta_from_B 分别替换问题文本 cache、DeltaNet 状态与两者。txt_delta_from_B 转向 B 的比例接近 all_from_B 时，CoT 的位置信息全部来自 prefill 中形成的文本位置摘要与循环状态；两种单独替换的比例说明二者各自的份额。all_from_B 应与 clean_B 一致，作为实现的正确性检查。

### 4.4 E3d：与 3.10 的连接

rowcol 的 32 张图像各做两次 prefill：一次无干预，一次在层 9–16 的 DeltaNet 层中对图像 token 令 $\alpha = 1$（3.10 中破坏行计数的干预）。decoding 条件为：编辑后的 prefill 整体；干净 prefill 配编辑后的图像 cache；编辑后的 prefill 配干净的图像 cache。若 3.10 的编辑只经由图像 cache 进入 CoT，第二个条件应复现编辑整体的少计，第三个条件应恢复正确行号。

### 4.5 运行规则

1. 每个机理任务先运行第一批 16 对；若第一批 clean_A 的精确准确率低于 0.25（point 不适用，因为它按更近的目标计），模型没有在图像中找到目标，通道替换无法区分 A 与 B，停止该任务的后续批次。
2. 截断检验使用全部以 eos 结束的 think 回答；编辑检验每个任务取前 32 条，续写上限 512 tokens。
3. GPU 紧张时允许以 device_map 把模型拆分到两张卡上运行，拆分不改变数值路径。

### 4.6 指标与统计

比例报告分母与 Wilson 95% 区间；同一批图像的条件比较用双侧精确 McNemar 检验。批量 greedy 解码与 `qwen_runtime.generate`（batch size 1）在 20 个样本上做答案级一致性检查。自写 decoding 循环与 `generate` 的文本一致性在 smoke test 中检查。

## 5. 与已有工作的关系

Lanham et al. (2023, *Measuring Faithfulness in Chain-of-Thought Reasoning*) 在纯文本 LLM 上用截断、加错与改写检验 CoT 的忠实性；Turpin et al. (2023) 发现 CoT 会为受偏置影响的答案给出事后理由。本实验沿用截断与编辑两种行为检验，但目标在于视觉定位，并把忠实性问题落到架构上的信息通道：在 Qwen3.5 的混合架构中，decoding 只能经由图像 cache、问题文本 cache 与 DeltaNet 状态接触 prefill，逐通道替换可以直接回答“CoT 是否重新读取图像”，这是纯文本工作无法提出的问题。

## 6. 产物

- 代码：`analysis_tools/behavior/cot_mechanism.py`（提示、刺激、规则编码）、`cot_mechanism_engine.py`（decoding 循环、cache 编辑与注意力探针）、`cot_mechanism_run.py`（GPU 阶段）、`cot_mechanism_taxonomy.py`（语料编码）、`cot_mechanism_analysis.py`（统计与作图）。
- 入口：`scripts/analysis/behavior/cot_mechanism.sh`；测试：`tests/analysis/behavior/test_cot_mechanism.py`。
- 数据：WORK_DIR 下 `analysis/behavior/2026-10-21_cot_mechanism/`；图：`docs/analysis/behavior/figures/cot_mechanism/`。
