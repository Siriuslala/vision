# BabyVision 评测解码协议

评测需同时固定模型、输入、生成预算、解码配置和评分方法。Greedy decoding 适合建立稳定的单次输出基线；sampling 用于考察指定随机生成策略下的平均表现。两者回答不同的实验问题，模型规模和发布时间不能单独决定评测策略。

## 参数定义

| Bash 参数 | 含义 | 禁用或中性设置 |
| --- | --- | --- |
| `DO_SAMPLE` | 是否按经过筛选的 token 概率分布随机采样；关闭时每步选择最高分 token | `false` 使用 greedy |
| `TEMPERATURE` | 采样前缩放 logits；较低值通常使概率更集中 | `1.0` 不缩放；必须大于零 |
| `TOP_K` | 每步最多保留概率最高的 k 个候选 token | `0` |
| `TOP_P` | 保留按概率降序累积达到阈值的候选 token 集合 | `1.0` |
| `MIN_P` | 去掉概率低于最大 token 概率乘以该阈值的候选 | `0.0` |
| `REPETITION_PENALTY` | 对已出现 token 的 logits 施加乘性惩罚；大于 1 通常抑制重复 | `1.0` |
| `RANDOM_SEED` | 控制随机状态；采样时由实验 seed 和原始 batch 编号派生 batch seed | 无 |

`TOP_K` 的 k 是每一步的 token 候选数，与 pass@k 中的完整答案数量无关。多个筛选器共同约束候选集合。Greedy 模式不使用 temperature、top-k、top-p 或 min-p，但重复惩罚仍可改变输出。当前实现每个样本生成一个答案，使用 `num_beams=1`。

## Qwen3.5 的官方建议

以下配置按指定模型卡截图选择。[Qwen3.5-9B 模型卡的 Best Practices](https://huggingface.co/Qwen/Qwen3.5-9B#best-practices) 推荐按模式和任务选择 sampling 参数：

| 场景 | temperature | top-p | top-k | min-p | presence penalty | repetition penalty |
| --- | --- | --- | --- | --- | --- | --- |
| Thinking 通用任务 | 1.0 | 0.95 | 20 | 0.0 | 1.5 | 1.0 |
| Non-thinking 通用任务 | 0.7 | 0.8 | 20 | 0.0 | 1.5 | 1.0 |
| Non-thinking reasoning，Best Practices 表述 | 1.0 | 1.0 | 40 | 0.0 | 2.0 | 1.0 |

Bash 入口对 Qwen3.5 的 non-thinking 模式统一选择上表 reasoning 配置，thinking 模式选择 general 配置。Qwen3-VL Instruct 的 VL 配置为 temperature=0.7、top-p=0.8、top-k=20、repetition penalty=1.0；Qwen3-VL Thinking 的 VL 配置为 temperature=1.0、top-p=0.95、top-k=20、repetition penalty=1.0。两种 Qwen3-VL 配置均启用 sampling，min-p 显式设为 0.0，不调整生成长度。Instruct 与 Thinking 是独立 checkpoint，脚本对不兼容的 thinking 开关报错。

当前项目使用的 Transformers 原生生成接口没有 presence penalty 参数，本次实现提供 repetition penalty，二者含义不同，不能互相替代。因此，设置表中的 temperature、top-p 和 top-k 只构成部分参数对齐，不能声称完整复现 Qwen 官方采样设置。严格对齐 presence penalty 需要支持该参数的后端或经过验证的额外 logits processor。

模型卡对多数请求建议 32,768 个输出 token，对高度复杂问题建议 81,920。BabyVision 实验可先使用脚本中的预算检查截断比例，再决定是否统一增加预算。生成上限、实际生成长度和 `finish_reason="length"` 数量均应报告；达到上限的失败样本仍计入评分。

## 本项目的实验设置

建议保留两个明确区分的实验组：

1. Greedy 基线：Python CLI 不传 `--do-sample`，设置 `--repetition-penalty 1.0`，并为该对照提供独立 Bash 入口，固定 thinking、图像分辨率和输出预算，用于算法消融和跨模型受控比较。
2. Sampling 对照：预先固定解码参数，使用多个 seed 各完整运行一次 benchmark，每题每轮生成一个答案，报告各轮 accuracy 的均值和标准差。三个 seed 可作为初步实验，正式结论应结合波动大小确定重复次数。

跨模型比较可采用统一解码配置来控制变量，也可采用各模型的推荐配置来考察各自建议使用方式；后者必须逐模型报告参数。不能在 benchmark 上反复调参后只汇报最高分。多次生成后择优、投票或计算 pass@k 使用了不同的推理预算，应单独命名和报告。

[BabyVision 官方评测说明](https://github.com/UniPat-AI/BabyVision/blob/main/babyvision_eval/README.md) 默认 `NUM_PASSES=3`，采用 LLM judge，并汇总多轮均值和标准差。本项目目前使用本地严格答案匹配，尚未自动运行多 seed 或聚合跨轮指标。不同 seed 需分别配置 `RANDOM_SEED` 和独立 `OUTPUT_FILE`，通过 `bash scripts/eva/evaluation.sh` 启动各轮；本地严格评分不能直接等同于官方 LLM-judge 分数。
