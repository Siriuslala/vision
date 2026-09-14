# 视觉推理训练数据与执行记录规范

本文定义候选数据格式 `1.1`，配合 [训练方案](visual_reasoning_training_plan.md) 使用。格式适用于单张图像上的直接回答、连续视觉读取及视觉语言交替推理。本文是实现约定，尚未实现相应训练加载器或 GRPO 执行器。

静态训练数据采用 UTF-8 JSONL，每行一个完整对象。SFT 保存完整专家示范，RL 题目保存问题、答案与验证配置；模型自主产生的 rollout 单独记录。字段未注明可选时均为必填；不接受未定义字段，扩展信息统一写入 `metadata`。JSON 数值必须有限，不允许 NaN 或 Infinity。

## 1. 样本字段

| 字段 | 类型与取值 | 定义 |
| --- | --- | --- |
| `schema_version` | string，固定 `"1.1"` | 本规范版本 |
| `record_type` | `"sft"` 或 `"rl"` | 训练用途 |
| `sample_id` | 非空 string | 当前数据发布版本内唯一；不同问题、模式或示范分别编号 |
| `task_type` | 非空 string | 使用第 1.1 节的任务名称 |
| `image` | object | 单张输入图像，见第 2 节 |
| `query` | 非空 string | 实际输入问题；不拼接来源注释或参考答案 |
| `answer` | object | 独立监督或评分目标，见第 5 节 |
| `reasoning_mode` | `"visual_only"` 或 `"interleaved"` | 中间过程是否允许自然语言 |
| `coordinate_system` | 固定 `"image_long_side_normalized_xy"` | 横纵坐标及尺寸均除以原图长边 |
| `initial_region` | object，覆盖整图 | `region_id` 为 0；`size` 为原图宽高分别除以长边，`center` 为该尺寸的一半，见第 2 节 |
| `trajectory` | array 或 null | SFT 为完整事件数组；RL 题目必须为 null |
| `verification` | object 或 null | 验证器配置；RL 必填非 null，开放式 SFT 可为 null |
| `provenance` | object | 原始来源、结构分组与转换版本，见第 6 节 |
| `quality` | object | `status` 与 `notes`，见第 6 节 |
| `metadata` | object，可选 | 离线扩展信息；每个数据发布版本另行声明内部字段 |

`sample_id` 标识一条训练记录，`image.image_id` 标识图像，`provenance.group_id` 标识必须共同划分的一组相关样本。三者用途不同。相同结构的不同渲染、不同问法、纯视觉与交替版本、SFT 与 RL 版本，均不得跨训练、验证和测试划分。

### 1.1 任务名称

首版使用 `recognition`、`captioning`、`general_vqa`、`grounding`、`detection`、`counting`、`ocr`、`document_qa`、`table_qa`、`chart_qa`、`spatial_reasoning`、`curve_tracing`、`maze`、`difference_detection`、`visual_comparison`、`puzzle`。类别用于采样与评价，不选择题型专用的推理求解器。

扩展任务名称须更新数据发布说明与加载器注册表，不将未知名称自动归入 `general_vqa`。一题按主要评价任务归类，其他能力标签可置于 `metadata`。

## 2. 图像与坐标

`image` 必须包含以下字段：

| 字段 | 类型 | 定义 |
| --- | --- | --- |
| `image_id` | 非空 string | 数据发布版本内稳定的图像编号 |
| `path` | 非空 string | 相对于 `DATA_DIR` 的 POSIX 路径；禁止绝对路径及 `..` 路径段 |
| `width`、`height` | 正整数 | 标注所依据图像的像素宽、高 |
| `sha256` | 64 位小写十六进制 string | `path` 对应文件的 SHA256；仅格式示例允许 null |

坐标依据统一朝向后的原图文件；若源图带 EXIF 旋转，先规范化朝向再计算哈希、尺寸和坐标。模型 resize、padding 和视觉网格坐标不得替代这里的原图坐标，其映射由预处理配置记录。

### 2.1 长边归一化与有效边界

原图横轴向右、纵轴向下。长边取 `image.width` 和 `image.height` 的较大值，横纵坐标、窗口宽高均除以该长度。两轴使用相同单位，保留原图像平面中的长宽比例与方向。图像左上边界为 `[0, 0]`，右下边界为原图宽高分别除以长边的结果。

`center` 是 `[cx, cy]`，`size` 是 `[width, height]`；两者各含两个 number，表示矩形中心与完整宽高。整图、前序参考区域、当前读取窗口以及几何答案共用该坐标系。局部参照不改变坐标单位。

`initial_region.size` 等于归一化后的整图宽高，`initial_region.center` 等于该尺寸的一半。每个窗口的尺寸严格为正，且各轴不超过对应整图尺寸；中心减半尺寸不小于 0，中心加半尺寸不超过对应整图边界。校验采用 `1e-6` 归一化坐标容差，越界标注明确报错。

例如，1000 × 500 像素图像的有效范围为 `[0, 1] × [0, 0.5]`，整图区域的中心为 `[0.5, 0.25]`，尺寸为 `[1, 0.5]`。中心 `[0.6, 0.2]`、尺寸 `[0.2, 0.1]` 的窗口对应像素中心 `[600, 200]`、完整尺寸 `[200, 100]`，像素边界为 `[500, 150, 700, 250]`。

### 2.2 像素坐标还原与存储精度

窗口的标准字段只保存 `region.center` 和 `region.size`。像素中心与宽高由这四个数分别乘以原图长边得到；像素边界由中心加减半尺寸计算。图像尺寸与哈希确定坐标所依据的文件，执行记录通过 `sample_id` 关联同一图像。

像素坐标用于可视化和外部格式导出时即时计算。训练记录与 rollout 的窗口字段不增加像素副本，使标注、训练、重放和显示始终使用同一份几何参数。源数据中的原始像素标注保存在离线来源文件中，通过来源版本和记录编号追溯，文件路径与坐标约定登记于来源 manifest。

转换与校验使用 float64，JSON 导出保留 float64 往返所需的有效数字，不预先按固定小数位截断。模型采用较低计算精度时，rollout 保存执行器实际使用的窗口值。模型坐标精度由执行配置记录，日志序列化保留该值的精度。

像素坐标表示连续图像边界，允许小数。栅格化操作另行记录取整规则，例如左上边界向下取整、右下边界向上取整并采用半开区间；取整后的结果仅用于对应显示或栅格化操作。视觉特征取样使用实际连续坐标及预处理映射。

### 2.3 标注转换与校验

转换器先根据源数据约定还原像素几何，并应用必要的 EXIF 朝向或子图到原图映射，再将全部坐标和尺寸除以规范化朝向后的原图长边。源格式包括像素坐标、宽高分别归一化坐标及其他刻度，其单位、点位定义和边界端点约定由来源 manifest 明确记录。

对每条转换记录检查像素几何与导出坐标的往返误差，容差为原图长边乘以 `1e-6`；涉及取整时另行检查取整规则。短轴边界按实际宽高比校验。加载器仅接受本规范声明的版本与坐标标识，转换程序根据明确的源约定生成完整记录。

`region` 是观察窗口，可能包含多个物体及上下文；`answer` 中的目标框才是任务要求输出的框。两者不得互相替代。观察窗口也不是迷宫中实际行走的轨迹。

## 3. 事件序列

SFT 的 `trajectory` 为非空数组。`event_id` 从 1 开始按数组顺序连续递增；所有事件共享这套编号。视觉事件另有 `region_id`，只对视觉读取从 1 连续编号，不随文字事件递增。

所有 SFT 事件包含 `event_id`、`type`、`supervise`。其中 `event_id` 为正整数，`supervise` 为 boolean；可选 `annotation` 为离线说明字符串，不进入模型上下文。各类型只增加下列专属字段。

### 3.1 视觉读取：`visual_read`

| 专属字段 | 类型 | 约束与用途 |
| --- | --- | --- |
| `region_id` | 正整数 | 本次新建的区域编号，等于当前第几次视觉读取 |
| `reference_region_id` | 非负整数 | 指向整图 0 或此前已执行的区域；不能引用当前或未来区域 |
| `region` | object | 仅含 `center` 和 `size`，各为长度为 2 的数值数组 |

`reference_region_id` 明确给出本次操作采用的参照，`region` 给出最终执行几何。参照的中心与范围通过同一记录中的整图或前序事件唯一解析；不重复保存另一份参照几何，以免两份数值冲突。展示时应同时显示解析后的参照框与当前框。

参考区域不限制新窗口必须包含于其中。窗口可从局部区域移动到邻近位置，也可选择更早的区域重新定位。返回同一位置仍创建新的 `region_id`，保留本次读取产生的新模型状态；不得覆盖旧记录。仅引用旧区域不等于返回，是否返回应结合实际执行窗口判断。

例如，以下事件重新读取区域 2 所在的分叉。当前窗口与区域 2 的几何相同时，能够直接在原图上显示返回过程：

```json
{
  "event_id": 5,
  "type": "visual_read",
  "supervise": true,
  "region_id": 4,
  "reference_region_id": 2,
  "region": {"center": [0.2, 0.6], "size": [0.2, 0.2]}
}
```

模型可以使用相对偏移和尺度调整预测窗口。静态数据保存统一坐标下的结果，加载器根据参照换算监督；内部参数变换由模型配置定义，不将多套冗余坐标混写为真值。

### 3.2 中间语言：`reasoning_text`

专属字段为非空字符串 `text`。该文本是训练时进入上下文、推理时由模型生成的内容。例如，在观察死路后记录“这条支路不通，回到上一个分叉点”，随后执行返回操作。

`visual_only` 禁止此类事件；`interleaved` 允许任意数量，包括 0。语言可表达计划及中间结论，不要求逐步复述坐标。离线的标注理由、教师检查结果和完整结构答案应写入 `annotation` 或 `metadata`，不能自动转成语言事件。

### 3.3 结束中间过程：`finish`

没有专属字段。SFT 中必须恰好出现一次，并位于最后，`supervise` 必须为 true。随后加载器以 `answer` 构造最终回答的生成目标。

`finish` 表示进入最终回答阶段，可由聊天模板的答案起始标记或专门控制 token 实现；整个序列的 EOS 位于答案之后。其具体 token 映射属于训练配置。它不同于每次读取后由执行程序固定插入的 `<end_read>`。

直接回答样本的事件数组仅包含 `finish`。`trajectory: []` 无效；`trajectory: null` 仅表示 RL 题目未提供专家过程，不能解释成“应立即作答”。没有完整过程的原始样本不得伪装成完整过程 SFT。

### 3.4 监督与实际输入

正常专家示范的 `supervise` 全部为 true。少量恢复示范可将人为设置的错误前缀标为 false，保留其观测作为上下文，只训练后续恢复。发现死路前的合法探索可以正常监督，不能仅因局部未到达出口就屏蔽。

`supervise: false` 取消该事件的直接动作或文本损失，不移除事件，不默认截断其计算图。视觉事件为 true 时监督读取决策、参照及窗口；语言事件为 true 时监督对应文本。视觉特征、程序插入的标记、来源字段、离线说明不计算生成损失。

模型前缀仅由图像、问题、模式控制和实际事件构建。`answer` 只在最终回答位置用于教师强制；`verification`、`provenance`、`quality` 和 `metadata` 不进入推理前缀。图像特征及 hidden states 在当前模型前向中计算，不存入通用静态样本。

## 4. SFT 与 RL 题目的完整性约束

| 检查项 | SFT | RL 题目 |
| --- | --- | --- |
| `trajectory` | 非空、顺序完整，最后为 `finish` | 必须为 null |
| 几何与参照 | 每次读取完整，引用可解析 | 自主执行时生成 |
| 中间语言 | 遵守模式；与事件顺序一致 | 由策略按模式生成 |
| `answer` | 最终输出监督 | 验证器参考信息，不提供给策略 |
| `verification` | 客观任务建议提供；开放式可为 null | 必须存在且已注册 |
| 可视化依据 | 专家实际示范的几何 | 模型 rollout 实际执行的几何 |

RL 没有预标注专家过程，但每次自主操作的中心、范围、参照都必须由执行器记录。可解释性来自该执行过程，不要求 RL 题目提前提供唯一标准轨迹。

## 5. 答案与验证器

`answer` 仅含 `type` 与 `value`。允许的类型如下；所有几何答案使用第 2 节的长边归一化坐标。点坐标和框边界的横纵上限分别为归一化后的整图宽、高，且下限为 0；路径中的每个点遵守相同范围。

| `type` | `value` 类型 | 语义 |
| --- | --- | --- |
| `text` | string | 文字答案；允许空字符串表示无文字内容 |
| `choice` | 非空 string | 选项的稳定标识，如 `"A"`，不存选项序号与文本的混合值 |
| `number` | 有限 number | 数值答案；容差和单位转换在验证配置中定义 |
| `points` | `[x, y]` 数组的数组 | 点集，允许空集；是否要求顺序由验证器明确规定 |
| `boxes` | object 数组 | 每项仅含 `label` 与 `xyxy`；`label` 为 string，`xyxy` 为 `[xmin, ymin, xmax, ymax]`，宽高严格为正；允许空集 |
| `polyline` | 至少两个 `[x, y]` 的数组 | 模型需要提交的有序路径；不得从观察窗口自动推导 |

本版不混用一个 `value` 的多种类型。复杂结构答案可按明确模板表示为 `text`，并使用版本化解析器；若需要新的原生结构类型，应升级规范。多个等价答案或数值容差记录在验证器参数中。

`verification` 非 null 时仅含：`verifier_id`（非空 string）、`version`（非空 string）和 `params`（object）。验证器在代码注册表中明确输入类型、答案解析、等价关系、质量分数和失败规则；`params` 不包含可执行代码。

推荐首版分别注册选择题精确匹配、文字规范化匹配、数值容差、点匹配、框匹配和路径合法性验证器。OCR 的大小写、空格及字符规范化必须固定；不能在看到结果后修改。每个验证器返回 0 至 1 的质量分数，执行有效性由运行器另行判断。

路径评分可在 `params` 中引用相对 `DATA_DIR` 的通道或障碍标注文件。这些文件仅供评分使用，不能输入策略。验证器只检查提交结果，不代为搜索正确答案。尚未实现或无法可靠评分的验证器不能用于正式 GRPO。

## 6. 来源、划分与质量

`provenance` 必须包含：

| 字段 | 类型 | 定义 |
| --- | --- | --- |
| `source_name` | 非空 string | 原始数据集或生成器名称 |
| `source_revision` | 非空 string | 原始数据版本、提交或发布快照 |
| `source_sample_id` | 非空 string | 原始记录编号 |
| `source_split` | 非空 string | 原始 split；程序生成使用 `generated` |
| `split` | `train`、`validation` 或 `test` | 本项目用途，不能由原始名称直接推定 |
| `group_id` | 非空 string | 用于禁止跨划分泄漏的结构或关联组 |
| `annotation_origin` | `human`、`program`、`teacher`、`derived` 或 `mixed` | 过程和答案标注的来源；不同字段的细分来源可放 `metadata` |
| `conversion_version` | 非空 string | 转换或过程生成程序版本 |
| `seed` | 非负整数或 null | 程序随机种子；不适用时为 null |

来源 manifest 另存官方入口、原始许可、坐标约定、教师版本、下载分片和处理统计，具体要求见 [公开数据调研](visual_reasoning_data_sources.md)。样本通过 `source_name` 与 `source_revision` 关联 manifest，不重复存储长篇许可文本。

由 point 派生的窗口可在 `metadata.window_derivations` 保存逐事件的标注来源，采用 [派生记录定义](point_to_window.md#5-标注记录与训练使用)。每项明确视觉事件编号、几何或分割方法、配置、来源点编号及离线审核记录路径。原始像素标注、候选 mask 与候选实例框保存在离线文件中，实际读取窗口仍由 `region` 唯一定义。

Counting 过程样本使用 `metadata.counting` 记录 bbox 空间顺序、分组计数模式、各步可见实例与新增实例、局部数量、累计数量和汇总事件，字段及校验要求见 [Counting 数据定义](counting_trajectories.md#6-数据字段与完整性检查)。这些标签用于离线审核及监督构造；交替模式中实际生成的计数文本保存在 `reasoning_text` 事件，模型自主执行时从视觉观测产生该文本。

`quality` 仅含 `status` 与 `notes`。`status` 为 `pending`、`validated` 或 `example`，`notes` 为 string。正式训练仅接受 `validated`；该状态表示完成图像、字段、几何、引用和任务要求的自动校验，并通过所属批次的质量验收。人工抽检范围及结论记录于批次报告，不能将批次抽检说成逐条人工审核。

`example` 只用于格式说明，允许图像文件不存在、哈希为 null；不能进入训练。`pending` 用于尚未完成质量审核的完整候选记录，仍须具有有效文件哈希。拒绝原因保留在离线审核记录中，不将失败记录默认为 `validated`。

## 7. GRPO 执行记录

Rollout 与静态题目分开保存，通过 `sample_id` 关联。执行事件及几何答案沿用关联题目的长边归一化坐标，像素几何按该图像的原图长边还原。执行配置记录坐标变换版本与计算精度。原始答案、隐藏结构及专家轨迹仍属于评分或监督侧；不能混入策略提示。下列字段为执行记录的最小必需内容。

| 字段 | 类型 | 定义 |
| --- | --- | --- |
| `schema_version` | 固定 `"1.1"` | 执行记录版本 |
| `rollout_id`、`sample_id` | 非空 string | 轨迹编号与源题目编号 |
| `rollout_group_id` | 非空 string | 同一题目、模式及 old policy 的一次采样组；不同于数据划分的 `group_id` |
| `policy_revision` | 非空 string | 实际采样的 old policy 快照 |
| `reference_policy_revision` | 非空 string | 固定 SFT reference policy 快照；不是参考区域编号 |
| `execution_config_id` | 非空 string | 关联不可变配置，含 tokenizer、模板、预处理、几何变换、最小窗口、预算、采样分布、标准差和随机种子 |
| `reasoning_mode` | 同静态样本 | 实际执行模式 |
| `events` | array | 实际执行事件；采用第 3 节事件字段，但不含 `supervise`、`annotation` |
| `policy_trace` | array | 按时间排列的随机决策与行为策略概率，定义见下文 |
| `final_answer` | `answer` 格式的 object 或 null | 解析后的实际提交答案，解析失败时为 null |
| `reward` | object | 仅含 `validity`、`quality`、`total`；前者为 0 或 1，后两者为 0 至 1 的 number，total 为二者乘积 |
| `termination` | string 枚举 | `finished`、`budget_exceeded`、`invalid_action` 或 `error` |
| `error_message` | string 或 null | 非正常终止的具体原因；正常完成为 null |

`events` 的几何必须是执行器实际使用的值；成功完成时末项为 `finish`，失败或超限可为不完整前缀。`finish` 仅表示已进入答案阶段，答案生成仍可能截断或解析失败。无效候选动作保留在 `policy_trace`，不伪装成已读取的区域。

`policy_trace` 每项含 `decision_id`（从 1 连续编号的正整数）、`kind`（下表枚举）、`event_id`（关联事件的正整数编号或 null）、`behavior_logp`（有限 number）。`behavior_logp` 是采样时实际使用的完整分布对该决策的 log-probability 或 log-density。决策类型及专属字段为：

| `kind` | 专属字段 | 概率计算单位 |
| --- | --- | --- |
| `token` | `token_id`：非负整数；`channel`：`reasoning`、`finish` 或 `answer` | 每个实际生成的语言、答案起始或最终答案 token；中间文本关联相应事件，最终答案 token 的 `event_id` 为 null |
| `visual_read` | `reference_region_id`：非负整数；`raw_window_action`：四维有限数数组；`component_logp`：含 `read`、`reference`、`window` 的数值 object | 一次完整读取，包括读取控制 token、参照与四维连续参数；`behavior_logp` 为三项之和 |

成功读取决策关联同编号的实际视觉事件。导致无效动作的读取决策保留原始采样值，`event_id` 为 null。四维原始参数经过配置指定的变换得到事件中的执行窗口；不以裁剪后的坐标替代原始动作。读取控制 token 已计入视觉决策，不能再记录为另一个 token 决策。

固定插入的视觉特征及 `<end_read>` 没有策略概率。采样分布的温度、掩码和支持集必须与新旧概率计算一致；首版不使用难以保持共同支持集的 top-p 截断。连续密度可大于 1，不能以“log-probability 必须非正”的文本规则误判。

训练根据实际动作和 token 前缀重算当前模型及 reference policy 的概率；不得把缓存的旧 hidden states 当作当前状态。原始 token 序列由 `policy_trace` 保留，解析失败也不丢弃。执行配置、模型版本和原始概率共同支持轨迹重放与策略比率核查。

## 8. 示例与入库检查

[JSONL 示例](examples/visual_reasoning_samples.jsonl) 包含交替迷宫示范、对应纯视觉示范、无专家轨迹的 RL 题目、直接回答样本、非正方形图像的局部定位示范及分组计数示范。定位示例采用 1000 × 500 像素图像，展示第 2 节的整图边界、局部窗口与几何答案。计数示例将 15 个目标分为 6、5、4 三组，另附 [离线实例标注示例](examples/counting_annotations.json) 用于核查排序、分组覆盖与求和。示例图像路径和答案仅用于解释字段，均标记为 `example`；文件不包含真实题图，不构成可训练数据或正确路径的实验证据。

入库时依次检查 JSON 类型与必填字段、图像尺寸和哈希、坐标范围、引用关系、事件顺序、模式约束、答案与验证器兼容性、来源及划分一致性。过程样本还需重放并核查区域是否支持题目所需信息；RL 另需检查验证器以及采样与概率记录的一致性。

几何合法并不等于示范正确。任务标注检查应覆盖曲线接续、迷宫探索、字段对应及最终答案；质量报告分别记录格式通过率、任务正确率和自主 rollout 成功率。
