# 一般视觉推理任务的 SFT 数据生产

本流程将已下载的图像与 QA 转换为 schema 1.1 的局部观察和连续过程候选，优先保留有图像依据的 interleaved 示范。原始来源、候选标注和正式验收分开保存。当前实现支持五个来源的有限试制和流式转换，尚未形成完成批次验收的训练集。

## 1. 数据来源与转换方法

| 来源 | 实际主要字段 | 过程构造 | 详细说明 |
| --- | --- | --- | --- |
| Visual-CoT | metadata JSONL 的 question、answer、image、bboxs；部分 GQA 另有 thought | 原图像素 ROI 合并为观察窗口，读取后保留原 thought；无 thought 时保留纯观察示范 | [Visual-CoT](sft_visual_cot.md) |
| Thyme-SFT | Parquet 的 image（Base64 列表）、question、response | 静态解析受支持的 crop 与 resize，将嵌套窗口映射回原图，核对真实派生图并保留语言顺序 | [Thyme](sft_thyme.md) |
| PixMo-Count | image_url、image_sha256、points、count、label | 下载并核对原图，point→候选实例框→人工审核→空间排序与独立分组→局部数量、累计和求和 | [PixMo](sft_pixmo.md) |
| CoSyn-400K | image、id、qa_pairs、metadata、data、code | 保留原 QA 与离线 explanation，本地教师或人工补充与静态原图对应的 ROI 和语言 | [CoSyn](sft_cosyn.md) |
| LLaVA-OneVision | image、id、conversations、data_source | 从注册子集提取独立单轮 QA，明确答案边界，本地教师或人工补充 ROI 过程 | [LLaVA](sft_llava.md) |

Parquet 中的内嵌图片直接解码；Visual-CoT 的分片 tar 使用独立、安全、可限量的流式提取。转换程序不执行来源中的任意 Python、DOM 脚本或工具代码。正在下载的临时文件不作为完成分片读取，输出目录与下载源分离。完整但损坏的分片明确报错；缺图、错误框、无过程证据等逐样本原因进入拒绝日志。

同一数据集可能包含多个监督等级。仅有 QA 不足以建立完整读取轨迹；仅有 explanation 也不能确定每一步区域。每条标准过程都必须具有实际图像、问题、窗口与参照、事件顺序和最终答案。

## 2. 公共格式与存储

`data/sft/core/common.py` 构建 schema 1.1。adapter 的内部候选以原图像素 XYXY 表示窗口，公共构造器将其转换为长边归一化的中心和完整宽高。源图片使用内容哈希命名并保存原始编码；遇到非 identity EXIF 明确要求进行坐标映射，首版不自动改变标注朝向。

数据图片写入 `DATA_DIR` 下的独立目录，运行日志和候选 JSONL 写入 `WORK_DIR`。标准窗口只保存一份归一化几何。来源 explanation、程序摘要、实例标签和审核信息仅保存在 metadata，训练加载器不得将其整体拼入模型输入。

每次生产生成：

- `candidates.jsonl`：完整格式的 `pending` 候选；不能直接当作正式训练输入。
- `rejections.jsonl`：来源 ID、拒绝原因及适用时的标注 key。
- `manifest.json`：配置、输入文件快照、模式数量、转换与拒绝统计、完成状态。

无可用候选时返回非零状态；运行异常保留失败 manifest。源样本重复编号、错误几何和模式错误均不伪装为成功。

所有来源共用同一个 `split_registry.sqlite`，默认位于各来源图片输出目录的父目录。注册表同时检查原图 SHA256 和结构 group_id 的 split 一致性。使用不同存储布局时，须通过 `--split-registry` 显式指定同一文件。冲突记录拒绝；该检查不能代替近重复图、重新压缩图、结构变体和上游评测来源的去污染。批次验收仍需记录外部测试重叠检查。

## 3. 本地教师与人工标注

`data/sft/tools/teacher.py` 使用项目缓存中的 VLM，根据原图和问题提出最多四次读取及相应自然语言。原始答案不输入教师，输出另存，便于独立比较。模型快照、seed、prompt、完整返回和截断状态保留在试制记录。缺少自然语言、非法框、无 EOS 或无法解析的结果明确拒绝。

教师坐标采用提示中声明的按宽高分别映射到 0–1000 的刻度，解析器立即还原为像素坐标。该刻度只属于教师接口，标准训练数据仍采用长边归一化。

人工或教师标注使用 JSONL，每条包含：

```json
{
  "source_key": "由原始样本ID、问题、答案及图像哈希共同计算的内容key",
  "status": "reviewed",
  "reviewer": "审核者标识",
  "events": [
    {"type": "visual_read", "box": [100, 50, 300, 150], "reference_region_id": 0},
    {"type": "reasoning_text", "text": "与该次观测一致的具体判断。"}
  ],
  "visual_only_safe": false,
  "visual_only_basis": ""
}
```

此处 `box` 是原图像素连续边界，示例数值只说明格式。`status` 支持 `proposed`、`reviewed` 和 `rejected`；拒绝标注必须给出 `reason`。内容 key 防止同路径换图、修改问题或答案后错配旧轨迹。`reviewed` 表示区域标注已审核，公共输出仍需经过独立的批次和实际特征检查。

对 interleaved，检查语言是否依据此前的原图输入或实际读取，是否存在遗漏证据、错误计数和模板化猜测。对 visual_only，标注必须以布尔值明确许可，并给出删除语言后的过程依据。单次直接读取题可以共用几何；假设增加物体、复杂算术及依赖语言维护的状态须单独判断。Counting 的纯视觉对照按 [计数方案](../counting_trajectories.md) 保留相同几何，并正确更新计数 metadata 中的事件编号及空文本引用。

## 4. Point 到窗口

规则图形路线需显式声明 `regular_compact` 条件，使用最近邻间距提出多个尺度，并通过图像连通区域和阈值稳定性细化。结果继续作为可选候选框。

SAM 路线使用本地 `Sam3TrackerModel` checkpoint，缓存图像编码，通过正负点取得多候选 mask。模型配置、权重完整性及数值有效性必须通过检查。当前已核对安装版本的 processor 维度与后处理接口，尚未使用真实 checkpoint 验证推理。mask 采用半开像素边界形成 bbox，目标身份、部件层级、遮挡与完整性由审核记录确认。

point 候选保存在 `candidate_instances`。审核者选择后填入 `instances` 并确认实例框及完整性，之后才产生分组计数轨迹。当前计数实现提供保守的 `exclusive` 分组；相交实例无法独立计数时明确拒绝，`new_instances` 尚未实现。空间面积阈值不是实际缓存特征可辨性的证明，该项仍需按最终读取配置审核。

## 5. 审核与入库

`data/sft/tools/review.py` 可以只为几个候选生成窗口叠加图、问题、答案、完整事件和离线来源信息，供检查。审核模板使用 `pending`，不会自动同意。

正式入库要求逐条接受决定绑定候选内容哈希，确认任务答案、观察过程及实际特征可辨性，并提供与候选文件 SHA256 绑定的批次验收。批次 JSON 包含 `approved`、`reviewer`、`report`、`source_overlap_check`、`feature_config_id`、`candidates_sha256` 和 `split_registry`。后两项分别绑定具体候选文件和跨来源划分注册表。批次报告中的抽检范围如实记录；首版入库工具只导出明确接受的样本。

`validate_record` 检查主字段、事件、几何、来源、答案和扩展计数结构；计数检查涵盖引用、唯一实例、可见与新增集合、累计数、覆盖与最终答案。它不从格式推断语义正确，也不代替训练时的 verifier 注册表。当前产物全部保留 `pending`，未将图像审核或单元测试冒充完整验收。

## 6. 2026-09-14 小样本试制

| 来源 | 实际尝试 | 结果及边界 |
| --- | --- | --- |
| Visual-CoT | 官方 CUB 两图、四条 QA | 输出八条候选，两模式各四；无原 thought，未编造语言；未验证 GQA detailed 的完整图像轨迹 |
| Thyme | 八条真实来源记录 | 两条 interleaved 候选、六条派生图对应不一致拒绝；RGB MAE 阈值是启发式筛选 |
| CoSyn | 两条 QA，本地 Qwen3-VL-4B 提议后看图核查 | 标题读取一条通过区域初审，输出两模式各一；人口比较的教师框未覆盖充分证据，拒绝 |
| LLaVA | 两条 CLEVR-Math QA，本地 Qwen3-VL-4B 提议后看图核查 | 增加四个棕球的问题保留一条 interleaved；另一条原答案与可见物体计数矛盾，保留原标签并拒绝 |
| PixMo | 七条 URL 图片 | 下载成功但全部原始 byte SHA256 不匹配，没有输出真实 SFT；几何、分组与格式链用独立 fixture 验证 |

CoSyn 的部分 explanation 使用底层 CSV 的数值，静态图片未必呈现这些数值。Thyme 的部分派生图包含源码 crop 范围外的上下文。LLaVA 中也存在原始答案不可靠的情况。上述发现要求按来源审核图像可回答性和过程证据；本次极小样本不能估计总体合格率。

本地下载尚不完整：Visual-CoT 的 metadata 与大图像分片需完成；CoSyn 当前已读到 chart 子集，其他领域需相应分片。试制没有启动全量解压、全量制数或全量数据校验。安装与 Bash 入口见 README。

## 7. 开发约定与目录

数据集适配器集中在 `data/sft/datasets/`，公共格式、标注和存储在 `data/sft/core/`，生产及离线标注入口在 `data/sft/tools/`，交互浏览在 `data/sft/viewer/`。对应 Bash 入口位于 `scripts/sft_data/`，数据集处理文档位于 `docs/sft_data/`。研究设计文档保留在 `docs/`。

后续本地教师默认采用 `Qwen3.5-9B`；资源受限时可在脚本中选择 `Qwen3.5-4B`，亦支持对应 `models--Qwen--…` 别名。现有 Qwen3-VL 试制记录保留其真实模型来源，不改写为新模型结果。当前尚未开展 Qwen3.5 两种规模的质量与速度对照，默认 9B 表示资源充足时的运行选择。

Counting 生产暂停，待 SAM 3 下载完成并确认本地权重后恢复。现有 point 和 counting 代码保留，本阶段不继续下载图片或运行分割。

交互审核工具的配置与使用见 [网页说明](viewer.md)。
