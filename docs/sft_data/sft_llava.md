# LLaVA-OneVision ROI 过程数据生产

LLaVA-OneVision 混合数据以图像和问答为主要监督，当前已检查的记录没有独立 ROI 轨迹。生产流程先保留可可靠解释的题目和最终答案，再通过本地视觉教师提出区域及交替过程，由人工审核接口修订。完成轨迹后输出格式 `1.1` 的 `pending` 候选；批次质量验收完成后才能用于训练。

## 1. 本地结构与来源范围

源目录按子集组织 Parquet，文件直接支持逐 batch 读取，无须解压或改写。标准字段为 `id`、`image`、`conversations` 和 `data_source`。`image` 包含内嵌 `bytes` 与可选 `path`；`conversations` 为 `from`、`value` 消息数组，已检查的图像记录以 `human` 问题和 `gpt` 答案组成。问题用 `<image>` 标识单张图像。`Evol-Instruct-GPT4-Turbo` 的首条为 `image=null`，属于无图文本任务。

本次逐一读取 15 个本地子集的首条记录。观察到部分文件序号尚不完整，例如 `CLEVR-Math(MathV360K)` 仅出现 `of-00002` 中的一片，`VisualWebInstruct(filtered)` 仅出现 `of-00073` 中的六片。该观察是下载期间快照，不表示官方数据规模或下载完成。程序只枚举非隐藏目录下以 `.parquet` 结尾的文件；读取完成时检查大小和修改时间，文件损坏或变化明确报错。

| 子集 | 任务映射 |
| --- | --- |
| CLEVR-Math | `spatial_reasoning`，包含假设增删与数量变换 |
| FigureQA | `chart_qa`；已注册，当前未读到完成分片 |
| GEOS、GeoQA+、Geometry3K、UniGeo | `spatial_reasoning` |
| Super-CLEVR | `spatial_reasoning` |
| TabMWP | `table_qa` |
| IconQA | 明确包含 “how many” 的题目等待实例标注，其余为 `general_vqa` |
| MapQA、MathV360K_TQA、VQA-AS、VQA-RAD、PMC-VQA | `general_vqa` |
| VisualWebInstruct | `general_vqa`；明确描述图像的长答案为 `captioning` |

CLEVR-Math 首两题分别要求删除球后计数、假设增加四个棕色球后计数，属于假设场景变换后的数量推断。直接对图像实例进行计数的过程必须具备 `metadata.counting` 所定义的实例覆盖与累计审核链；目前 IconQA 的 “how many” 题目缺少这一元数据，明确拒绝并等待 point 或 bbox 实例标注，不能用普通 ROI 标注代替。

映射使用代码中的完整子集名称。未知来源拒绝并等待专项分析，不能随意归入通用任务。非 `train` 分片、显式 BabyVision 来源、无内嵌图像、无效角色或多图提示均拒绝。混合数据的 `train` 标签不能证明其上游题目没有出现在外部测试中，正式发布前仍需进行跨来源图像、问题与结构去重以及外部评测隔离。

## 2. 题目与答案

`source_id` 由子集、原始 `id` 和轮次 `turn0` 构成；分片路径、来源对话和答案提取方法写入离线 `metadata`。图像按内容分组，两个 reasoning mode 共用该组。来源 revision 优先使用 Hugging Face 下载 metadata 的 commit，缺失时记录本地分片名称、大小与修改时间；生产 manifest 另外保存输入快照。

题目只移除图像占位符，保留原始题干、提示与选项。当前仅转换一问一答：多轮对话整体进入拒绝记录，等待可保存上下文的专门转换，避免把依赖前文的追问独立成题。

答案优先解析明确的 `<answer>` 标签或已观察到的 `The answer is ...` 结构。选择题要求答案属于题目列出的选项标识。短直接回答可保存为 `text`；数值字符串沿用来源表示，避免引入未经定义的单位和容差。长描述仅在明确 caption 问题下作为最终文字答案保留。混杂推导但没有明确最终答案边界的响应拒绝。原始对话完整保存在离线 metadata，不自动复制成 `reasoning_text`。

## 3. ROI 与 interleaved 监督

`iter_tasks` 提供源题目给本地教师，`iter_candidates` 通过公共审核接口导入补充轨迹。标注键绑定源编号、问题、答案和图像哈希；修改任一内容后，旧标注不再匹配。缺少标注的源题目输出 `missing_region_annotation`，不会伪造仅包含 `finish` 的过程样本。

人工审核重点包括：读取窗口覆盖实际证据并保留必要标签与上下文；自然语言结论出现在对应视觉证据之后；完整过程足以支持源答案；教师没有从不可见内容臆测中间事实。原始数据中的简短答案可以监督最终输出，但不能作为区域或中间语言的证据。

审核接口接受 `proposed` 或带审核人身份的 `reviewed` 标注。两者经当前生产器都只生成 `pending`，不等同于批次验收。`visual_only_safe` 与 `visual_only_basis` 由审核记录明确声明；只有去掉语言仍能通过窗口序列完成任务的候选才生成纯视觉版本。公共生产器负责长边归一化、事件编号、引用检查、图像保存及格式校验。

## 4. 入口与有限验证

运行入口为 `scripts/sft_data/produce_llava.sh`，源目录、独立输出目录、图像目录、标注文件及样本上限均在脚本顶部设置。默认检查 10 条源记录；`LIMIT=0` 用于后续扫描全部已完成分片。源下载目录保持只读，输出不能放入其中。标注格式与本地教师调用沿用公共数据生产说明和工具。

`tests/test_sft_llava.py` 的六项测试覆盖选项答案、源身份、缺少区域监督、计数任务与实例监督边界、多轮依赖拒绝、混合推导答案拒绝、非训练来源、无图输入及标注内容绑定。验证范围为接口及语义边界，不表示全部子集已完成审核，也不表示教师轨迹可直接作为训练真值。
