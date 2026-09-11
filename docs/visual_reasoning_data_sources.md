# 视觉推理训练数据：文献与公开资源核查

核查日期：2026-09-11。范围：原论文、作者仓库、官方数据卡与文件目录。已核查页面和公开样例，不代表完成了全量下载、逐条质量审计或本地可用条数统计。服务器直连 Hugging Face 未成功，发布情况以本次可读取的官方网页为据；不能把网页可见误写为数据已下载。

训练流程见 [训练方案](visual_reasoning_training_plan.md)，字段定义见 [数据规范](visual_reasoning_data_format.md)。本文分别记录论文训练规模、公开版本规模与清洗后的计划用量。不同集合之间大量重叠，以下数字不能直接相加。

## 1. 代表性工作的训练流程

| 工作 | 已核实的训练规模与流程 | 对本项目最直接的用途 |
| --- | --- | --- |
| [Qwen2.5-VL](https://arxiv.org/html/2502.13923)，§2.2–2.3 | 预训练三个阶段分别 1.5T、2T、0.6T tokens；SFT 约 200 万条，文本与多模态各半，之后 DPO。包含 caption、知识、OCR、grounding、文档、图表、VQA、视频与 agent。 | 使用预训练 VLM；采用按领域分类再过滤的流程。定位使用公开框/点和伪标注；图表程序合成，表格伪标注后过滤。其定位数据明确包含 PixMo。 |
| [Qwen3-VL](https://arxiv.org/html/2511.21631v2)，§3–4 | S0/S1/S2/S3 分别 67B、约 1T、约 1T、100B tokens。SFT 约 120 万条，约 1/3 纯文本；另有蒸馏与 RL。thinking-with-images 从约 1 万 grounding 冷启动扩展到约 12 万多轮交互。 | 多样性应按能力组织。定位结合 COCO、Objects365、OpenImages、RefCOCO 和伪标注；文档结合布局与区域识别；几何程序渲染。内部 OCR、文档、题库不能作为现成下载资源。 |
| [Molmo / PixMo](https://arxiv.org/html/2409.17146)，§3–4 | 描述预训练＋多任务微调；人工详细描述、人工问答和指点数据，辅以程序化合成。论文报告约 71.2 万张描述图、7.3 万张图上的 16.2 万问答、超过 230 万 grounding 标注。 | 天然图像的内容覆盖与位置监督有公开来源；不必对所有已有标注重新调用强教师。点标注适合大规模定位，但不自带连续观察顺序。 |
| [LLaVA-OneVision](https://huggingface.co/lmms-lab/llava-onevision-qwen2-7b-ov)，[训练说明](https://github.com/LLaVA-VL/LLaVA-NeXT/blob/main/docs/LLaVA_OneVision.md) | 模型卡列出 55.8 万对齐、470 万 mid-stage、360 万 final-image、160 万 onevision 混合训练数据；这些是阶段口径。 | 单图训练混合集适合做通用能力保持和任务补齐；按子集挑选，避免把多阶段用量当独立图像总数。 |
| [LVR](https://arxiv.org/html/2509.24251v2)，§3–4，[代码](https://github.com/VincentLeebang/lvr) | Qwen2.5-VL 3B/7B；视觉编码与投影冻结。VisualCoT SFT 用区域视觉嵌入作目标，MSE＋答案 CE；ViRL 上的 RL 只报告 3B 实验。 | 可以复用图像、QA 与 ROI 标签；原始 ROI patch 序列不是连续追踪轨迹。新增模块直接读取区域特征，可复用 ROI 监督，不要求重建视觉向量。 |
| [Thyme](https://arxiv.org/html/2508.11630)，§3–4，[代码](https://github.com/yfzhang114/Thyme) | SFT 冷启动＋工具交互 RL。候选来源包含 LLaVA-OV、MM-RLHF、SMR、V*、MM-Eureka、arXivQA、Retool；代码执行、模型核查和人工筛选。论文描述约 50 万 SFT 规模。 | 可借鉴少量恢复示范中的损失屏蔽；其 crop、rotation、contrast、computation 混合轨迹需按本方法支持的操作转换，语言可保留。 |
| [CoSyn](https://arxiv.org/html/2502.14846)，§4–5，[代码](https://github.com/allenai/pixmo-docs) | 20 条 pipeline、11 种渲染工具；主题→数据→代码→渲染与 QA。论文约 40 万图、270 万 instruction；类别包含图表、文档、数学、表格、流程图、矢量图、乐谱、电路、化学结构。 | 最适合借鉴的可扩展结构化制数范式之一。我们应额外保存几何与证据，并验证最终像素；原公开 QA 不自动具备连续过程标签。 |
| [DeepSeek-R1](https://arxiv.org/html/2501.12948v1)，§2 | R1-Zero 通过 RL 出现重新评估与反思；R1 使用包含冷启动 SFT 的多阶段流程。 | 支持以 SFT 提供初始能力、以 RL 优化自主探索；文中结果不等于本方法已具备视觉纠错能力。 |
| [DeepSeekMath](https://arxiv.org/html/2402.03300v3)，§4 | 提出 GRPO，以组内相对奖励计算优势，并采用策略约束，无需单独训练 critic。 | 本项目需将读取、参照和连续参数纳入随机策略；这种连续视觉动作扩展属于候选设计。 |

Qwen 两份报告给出的规模说明基础感知与知识来自长期、大量训练。我们的工作定位为已有 VLM 上的增量训练，不能承诺几十万条数据重建同等基础能力。本次在所查官方资源中未找到可直接下载的完整 Qwen2.5-VL / Qwen3-VL 训练混合语料；开源模型与微调代码不等于公开完整训练集。

两个有实际转换影响的细节：Qwen2.5-VL 报告的定位输出采用绝对像素坐标；Qwen3-VL 使用 0–1000 归一化坐标。Qwen3-VL 的“合成 function calling response”方案适合其训练目的，但本项目的读取结果应来自真实执行，不能用语言模型虚构的反馈充当视觉观测。

## 2. 已核实发布的候选数据

以下“可用”表示有可核查的公开入口，仍需按原始划分、图像有效性与监督质量处理。

| 数据及官方入口 | 发布规模／格式 | 能提供的监督 | 建议与限制 |
| --- | --- | --- | --- |
| [LLaVA-OneVision-Data](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Data) | 当前页面 **3,938,627 行**，约 303 GB；多个子集，图像＋conversations。部分 UReader 子集另以 JSON/图像压缩包发布。 | 通用问答、OCR、文档、图表、科学图形、视觉知识等。 | **优先通用池。** 只取符合单图约束的样本；核查子集、原始 train split 和重复图。页面汇总不等于独立图像数。 |
| [Visual-CoT 数据卡](https://huggingface.co/datasets/deepcs233/Visual-CoT/blob/main/README.md)、[文件目录](https://huggingface.co/datasets/deepcs233/Visual-CoT/tree/main) | 训练文件 `viscot_363k.json`；复现混合文件 `viscot_mixed_2m.json`；另有 metadata、详细推理、图像包，仓库约 143 GB。论文整体 **438k**，不能当成全部训练样本。 | 图像、问题、答案、关键区域；部分额外文字步骤。 | **优先 ROI 池。** 先用 363k 训练清单及原始 metadata；文字可作为交替模式的候选中间过程，须核查与区域和答案的一致性。 |
| [LVR 官方数据准备](https://github.com/VincentLeebang/lvr#dataset-preparation) | README 提供格式化数据下载入口与 JSON 样例；SFT 含 bbox，RL 为图像 QA。 | 与其实现匹配的 VisualCoT / ViRL 格式。 | 适合对齐 LVR 基线；不能作为独立的新数据量叠加。此次未逐个下载网盘文件或验证全部内容。 |
| [PixMo-Points 数据卡](https://huggingface.co/datasets/allenai/pixmo-points/blob/main/README.md) | `train` **2,376,222 行**；字段含 URL、SHA256、points、count、label、collection_method。标注下载约 198 MB，图像另取。 | 真实图像中的点、指代表达、计数。 | **优先定位/计数池。** 同一图有多行；下载须核对图像哈希。数据卡明确坐标为像素。点集不等于精确物体框。 |
| [PixMo-AskModelAnything](https://huggingface.co/datasets/allenai/pixmo-ask-model-anything)、[格式卡](https://huggingface.co/datasets/allenai/pixmo-ask-model-anything/blob/main/README.md) | 数据卡 `train` **161,737 行 QA**；论文约 73k 图。图像以 URL 和哈希提供，同图可有多问。 | 人工参与构造的自然图像问答。 | **优先通用池。** 有助减少单一视觉教师的偏好；不自带完整读取轨迹。 |
| [PixMo-Count](https://huggingface.co/datasets/allenai/pixmo-count/blob/main/README.md) | 数据卡 **36,916 train 行**，validation/test 各 540 行；图像 URL 可重复。 | 检测器标注后过滤得到的计数与指点；验证/测试由人工核查，但不提供点信息。 | 补充计数，独立保留评估划分；与 Points 去重。不要将训练伪标注当作人工精确真值。 |
| [CoSyn-400K](https://huggingface.co/datasets/allenai/CoSyn-400K) | 当前页面 **408,227 行**，约 67.8 GB；多类图像，每图多 QA；公开样例含 metadata、底层数据和生成代码。 | 文档、图表、表格、科学图形及程序结构。 | **优先结构化池。** 选定版本的 train；并非所有样本已提供证据区域。需逐类验证代码、渲染及问答一致性。 |
| [CoSyn-point](https://huggingface.co/datasets/allenai/CoSyn-point) | 当前页面 **69,051 行**，约 12.4 GB；图像、questions、answer_points、labels。 | 文档/表格/界面等的指点。 | **优先精细定位补充。** 坐标为 **0–100**；一行多个问题，不能与 QA 数混淆。 |
| [PixMo-Docs](https://huggingface.co/datasets/allenai/pixmo-docs/blob/main/README.md) | 四类 train：charts 116,814、diagrams 16,551、other 71,282、tables 46,518；合计 **251,165 训练图**。各有 1,024 validation。 | 合成图像及每图多 QA。 | 次选。数据卡已推荐更新的 CoSyn-400K / CoSyn-point；避免旧新数据重复。 |
| [Thyme-SFT](https://huggingface.co/datasets/Kwai-Keye/Thyme-SFT) | 当前页面 **345,973 行**，约 105 GB，3 个 split；可见 2round 约 152k 行；image 为列表，question/response 为字符串。 | 工具轨迹、多轮改进与错误恢复。 | **选择性转换。** 含原图/处理图、思考与代码；必须区分实际操作。不能按论文约 50 万认定全量可用，也不能把多个训练类别误当 train/validation/test。 |
| [Thyme-RL](https://huggingface.co/datasets/Kwai-Keye/Thyme-RL) | 当前 `train` 页面约 **55.2k 行**；messages、images、solution、question。 | 有答案的难问答。 | 作为困难 QA 候选池；页面样例不提供完整读取过程。来源含 V* 等再加工材料，须还原 ID 与评测做隔离，不能仅凭来源名判断污染或无污染。 |
| [ViRL39K](https://huggingface.co/datasets/TIGER-Lab/ViRL39K)、[官方仓库](https://github.com/TIGER-AI-Lab/VL-Rethinker) | 数据卡 **38,870 可验证 QA**，约 1.78 GB；question、answer、category、source、qid、image、pass rate。 | 可规则评分的视觉问答与难度参考。 | 可用于 SFT 后的 GRPO 任务筛选。无 ROI/轨迹。当前网页 viewer 有 Parquet footer 错误，发布入口及样例存在，整包可读性仍须落地核查；不是已完成下载。 |
| [The Cauldron](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron) | Idefics2 发布的 **50 个视觉语言训练数据集**合集；按 config 加载，每行 images＋texts。 | 通用 VQA、OCR/文档、图表、计数、逻辑和科学图形等。 | **按缺口补齐。** 官方说明采用原 train，并对部分基准去污染；还需针对本项目测试重查。与 LLaVA-OneVision 重叠，不能按两个独立池计数。 |

### 2.1 可直接补充的原始结构标注

- **COCO、Objects365、OpenImages、RefCOCO/+/g**：是 Qwen3-VL 报告明确列出的 grounding 来源，可用于物体位置和指代表达；首版可先从已选择集合中回接原始标注。这里未全量核对这些原始包的具体版本和独立条数，不将它们额外计入预算。[Qwen3-VL §3.2.4](https://arxiv.org/html/2511.21631v2)
- **TextOCR**：作者说明提供文字与多边形标注，图像 ID 与 TextVQA 对应；不能把不可读字符标记当作可识别文字。[官方数据说明](https://textvqa.org/textocr/dataset/)。本次官方页面直开正文不稳定，读取到官方检索说明，落地前要检查下载包；不虚报已验证样本数。
- **CLEVR**：原生成器公开，可提供场景和问题程序，用于空间关系与组合推理。新制数应在场景生成时额外记录所需证据位置；程序里的中间集合并不是模型的 hidden state。[官方生成器](https://github.com/facebookresearch/clevr-dataset-gen)
- **GQA/VSR、ChartQA、AI2D、DocVQA**：可从已核查的混合集选训练子集；需要过程监督时回到原始框、图结构、OCR 或底层表格，不能假定混合格式保留了全部标注。[The Cauldron 原始来源与统计](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron)

## 3. 过程监督的转换方法

### 3.1 VisualCoT 的区域监督

论文从多个来源构造图像、QA 与关键区域，包括已有框、OCR 定位和基于标注的问答生成。来源横跨文档/文字、细粒度识别、一般 VQA、图表和关系。它适合让我们的模型学会问题相关读取；对于连续追踪，必须另有跨步结构监督。[VisualCoT 论文](https://arxiv.org/html/2403.16999v3)

训练时优先采用原始可靠区域，保留标注来源。文本匹配得到的框、模型猜测的框和人工/程序精确框应分别记录；不要一律当等质量真值。

### 3.2 Thyme 的操作转换与恢复示范

该文发现只用单轮数据会使模型在错误 crop 后直接作答，于是构造扩大窗口后再精确定位、错误区域后重新定位等多轮数据。其训练对恢复样本屏蔽前缀，并屏蔽工具执行返回内容的预测损失。本项目可用少量此类示范提供基本恢复能力，并在 SFT 后通过 GRPO 优化自主策略。[Thyme §3](https://arxiv.org/html/2508.11630)

转换时保留与观测一致的规划和反思文本，构造交替模式示范；纯视觉版本需单独核查。旋转、增强及外部计算不能仅删除代码后直接转为本方法的读取过程。

候选数据经过执行和模型核查仍可能有错误；教师答案正确也不说明裁剪过程有必要。转换时应记录“操作兼容通过率”“坐标映射通过率”“学生特征可用率”，由试制结果决定最终采用规模。

### 3.3 CoSyn 的结构生成与程序渲染

公开 pipeline 可以产生多种视觉形式，公开数据能直接训练 QA；本项目需增加渲染时的几何记录及操作序列生成。只给 code-conditioned QA 改写更多问法，不会自动得到连续视觉依赖。[CoSyn 官方代码](https://github.com/allenai/pixmo-docs)

例如，相同的“读两处并比较”可以出现在自然物体、表格单元格、图表系列、相似图案中；相同的“沿连接继续”可以出现在曲线、线路图、流程图中。相同视觉操作在不同内容中的组合，可用于研究跨任务泛化。

## 4. 来源记录与转换统计

每个来源建立 manifest。下列为来源层面的记录项，不替代 [样本及 rollout 字段规范](visual_reasoning_data_format.md)：

```text
source_name, official_url, source_revision, source_config, source_split
original_image_id, image_sha256, original_sample_id
original_license, original_coordinate_convention
raw_count, downloaded_count, decoded_count, deduplicated_count, accepted_count
available_supervision, annotation_origin, conversion_version
split_group, exclusion_reason, teacher_version, review_status
```

`raw_count` 要标明是图像、QA、对话行还是训练 token。只下载被选中的配置和分片；先抽取小样本验证读取、图像和坐标，再扩大。

许可字段不能只复制页面顶端标签：例如 VisualCoT 页面的 Apache-2.0 标签与正文 Attribution-NonCommercial 4.0 表述不同；LLaVA-OneVision 同样有页面许可标签与学术用途说明。应保存原声明和来源引用，在确定数据用途时核对原始数据条款，不能把混合集标签当作所有原图的统一授权。

在所核查的公开资源中，尚未发现覆盖上述任务、且完整提供本方法逐步中心、范围与参照的现成训练集。通用视觉数据可复用；新增工作集中在操作序列标注、程序化连续任务与可验证的 RL 题目。SFT 保存完整示范，RL 记录自主执行过程，两者均支持按实际几何可视化。
