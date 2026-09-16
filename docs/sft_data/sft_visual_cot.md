# Visual-CoT 数据构造

Visual-CoT 提供问题、答案和支持回答的 bbox，部分 GQA 标注还包含自然语言 `thought`。它适合构造有明确区域依据的视觉推理数据，但多个框并不自动构成多步推理。转换时先恢复可靠的原图区域，再确定观察顺序，并检查语言是否与所观察的信息一致。

## 1. 文件结构与字段

本项目优先读取原始 metadata，其中 bbox 使用原图像素坐标。处理后的对话文件另有坐标变换和精度损失，不与原始框混用。

| 文件 | 内容及用途 |
| --- | --- |
| `metadata/*_cot_train.jsonl` | 每行一个问题，主要字段为 `question`、`answer`、`image`、`width`、`height`、`bboxs`、`dataset`、`split` |
| `cot_with_detailed_reasoning_steps/gqa_cot_train.jsonl` | GQA 的详细标注，额外包含 `full_answer`、功能程序 `reasoning` 和自然语言 `thought` |
| `cot_with_detailed_reasoning_steps/gqa_cot_val.jsonl` | 来源 validation，不进入当前训练转换 |
| `viscot_363k.json` | 已转换的多轮对话，坐标经过居中 square padding、归一化和三位小数取整 |
| `viscot_mixed_2m.json` | 包含其他来源的混合数据，当前不扫描 |
| `cot_images_tar_split/cot_images_00` 至 `cot_images_12` | 同一个 tar 的连续分片，图像成员位于 `cot_image_data/<dataset>/...` |

`bboxs` 是原图像素 `xyxy` 框列表。`width`、`height` 用于检查图像尺寸；`dataset` 与 `split` 用于来源分类和划分检查。`reasoning` 的功能程序包含对象编号和操作结构，与自然语言 `thought` 含义不同，不能直接拼成模型推理文本。

转换器也兼容官方代码仓库的 `viscot_dataset/` 标注目录及 `cot/<dataset>/...` 图片布局。`textcap` 图片按来源约定映射到 `textvqa`。来源和字段背景见 [Visual-CoT 数据集](https://huggingface.co/datasets/deepcs233/Visual-CoT)及[官方转换代码](https://github.com/deepcs233/Visual-CoT/blob/main/tools/convert_data_to_llava_format.py)。

## 2. 当前发现的问题

Visual-CoT 的局部框提供了候选证据范围，但还不足以直接确定完整 interleaved 轨迹。需要重点处理以下问题。

1. **部分来源没有自然语言过程。** 例如早期 CUB 样本只有问题、答案和框。窗口转换完成不等于已经得到完整的自然语言推理示范，后续需要教师补充有图像依据的观察结论。
2. **多个框没有可靠的事件顺序。** 将全部框取最小包围矩形可以恢复作者的一次读取策略；若要构造多次读取，需要结合题目和图像确定每次用途，不能把列表顺序当作推理顺序。
3. **部分标注越界。** 已读 GQA 片段出现 500 × 375 图片上的 y1=376。此类标注明确拒绝，不能静默压回图片边界。
4. **框内信息可能不完整。** 关系题可能需要框外参照物；文字题需要上下文。教师应检查目标与参照是否都足以辨认，而不只检查目标是否落在框内。
5. **重复问题可能具有冲突答案。** 同图同问题的重复条目需要去重；答案不同的条目保留冲突记录，不能同时作为监督。

## 3. 基础转换

基础转换保留 `question` 和 `answer`，检查每个框为有限、非退化且位于原图内部，并要求标注尺寸与解码尺寸一致。多个框取最小包围矩形，形成参照整图的一次读取。公共构造器将像素窗口转换成长边归一化的中心与完整宽高。

来源已有的 `thought` 放在读取之后，前提是相应图片证据能够支持其内容。功能程序和对象编号保存在离线 metadata；`full_answer` 不添加到问题中。没有 `thought` 的来源仍能生成几何候选，但需要明确区分这种结果与经过教师补全和检查的 interleaved 示范。

相同数据集、图片和问题生成稳定的来源编号。详细 GQA 标注优先读取，后续重复记录写入拒绝日志。同图不同问题、两种 reasoning mode 以及不同轨迹版本共享数据划分。

## 4. 多次观察与教师补充

多次观察优先用于需要比较目标、确认空间关系或依次读取不同文字区域的问题。教师利用原图、问题和可用 bbox 判断哪些区域分别提供了信息，再安排读取顺序，并在每次读取后说明具体发现。如果多个框实际描述同一物体的重叠范围，应合并或筛除重复观察。

已有 `thought` 可以辅助解释任务，但仍需逐步检查。缺少 `thought` 时，教师独立生成观察结论和最终回答，再与来源答案比较。原有单窗口版本和经过检查的多窗口版本可以同时保留，但两者分别记录构造方式，不能把自动拆框当作已经验证的多步轨迹。

任务类型目前按来源映射：CUB 为 `recognition`；DocVQA、DUDE、InfographicsVQA、SROIE 为 `document_qa`；TextVQA 为 `ocr`；TextCap 为 `captioning`；VSR 为 `spatial_reasoning`；其他已支持自然图像问答为 `general_vqa`。更细的分类需要结合具体问题进行检查。

## 5. 图像提取与运行

生产入口为 `scripts/sft_data/produce_visual_cot.sh`。在脚本顶部配置来源、图片缓存、输出目录、`LIMIT` 和提取参数。默认有限量读取，`LIMIT=0` 表示处理全部已完成的训练标注；失败记录同样计入检查数量。

完整图片提取按顺序读取全部 13 段 tar，避免额外生成完整 tar 副本。前 12 段各为 10,737,418,240 bytes，末段为 10,403,440,640 bytes。提取器检查文件大小，并将图片写入独立新目录。少量试提取可以只处理前缀，但得到的图片未必对应标注前几行，因此缺图需要如实报告。

提取器拒绝绝对路径、上级路径、链接和非图片成员；不写入来源目录。已经提取的图片目录可以直接配置复用。输出和可视化位置遵循[公共说明](sft_production.md)。

## 6. 验证范围

早期试制使用两张真实 CUB 图片及四个问题，生成两种 reasoning mode 各四条记录。图像尺寸分别为 500 × 336 和 331 × 380，验证了非方形图像坐标与双模式输出。由于这批来源没有 `thought`，它没有验证教师补充语言或多次观察的质量。

已有测试覆盖框合并、语言顺序、功能程序不进入轨迹、越界与划分拒绝、读取上限及路径处理。后续试制应覆盖多个 bbox 和有 `thought` 的问题，并报告真实图片检查结果。窗口合法性、教师语义判断与最终视觉特征可辨性分别检查。

本地教师的自动生成与检查使用 `scripts/sft_data/produce_automatic.sh`。本轮实际展示的例子、没有通过的情况及解释见[试制报告](sft_pilot_v2.md)。
