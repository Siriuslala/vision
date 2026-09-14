# Visual-CoT 数据生产

Visual-CoT 提供问题、答案及支持回答的区域标注，适合构造一次局部读取的 SFT。转换器优先读取原始像素框 metadata，按作者的区域合并规则构造一次 `visual_read`，再保留来源已有的自然语言 `thought`，最后进入答案阶段。导出遵循 [数据规范 1.1](../visual_reasoning_data_format.md)，全部候选保持 `quality.status=pending`，等待图像证据与特征可辨性审核。

## 1. 来源与真实字段

来源为 [deepcs233/Visual-CoT](https://huggingface.co/datasets/deepcs233/Visual-CoT)。[官方转换代码](https://github.com/deepcs233/Visual-CoT/blob/main/tools/convert_data_to_llava_format.py)说明了 metadata 像素框、union 区域及训练对话之间的关系。

| 文件 | 内容与转换用途 |
| --- | --- |
| `metadata/*_cot_train.jsonl` | 每行一个问题，包括 `question`、`answer`、`image`、`width`、`height`、`bboxs`、`dataset`、`split`。`bboxs` 是原图像素 xyxy 框列表。部分来源还有结构化 `reasoning`。 |
| `cot_with_detailed_reasoning_steps/gqa_cot_train.jsonl` | GQA 的增强标注，额外提供 `full_answer`、`reasoning` 功能程序、自然语言 `thought`。 |
| `cot_with_detailed_reasoning_steps/gqa_cot_val.jsonl` | 来源保留的 validation；生产扫描只选 `*_train.jsonl`。行内非 train split 仍明确拒绝。 |
| `viscot_363k.json` | 处理过的多轮对话，框文字使用居中 square padding 后的长边归一化，且四舍五入到三位小数。首版转换使用精度更充分的原始 metadata。 |
| `viscot_mixed_2m.json` | 混合训练来源，包含 Visual-CoT 以外的数据；首版不扫描该文件。 |
| `cot_images_tar_split/cot_images_00` 至 `cot_images_12` | 同一未压缩 GNU tar 的连续分片。前 12 段每段 10,737,418,240 bytes，末段 10,403,440,640 bytes，成员位于 `cot_image_data/<dataset>/...`。 |

源码也支持官方 GitHub 的 `viscot_dataset/` metadata 目录，以及 `cot/<dataset>/...` 的旧图像布局。`textcap` 图像按作者代码映射至 `textvqa`。

首次检查下载目录时仅 README、assets 已完成，annotation 与图像分片尚未完成。试制额外读取官方文件列表、少量 GQA detailed 标注、SROIE/CUB 标注，以及 tar 首个 1 MiB，在独立临时目录提取两张真实 CUB 图像。未写入或重命名下载源。

## 2. 转换规则

1. **保留实际问题与答案。** 原始 `question` 直接作为 query；`answer` 作为 text 答案，不把 `full_answer` 或功能程序拼入问题。来源原始字符串仅移除首尾空白。
2. **检查原图像素框。** `width`、`height` 必须与图像解码尺寸一致，每个 xyxy 框必须有限、非退化且在原图内。多框取最小包围矩形，保留作者的一次读取策略；新窗口参照整图 0。common 根据原图长边转成中心与完整宽高。
3. **保持语言证据来源。** 完整 `thought` 放在读取事件之后。来源未标出句子与各个框的逐步对应，转换器不构造未经标注的逐句逐框轨迹。`reasoning` 中的 object ID 和功能程序留在离线 metadata；来源缺少 `thought` 时不添加模板化理由。
4. **区分模式。** 缺少 thought 的单窗口样本提供同几何的 interleaved 与 visual_only；数据规范允许 interleaved 无语言事件。具有 thought 的样本首版只提供 interleaved，待独立核查删除语言是否仍保留完整示范后再扩展纯视觉版本。
5. **保留划分与去重。** detailed 文件先读取，相同 dataset、image、question 的记录稳定生成相同 source ID，后续重复行写入拒绝记录。同图问题和模式由 common 按图像哈希共同分组。不同答案的重复问题也需人工核查，不能同时作为相互矛盾的监督。
6. **显式记录失败。** 缺图、坏 JSON、非法 split、未知来源、尺寸不匹配、越界框都写入 `rejections.jsonl`。例如实际 GQA 片段存在 500 × 375 图像上的 y1=376；首版拒绝此类标注，不静默裁剪。

任务类别按来源登记：CUB 为 recognition；DocVQA、DUDE、InfographicsVQA、SROIE 为 document_qa；TextVQA 为 ocr；TextCap 为 captioning；VSR 为 spatial_reasoning；其余已登记自然图像 QA 为 general_vqa。该来源级分类用于首版分层，细粒度任务再分类需要单独版本化规则，不能仅凭关键词赋予定位或计数监督。

## 3. 运行与解压

运行入口为 `scripts/sft_data/produce_visual_cot.sh`。在 `.env` 配置 `DATA_DIR`、`WORK_DIR`，在脚本顶部设置 `LIMIT`、`RUN_NAME`、`EXTRACT_IMAGES`、`EXTRACT_LIMIT`，然后运行：

```bash
bash scripts/sft_data/produce_visual_cot.sh
```

默认只检查 10 条来源记录，limit 包括失败行。`LIMIT=0` 才扫描全部完成的训练标注。输出目录必须是新目录；`candidates.jsonl`、`rejections.jsonl` 和 manifest 位于 WORK_DIR，内容寻址图像位于 DATA_DIR 下的 `sft_images`，记录里的图像路径相对 DATA_DIR。

首次完整处理前，将 `EXTRACT_IMAGES` 设置为 true、`EXTRACT_LIMIT` 设置为 0；解压器要求全部 13 段具有官方确切大小，直接流式连接，无需再生成一个完整 tar 副本。解压输出位于独立 IMAGE_CACHE；完成后关闭解压开关，再运行不同生产版本。已有外部解压目录可以直接在脚本中设置 IMAGE_CACHE。

前缀试提取允许设置正数 `EXTRACT_LIMIT` 并仅使用已完成的前几段。前缀图像与标注前 N 行不保证对应，缺图会完整计入拒绝报告。解压器要求新输出目录，拒绝绝对路径、上级路径、软硬链接和非图像成员；源文件始终只读。异常中止后保留已提取文件供审计，重新执行应换新目录。

## 4. 小样本验证与边界

本次以官方 CUB annotation 的第 1、2、5、6 行及匹配的两张真实图像执行 common 生产入口，4 条来源记录输出 4 条 interleaved 与 4 条 visual_only，未发生转换拒绝。两张图像分别为 500 × 336 和 331 × 380，覆盖非正方形图像。原始 CUB 没有 thought，因此该试制验证了图像、区域、答案和双模式导出，尚未验证自然语言推理质量。

4 项单元检查覆盖像素 union、thought 的事件顺序、功能程序不进入事件、越界拒绝、缺语言模式、行数限制、held-out split 和路径安全。GQA 的真实 detailed 片段已核对字段，但匹配图像未完成下载，未声称通过其完整图文轨迹审核。

区域重放合法性不能代替证据充分性。自然图像关系题可能需要 ROI 外上下文；模型初始整图仍可访问，但候选应抽检参照物、目标物是否可辨。thought 可能描述多个步骤或直接包含观察结论，需要核查原图支持程度。最终使用前还需检查当前 VLM 缓存特征能否分辨局部属性与文字。
