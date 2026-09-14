# CoSyn-400K 过程 SFT 数据生产

CoSyn 的图像、问题和答案可作为过程监督的基础，局部观察窗口及其顺序需要与实际图像证据逐项对齐。当前 adapter 提取源 QA，再导入本地视觉教师或人工编写的区域过程。原始 explanation 保存在离线 metadata；只有独立提供的观察与语言事件进入 trajectory。输出遵循 [数据格式 1.1](../visual_reasoning_data_format.md)，质量状态保持 pending，供人工审核与视觉特征检查。

## 1. 已下载数据与字段

本次检查发现 `CoSyn-400K/chart/` 已有部分完成的 Parquet shard；每个 shard 直接包含图像 bytes，无需解压或另取图像文件。其余 subset 尚未出现在本地。数据卡列有 chart、chemical、circuit、diagram、document、graphic、math、music、nutrition 和 table。已完成的 chart 足以开发和小样本验证，document/table 等领域的覆盖需要后续完成对应下载。

| 字段 | 实际结构及用途 |
| --- | --- |
| `id` | 合成图像标识，通常包含生成 pipeline、图形类型和编号 |
| `image` | `bytes` 与 `path`；直接读取 bytes，保留原图 |
| `qa_pairs` | 包含平行 `question`、`explanation`、`answer` 列表的 struct；一张图有多条 QA |
| `metadata` | `figure_type`、`persona`、`topic` |
| `data` | 图形生成所用底层数据，常为 CSV 字符串 |
| `code` | 图形生成代码，只读并记录摘要 |

抽读的 treemap 图像记录包含 10 条 QA，涉及标题、最大类别、金额比较、加总等问题。静态 PNG 的可回答性必须单独审核：该记录代码将数值写入 hovertemplate，部分 explanation 使用底层 CSV 的精确金额；交互提示的内容未必在静态图中显示。底层数据能够校验数值计算，但不能证明图像含有所需证据。

## 2. 提取与过程构造

`data/sft/datasets/cosyn.py` 提供 `iter_tasks` 和 `iter_candidates`。前者按稳定文件顺序展开 QA，后者通过共享 annotation 接口关联过程。`limit` 对检查的 QA 候选计数，拒绝项同样占用预算；`0` 表示遍历全部完成的 shard。源 QA 标识包含 subset、原图 id 和 QA 序号；同图问题共享 group_id，防止随机划分拆散同源图像。平行列表长度不一致、缺少图像或 QA 时明确返回拒绝原因。

chart、table、document 分别映射到 chart_qa、table_qa、document_qa；nutrition 暂映射到 document_qa，其他已登记 subset 使用 general_vqa。后续下载 subset 应先抽样复核该映射及其 schema，再规模化生产。

标注文件逐行记录 `source_key`、`status`、`events`、`reviewer` 和 visual_only 审核信息。source_key 绑定 QA 标识、问题、答案及图像内容，避免将旧标注应用于变动样本。事件中的 visual_read 使用原图像素 bbox 和 reference_region_id；共享构造器转换为长边归一化 center/size，并补充 finish。interleaved 的 reasoning_text 应放在支持该判断的观察之后；每次读取需保留标题、轴、图例、行列头等必要上下文。多区域比较按实际证据读取次序组织，语言中不引用不可见数据、代码执行或隐藏工具。

人工审核需核对答案可回答性、各步窗口覆盖、观察顺序、语言事实与任务完成条件。`visual_only_safe=true` 与非空 `visual_only_basis` 表示移除语言后的信息依赖已核查；通过后共享构造器输出 paired visual_only。自动教师结果只能作为 proposed 标注，正式训练前仍需审核。缺少标注时写入 rejections，保留源 QA 身份，不能补一个猜测窗口作为成功样本。

## 3. 运行与验证

运行入口为 `scripts/sft_data/produce_cosyn.sh`，在脚本顶部设置来源、独立输出目录、图像产物目录、标注路径及 QA 数量；从仓库根目录执行 `bash scripts/sft_data/produce_cosyn.sh`。环境沿用 README 的 uv 配置。原始下载目录只读，产物目录必须位于来源之外。脚本首次默认检查 10 个 QA，已有输出目录不能覆盖。

单元测试 `tests/test_sft_cosyn.py` 覆盖 QA 数量边界、错位列表拒绝、无标注拒绝、annotation 内容绑定、非方形图像坐标、两种模式生成，以及源 explanation 不进入 trajectory。实际数据验证仅抽取少量 QA；格式测试通过不代表源答案或教师轨迹已经通过人工语义审核。
