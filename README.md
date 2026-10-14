# BabyVision Local Inference and Data Processing

Current experiments focus on BabyVision. Set experiment parameters at the top of the corresponding Bash script, then run `bash scripts/xxx.sh` without command-line parameters. Each script includes its launch command in its opening comments.

## Environment and entry points

For initial setup, run this command from the repository root:

```bash
cp .env_example .env
```

Follow the comments in `.env_example` to configure:

- `ROOT_DIR`: the repository root containing `pyproject.toml`.
- `DATA_DIR`: the data root containing the `BabyVision` directory.
- `WORK_DIR`: the output root for generation results and other artifacts.
- `HF_HUB_CACHE`: an optional Hugging Face hub cache containing downloaded models and Parquet datasets. If omitted, the standard cache location is used.
- `REFCOCO_DIR`: an optional local RefCOCO dataset root used by the grounding analysis experiments.

Use quoted, literal absolute paths. Spaces are supported; variable references and shell commands are not part of the supported configuration format.

`.env` is private and ignored by Git. `.env_example` contains empty values and generic instructions and can be committed. Keep machine-specific paths in `.env` and experiment settings such as model, cases, batch size, and GPUs in the Bash scripts. Outputs go to `WORK_DIR`; placing it outside the repository keeps generated artifacts separate from source files. Runtime metadata may contain resolved local paths.

Bash entry points load configuration through `scripts/load_env.sh`. Direct Python entry points read the same file through `utils/project_paths.py`. Missing required paths produce an explicit error. The root `.python-version` file selects Python 3.12 for uv; `requires-python` in `pyproject.toml` declares the supported version range. Dependencies are managed with uv and pinned in `uv.lock`; scripts use `uv run --frozen`. Conda activation is unnecessary. Model weights are loaded offline from the local cache.

| Script | Purpose and configuration | Launch command from the repository root |
| --- | --- | --- |
| `scripts/env/setup_env.sh` | Install the locked environment | `bash scripts/env/setup_env.sh` |
| `scripts/eva/simple_test.sh` | Configure one model and paired image/prompt lists | `bash scripts/eva/simple_test.sh` |
| `scripts/eva/evaluation.sh` | Configure full-benchmark or subset evaluation | `bash scripts/eva/evaluation.sh` |
| `scripts/eva/smoke_models.sh` | Run paired image/prompt lists on a model array sequentially | `bash scripts/eva/smoke_models.sh` |
| `scripts/data/prepare_data.sh` | Configure the dataset source and export destination | `bash scripts/data/prepare_data.sh` |
| `scripts/data/download.sh` | 配置 Hugging Face 数据集名称、保存路径及下载参数 | `bash scripts/data/download.sh` |
| `scripts/data/validate_data.sh` | Configure the dataset and directory to validate | `bash scripts/data/validate_data.sh` |
| `scripts/testing/check.sh` | Check Bash syntax, unit tests, and Python CLIs | `bash scripts/testing/check.sh` |

Bash 入口按用途组织：`scripts/data/` 为数据下载、导出与验证，`scripts/eva/` 为推理、评测、诊断和结果查看，`scripts/env/` 为环境安装和内核检查，`scripts/testing/` 为项目测试，`scripts/sft_data/` 为 SFT 数据流程。`scripts/` 根目录仅保留共享的 `load_env.sh` 和 `decoding_presets.sh`。

测试按分析研究、推理评测、SFT 数据和基础工具分类，具体范围见 [测试目录](tests/README.md)。运行 `bash scripts/testing/check.sh` 可递归执行全部类别的测试，并检查 Bash 语法与推理 CLI。

For example, from the repository root:

```bash
bash scripts/eva/simple_test.sh
```

## Configuring experiments in Bash

视觉表征研究按研究问题组织，入口为[总研究计划](docs/analysis/plan.md)和[文档目录](docs/analysis/README.md)。当前研究 Qwen3.5-9B 的行为诊断、视觉细节表征、位置信息与坐标读取、多层视觉特征注入（与 Qwen3-VL 对照）以及局部重读；token 词汇读出已终止。完整模型实验固定 thinking off。新产物保存于 `WORK_DIR/analysis/<研究问题>/<日期_实验名称>/`，历史运行来源见[实验产物说明](docs/analysis/experiments.md)。

分析代码与 Bash 入口按研究问题分目录：`analysis_tools/<研究问题>/` 存放实验程序，`analysis_tools/common/` 存放跨问题共用的刺激生成与模型工具，`scripts/analysis/<研究问题>/` 存放对应入口，测试位于 `tests/analysis/<研究问题>/`。研究问题目录包括 behavior、representation、position、deepstack、reread 与 token_semantics。

| 入口 | 实验内容 |
| --- | --- |
| `bash scripts/analysis/representation/representation_patterns.sh` | 规则与扩展布局的表征差异，以及高范数 token 直接排除 |
| `bash scripts/analysis/representation/representation_cross_model.sh` | Qwen3-VL-8B-Instruct 的空间模式对照 |
| `bash scripts/analysis/representation/representation_pattern_figures.sh` | 从保存结果重绘布局与范数图 |
| `bash scripts/analysis/position/position_real_probe.sh` | RefCOCO 真实图像的 ViT／decoder 逐层位置 probe |
| `bash scripts/analysis/token_semantics/token_lens_layers.sh` | decoder 末尾连续层的 top-1/2/3 词图 |
| `bash scripts/analysis/behavior/behavior_figures.sh` | 已有 grounding 回答的输入、真值与预测图示 |
| `bash scripts/analysis/representation/distance_register.sh` | 2026-09-30 多图案距离与高范数实验 |
| `bash scripts/analysis/position/position_probe.sh` | 2026-09-30 合成图位置 probe 与前缀对照 |
| `bash scripts/analysis/token_semantics/token_lens.sh` | 2026-09-30 五个阶段的 raw logit lens |
| `bash scripts/analysis/behavior/comparison_grounding.sh` | 目标选择条件与答案格式对照 |

在各脚本顶部设置参数，在私有 `.env` 配置 `REFCOCO_DIR` 等本地路径。中文 token 图另使用 `TOKEN_MAP_FONT` 指向支持 CJK 的字体文件。科学曲线绘图使用由 `pyproject.toml` 与 `uv.lock` 管理的 matplotlib。早期细节实验和 point 实验分别保留 `visual_detail.sh`、`grounding_pilot.sh` 入口，结果在所属问题目录中按日期记录。

Edit the configuration section in `scripts/eva/simple_test.sh`:

```bash
MODEL="Qwen3.5-4B"
IMAGE_PATHS=(
  "$DATA_DIR/BabyVision/train/image/445.jpg"
  "$DATA_DIR/BabyVision/train/image/462.jpg"
)
PROMPTS=(
  "Describe the unusual tiger and give its row and column."
  "Which option matches the top pattern? Explain your choice."
)
OUTPUT_FILE="$WORK_DIR/qwen35_4b_images_4096.jsonl"
CACHE_ROOT="$HF_HUB_CACHE"
BATCHSIZE=2
GPU_IDS="3"
MAX_NEW_TOKENS=4096
THINKING="off"
MAX_PIXELS=""
DTYPE="bfloat16"
ATTN_IMPLEMENTATION="sdpa"
RANDOM_SEED=0
RESUME=false
DRY_RUN=false
```

Save the file and launch the script. No terminal parameters are needed; edit the script to record the intended experiment configuration. GPU 3 is an initial setting, so choose GPUs according to availability.

Supported models are `Qwen3.5-4B`, `Qwen3.5-9B`, `Qwen3-VL-4B-Instruct`, and `Qwen3-VL-8B-Instruct`. Names such as `Qwen/<name>`, `models--Qwen--<name>`, and absolute paths to local model directories or snapshots are also accepted. Resolution first checks `CACHE_ROOT`, then the default Hugging Face cache. It does not download weights.

`IMAGE_PATHS` and `PROMPTS` are nonempty Bash arrays of equal length. The image at index i is paired with the prompt at index i; one pair is one sample. Each array element must be quoted so that spaces and line breaks remain part of the same value. A single pair runs one sample, and the same image can be repeated with different prompts. Local images can come from any dataset or directory. These inputs do not require a benchmark manifest.

Simple test passes each prompt unchanged into the model's chat template. It does not append answer-format instructions or read reference answers. Image files are checked for readability before model loading. Empty prompts, unequal list lengths, missing images, and corrupt images produce explicit errors.

`BATCHSIZE` controls the number of image/prompt pairs in each generation call. For three pairs with `BATCHSIZE=2`, the batches contain two and one samples. Output order follows input order. `GPU_IDS="0,3"` enables automatic placement of one model across the specified GPUs.

`evaluation.sh` selects benchmark samples with `BENCH`, `SPLIT`, and `CASE_IDS`. An empty `CASE_IDS=()` selects the complete split; `(445 462)` selects those original task IDs. This entry point constructs benchmark prompts and computes scores.

`MAX_PIXELS=""` preserves the model processor's image-resolution settings. Setting a numeric limit changes the visual input budget; keep it consistent when comparing models. `DRY_RUN=true` validates cases and prompts without loading weights. Use a separate output file for dry runs.

## 解码配置

`scripts/eva/evaluation.sh`、`scripts/eva/simple_test.sh` 和 `scripts/eva/smoke_models.sh` 通过 `scripts/decoding_presets.sh` 根据 `MODEL` 和 `THINKING` 自动选择采样参数。Qwen3.5 的 `on` 使用 thinking general 配置，`off` 使用 non-thinking reasoning 配置；`default` 按模型默认解析为 `on`。Qwen3-VL Instruct 使用 `off`，Thinking checkpoint 使用 `on`，两者也接受 `default` 自动匹配权重类型；冲突配置会报错。

| 模型及模式 | temperature | top-p | top-k | min-p | repetition penalty |
| --- | --- | --- | --- | --- | --- |
| Qwen3.8-27B Thinking | 1.0 | 0.95 | 20 | 0.0 | 1.0 |
| Qwen3.8-27B Non-thinking | 0.7 | 0.80 | 20 | 0.0 | 1.0 |
| Qwen3.5 Thinking | 1.0 | 0.95 | 20 | 0.0 | 1.0 |
| Qwen3.5 Non-thinking reasoning | 1.0 | 1.0 | 40 | 0.0 | 1.0 |
| Qwen3-VL Instruct，VL 配置 | 0.7 | 0.8 | 20 | 0.0 | 1.0 |
| Qwen3-VL Thinking，VL 配置 | 1.0 | 0.95 | 20 | 0.0 | 1.0 |

上述预设均启用 sampling，覆盖当前支持的 4B/8B Qwen3-VL、4B/9B Qwen3.5 和 Qwen3.8-27B。Qwen3.8-27B 的 `THINKING=default` 解析为 `on`；其 thinking 配置不施加 presence penalty，non-thinking 推荐的 presence penalty=1.5 因接口限制省略。生成长度仍由 Bash 顶部的 `MAX_NEW_TOKENS` 独立控制。Qwen3-VL 截图未指定 min-p，项目显式禁用该筛选。`presence_penalty` 因 Transformers 原生生成接口不支持而省略，不用重复惩罚替代。Python CLI 保留手动解码参数，未指定 `--do-sample` 时使用 greedy。

所有解码参数及实际生成配置写入 `.meta.json`，每条预测记录解码模式、实验 seed 和 batch seed。采样按原始 batch 编号派生独立 seed；恢复时保留原始 batch 划分，部分写入的 batch 会重新生成完整 batch，仅追加缺失记录。相同环境、输入和 batch size 下，此策略避免恢复过程改变采样随机数序列；它不保证不同硬件或计算后端间的逐位一致。参数或 batch size 变更需要新的输出文件；缺少新增解码配置的旧 metadata 无法直接恢复。

实验协议、参数含义和 Qwen3.5 官方配置的适用范围见 [评测解码协议](docs/evaluation_decoding.md)。

## Generation length

Earlier tests with a 512-token limit sometimes stopped before producing a final answer. The three inference Bash scripts and the Python CLI therefore default to **4096 new tokens**. This is an initial budget informed by those observations, not a verified guarantee that every case will finish.

- With the default `THINKING="off"`, start with 4096 and inspect truncation on representative cases or the full benchmark.
- For Qwen3.5 with `THINKING="on"`, an initial budget of 8192 can be evaluated and adjusted based on observed truncation.
- `MAX_NEW_TOKENS` is an upper limit, not a required output length. Generation ends early when the model emits EOS.
- If many cases still have `finish_reason="length"`, inspect their responses and consider a larger consistent budget. Long output can also reflect repeated reasoning or loops.
- Keep generation and image budgets consistent across model comparisons. Report truncation counts and do not treat the earlier 128/512-token interface checks as benchmark performance estimates.

`THINKING` affects the two supported Qwen3.5 models. The Qwen3-VL Instruct checkpoints do not implement this switch. Qwen3.5 uses the installed `flash-linear-attention` kernels for Gated DeltaNet. The separate causal convolution kernels require a compatible `causal-conv1d` wheel; without it, convolution uses the PyTorch reference implementation.

## Simple-test outputs and resuming

Simple test writes one JSONL record per image/prompt pair with `sample_id` (zero-based input index), `image_path`, the exact `prompt`, full `response`, model name, generated token count, stop reason, and batch timing. Its summary reports sample count, truncations, and generated tokens. No accuracy is computed because these inputs have no reference-answer labels.

The companion metadata file records model and generation settings, ordered image/prompt pairs, and SHA256 hashes of image contents. With `RESUME=true`, settings and inputs must match the original run. Changing a prompt, the input order, or image bytes at the same path requires a new output file. Completed records are skipped by `sample_id`; interrupted samples without a saved record are regenerated from the start.

## Full evaluation, outputs, and resuming

在 `scripts/eva/evaluation.sh` 顶部设置 `MODEL`、`GPU_IDS`、`OUTPUT_DIR`、`OUTPUT_PREFIX` 和生成参数。输出文件名由前缀及 `thinking`、`random_seed`、`do_sample`、`temperature`、`top_p`、`top_k`、`min_p`、`repetition_penalty` 自动拼接，格式为 `<prefix>-thinking<value>-random_seed<value>-...jsonl`。模型前缀根据模型名称和 benchmark 自动生成；其他影响实验的配置仍完整保存在 metadata 中。`ATTN_IMPLEMENTATION="flash_attention_2"` 启用 FlashAttention 2。保留 `CASE_IDS=()` 选择完整 benchmark，然后运行：

```bash
bash scripts/eva/evaluation.sh
```

After an interruption, change `RESUME=false` to `RESUME=true` in the same script, keep the other settings unchanged, and run the same command. Model paths, parameters, selected cases, and the manifest hash must match. If you change the generation budget, choose a new output file; results from different configurations are not merged during resumption.

Each run produces:

- `*.jsonl`: task ID, categories, prompt, full response, prediction and reference answer, generated token count, stop reason, and batch timing.
- `*.jsonl.meta.json`: configuration, model snapshot, manifest hash, dependency versions, device mapping, and processor configuration.
- `*.jsonl.summary.json`: accuracy, unparsed answers, truncations, and breakdowns by category, subtype, and answer type.

生成开始时输出 batch size 和 padding 后的输入 token 数。生成过程中每 15 秒输出一次状态：首 token 返回前显示等待状态，之后显示已完成的 decode 步数和平均步速。计时信息保存在每条预测的 `generation_metrics` 中；batch size 大于 1 时，步数不等于所有样本的 token 总数。首 token 延迟包含 prefill 和可能的首次 kernel 编译开销。加载完成后同时打印各设备上的参数数量，用于识别 CPU 放置。

排查速度时，先在 `scripts/eva/diagnose_inference.sh` 顶部选择空闲 GPU 和短生成预算，再运行 `bash scripts/eva/diagnose_inference.sh`。诊断使用两条 BabyVision 样本，日志及结果写入 `WORK_DIR/diagnostics/` 的独立目录。短生成结果只用于性能诊断，不能作为 benchmark 分数。

Each record is flushed after writing; summaries are updated after each batch. Existing output files are protected from accidental overwrites. Malformed JSONL produces an error rather than being silently skipped. `batch_seconds` measures an entire batch and should not be summed once per case within that batch.

Model inputs contain only the image, question, and options. Reference answers and dataset CoT annotations are excluded. Local strict scoring preferentially extracts `Final answer: ...`. Multiple-choice predictions must be valid option letters. Blank answers are normalized only for Unicode, whitespace, case, and trailing periods; unparsed answers count as incorrect. Semantically equivalent text can fail strict matching, so inspect raw responses. This differs from the [official LLM-judge evaluation](https://github.com/UniPat-AI/BabyVision/tree/main/babyvision_eval).

## 推理结果查看网页

编辑 `scripts/eva/viewer.sh` 顶部参数后运行：

```bash
bash scripts/eva/viewer.sh
```

默认地址为 `http://127.0.0.1:1044`。服务器部署时可使用 SSH 端口转发：在本地终端执行 `ssh -L 1044:127.0.0.1:1044 <server>`，然后在本地浏览器打开上述地址。

- `JSONL_FILES`：结果文件列表，默认选择 `WORK_DIR/eval/*.jsonl`。每行需要 `taskId` 或 `sample_id`、字符串 `response`，以及 `image` 或 `image_path`。
图像直接读取结果行的 `image_path`，兼容 `image` 字段，无需配置图像根目录。绝对路径直接使用，相对路径以对应 JSONL 所在目录为基准；仅在图像字段来自可选 manifest 时，相对路径以 manifest 所在目录为基准。网页通过运行和样本索引访问已配置记录的图像，浏览器不能指定任意文件路径。
- `MANIFEST`：默认空字符串。结果 JSONL 已包含问题、答案和参考推理时无需设置；可选提供 BabyVision `train.jsonl`，仅按 `taskId` 补充结果文件中缺失的参考字段，不覆盖已有内容。
- `HOST` / `PORT`：监听地址和端口。

问题优先读取结果行的 `question`，否则回退到 `prompt`；参考答案支持 `reference`、`reference_answer`、`ground_truth`、`answer`、`blankAns`、`choiceAns` 和 `score.reference`。参考推理支持 `reference_reasoning`、`coT`、`cot`、`CoT`、`rationale`、`solution`；`reasoning`／`reasoning_content` 按模型推理展示在右侧。缺失参考推理时隐藏对应区块，缺失答案时显示未提供。

左侧展示图像、问题、参考答案和可展开的参考推理；右侧展示模型输出、独立保存的模型 reasoning、评分和停止原因。桌面端两列独立滚动，图像支持放大。网页支持运行切换、样本 ID／问题搜索、正确／错误／未评分／截断筛选、原文查看和复制。

模型输出通过本地 Markdown 渲染，保留普通换行，支持标题、列表、表格、引用和代码块；原始 HTML 按文本显示。`Plain text` 可查看原文，`Decode literal \n` 用于显示双重转义的换行，默认关闭以保留原始反斜杠含义。不自动补全未保存的 reasoning，也不将参考推理标记为模型推理。数学 LaTeX 命令保留为文本，未启用数学排版引擎。

`Reload files` 重新读取已配置的文件，适合查看运行中已写入的结果；增加新的文件路径需要修改 Bash 配置并重启。损坏 JSONL、重复 ID、丢失图像均明确报错。结果源文件保持只读，网页静态资源在 `eva/viewer/static/`，不依赖 CDN。

## Hugging Face 数据集下载

编辑 `scripts/data/download.sh` 顶部的数据集名称和保存目录，然后运行：

```bash
bash scripts/data/download.sh
```

大型共享数据集使用 `.env` 中的 `DATA_DIR_PUBLIC`。下载文件与续传状态均保存在 `LOCAL_DIR`，不在 home、`/tmp` 或共享模型缓存中额外保存数据副本。环境沿用 `uv run --frozen`。归档文件系统的透明压缩会影响 `du -sh`；程序显示逻辑字节数。

| 参数 | 含义 |
| --- | --- |
| `REPO_ID`、`LOCAL_DIR` | 数据集 ID 与绝对保存路径 |
| `REVISION` | 首次解析为固定 commit；再次运行继续该 commit，不自动追随 `main` 更新 |
| `MAX_WORKERS=2` | 同时下载的文件数；多个终端的并发数会累加 |
| `RANGE_MIB=64` | 每次 HTTP Range 请求的大小；每次都从 Hub resolver 获取地址 |
| `DOWNLOAD_TIMEOUT=120` | 网络读取无响应的超时秒数，不是整个文件的下载时间上限 |
| `ETAG_TIMEOUT=60` | 首次获取仓库清单的超时秒数 |
| `ATTEMPTS=10` | 单文件连续失败上限；一次 Range 请求成功后计数归零 |
| `ENDPOINT=""` | 继承 `HF_ENDPOINT`；可设置官方 `https://huggingface.co` 经代理下载 |
| `IMPORT_HTTP_PARTIALS=true` | 迁入已停止的旧 HTTP 下载器留下的临时文件；Xet 或来源不明的临时文件必须设为 `false` |
| `DRY_RUN=false` | 设为 `true` 时固定清单、校验已有文件并按配置迁入旧临时文件，不传输数据文件内容 |

Bash 入口加载私有 `.env` 并导出 `HF_ENDPOINT`、`http_proxy`、`https_proxy`。沿用服务器提供的镜像、代理和包缓存配置。镜像限流或持续返回过期链接时，可将下载脚本的 `ENDPOINT` 设置为官方地址；代理仍从 `.env` 继承。下载采用 HTTP 顺序写入，不依赖 `hf-transfer` 或 Xet 分块缓存。

网络中断、CDN 签名过期和重试耗尽均保留 `.download-state/partials/` 下的连续字节前缀。再次运行时从其长度发起 Range 请求。服务器忽略 Range、返回错误范围、权限不足或校验失败时明确报错，已有字节保留待检查。首次迁移已有完整文件会读取磁盘核验 SHA256（LFS 文件）或 Git blob SHA1（普通文件）；以后文件大小和修改时间未变时复用校验记录。新文件全部下载并校验成功后才出现在正式路径，可在其余分片下载期间读取。

首次迁移前停止旧下载进程。`IMPORT_HTTP_PARTIALS=true` 按远端内容 hash 匹配旧 `.cache/huggingface/download/` 临时文件，重命名最长匹配文件以避免复制大文件；其他残留不自动删除。已被旧库删除的字节无法恢复。进度条包含完整文件和可恢复的临时文件，启动时分别报告已校验字节、未校验字节与固定总量。

同一目录只允许一个本下载器运行；不同目录可并行下载。Ctrl+C 或 SIGTERM 请求停止，等待当前网络读取返回或超时后退出；SIGKILL 无法执行最后一次磁盘同步。移动数据集时包含整个 `.download-state`，保持仓库 ID 和 `REVISION` 一致；切换仓库或版本使用新目录。不要将新旧下载器同时用于同一目录。

PixMo-Count 的 Parquet 包含图片 URL 和标注，图片需另行下载；该入口保存仓库原始文件，不自动解压或下载标注中的外部图片。

## BabyVision data

The local dataset has been exported to `${DATA_DIR}/BabyVision`: 388 cases and 388 images, including 135 multiple-choice and 253 blank-answer cases. The official split is named `train`; this project uses it as an external test benchmark.

```text
${DATA_DIR}/BabyVision/
├── train.jsonl
├── train/image/<taskId>.jpg  # The extension follows the original image format.
├── dataset_info.json
└── SOURCE_README.md
```

Parquet image fields use `struct<bytes: binary, path: string>`. Export preserves the original encoded bytes without resizing or recompression. JSONL retains the original non-image fields, replaces image fields with relative paths, and adds `split`, `source_shard`, and `image_metadata` containing dimensions, format, original path, and SHA256.

Original fields are `taskId:int32`, `status/type/subtype/question/ansType/blankAns/coT:string`, `options:list<string>`, `choiceAns:int32`, and `image:struct`. `choiceAns` is zero-based and maps to A, B, C, and so on; some answer fields can be null. Task IDs are noncontiguous and range from 362 to 8343. See `dataset_info.json` for the complete ID list and category statistics.

The existing local export does not need to be recreated. To export again, configure a new `OUTPUT_ROOT` in `prepare_data.sh`; existing destination directories are not overwritten. Both preparation and validation scripts default to BabyVision only. Previously exported BabyVision-Gen data remains available, but it is outside the current inference and evaluation focus.

## Code organization and previous validation

`utils/project_paths.py` reads and validates private path configuration. `data/` handles export and validation. `eva/inputs.py` validates image/prompt pairs. `eva/benchmark.py` implements benchmark case selection, prompt construction, and scoring. `eva/inference.py` shares offline model loading and batched generation. `eva/simple_test.py` and `eva/evaluation.py` are Python entry points; use the scripts in `scripts/` for routine execution.

Previous local validation loaded all four models and performed GPU inference with 128/512-token limits. The newer 4096-token configuration has not yet been used for a new model run or full evaluation. Earlier output records retain their original settings.

## Prebuilt FlashAttention installation

This project pins a prebuilt FlashAttention 2.8.3 wheel from the administrator-approved
[mjun0812 release v0.10.0](https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/tag/v0.10.0).
It targets CPython 3.12, Linux x86_64, CUDA 13.0, and PyTorch 2.14.
The official releases inspected did not contain a matching PyTorch 2.14 wheel.

The direct wheel URL is recorded in `[tool.uv.sources]`; `uv.lock` records its hash.
Python is restricted to 3.12 and PyTorch/torchvision are pinned to 2.14.0/0.29.0 to
keep this binary dependency aligned with the runtime. If changing these versions
or moving to another platform, select a compatible prebuilt wheel and update the
lockfile rather than reusing this binary.

`no-build-package = ["flash-attn"]` prevents uv from compiling FlashAttention.
The installation script additionally uses `--no-build` to prohibit source builds
for every package during that operation. Do not replace the wheel source with a
source distribution or enable an automatic compilation fallback on the shared server.

```bash
bash scripts/env/install_flash_attention.sh
```

The script synchronizes the locked environment and reports the package and native
extension paths. Dependencies are installed inside `${ROOT_DIR}/.venv`, normally
under `.venv/lib/python3.12/site-packages/`. The uv cache is a separate download and
installation cache; use `uv cache dir` to locate it. `DATA_DIR` and `WORK_DIR` do not
control where Python dependencies are installed.

To verify GPU execution, set `GPU_ID` at the top of the check script and run:

```bash
bash scripts/env/check_flash_attention.sh
```

The check compares FP16/BF16 forward and backward attention with PyTorch's math
SDPA implementation and also checks packed sequences with unequal lengths. It
requires GPU access and uses only small tensors; it does not load a VLM.

After verification, set `ATTN_IMPLEMENTATION="flash_attention_2"` in an inference
Bash script to request this backend. The existing `sdpa` setting remains available.
FlashAttention does not install the separate `flash-linear-attention` or
`causal_conv1d` kernels used by Qwen3.5's linear-attention layers.

Installation was verified on 2026-09-10 with Python 3.12.3, PyTorch 2.14.0+cu130,
and an NVIDIA A800. Package and CUDA-extension imports passed. FP16/BF16 causal
and noncausal forward/backward checks and packed variable-length forward checks
passed against math SDPA. These are kernel correctness checks, not a VLM speed
benchmark or a full BabyVision evaluation.

## Qwen3.5 线性注意力内核

项目依赖固定 `flash-linear-attention[cuda]==0.5.2`，并通过 `uv.lock` 锁定 `fla-core`。这些 Python wheel 提供基于 Triton 的 Gated DeltaNet 内核，支持 prefill 和逐 token decode。环境安装由 `uv sync --frozen --no-build` 完成，日常安装和检查入口为：

```bash
bash scripts/env/install_linear_attention.sh
bash scripts/env/check_linear_attention.sh
```

在检查脚本顶部配置 `GPU_ID`。检查使用小张量，验证 Transformers 确实调用 FLA，以及 FP16/BF16 的输出和最终 recurrent state 与 PyTorch 参考结果一致，不加载完整模型。

包位于项目 `.venv/lib/python3.12/site-packages/`。FLA 首次运行会由 Triton 编译当前形状所需的 GPU kernel，Bash 入口将缓存默认放在 `WORK_DIR/cache/triton`，可通过私有 `.env` 的 `TRITON_CACHE_DIR` 覆盖。首次运行的编译属于 Triton 的正常 JIT 流程，安装过程不编译 CUDA 扩展源码。

`causal-conv1d` 与 FLA 是独立依赖。核对于 2026-09-15，官方 v1.7.0 release 未提供当前 PyTorch 2.14 / Python 3.12 / CUDA 13 环境对应的 wheel，因此本环境尚未安装该包，causal convolution 仍使用参考实现。项目显式禁止 uv 从源码构建 `causal-conv1d`，后续安装需提供经过验证的兼容 wheel。安装后需重新启动推理进程，已运行进程不会自动切换内核。

参考：[FLA 安装说明](https://github.com/fla-org/flash-linear-attention/blob/main/INSTALL.md)、[causal-conv1d 官方 release](https://github.com/Dao-AILab/causal-conv1d/releases/tag/v1.7.0)。

## 一般视觉推理 SFT 数据生产

SFT 生产覆盖 Visual-CoT、Thyme-SFT、PixMo-Count、CoSyn-400K 和 LLaVA-OneVision。方法、字段和小样本结果见 [生产说明](docs/sft_data/sft_production.md)。自动试制从 `produce_automatic.sh` 开始，人工抽检放在教师检查之后。先编辑脚本顶部参数，再运行对应 Bash 入口；默认仅处理少量样本，`LIMIT=0` 在生产 adapter 中表示全部已完成来源记录。本地教师入口要求正的显式数量上限。

共享下载源可在私有 `.env` 中设置 `DATA_DIR_PUBLIC`；未设置时使用 `DATA_DIR`。来源文件保持只读，提取出的原图和裁剪预览写到独立目录。常规转换的候选与日志写入 `WORK_DIR`，标准图片写入 `DATA_DIR/sft_images`；供 viewer 检查的新试制结果统一写到共享目录 `tmp` 下的独立子目录。所有来源应共用 `split_registry.sqlite`，检查同一图像是否被分配到不同的数据划分。

| 入口 | 用途 |
| --- | --- |
| `bash scripts/sft_data/produce_automatic.sh` | 使用 Qwen3.5 自动作答、选择观察窗口、生成过程和检查实际裁剪图；按读取次数及正常／纠错过程分存 |
| `bash scripts/sft_data/automatic_counting.sh` | 使用 SAM 3 候选与 Qwen3.5 自动核对实例、完整覆盖和分组计数过程 |
| `bash scripts/sft_data/produce_visual_cot.sh` | 从原始 metadata 与原图构造 ROI 过程；可选限量提取已完成 tar 分片 |
| `bash scripts/sft_data/produce_thyme.sh` | 恢复来源裁剪轨迹，核对工具返回图片与原图窗口 |
| `bash scripts/sft_data/produce_pixmo.sh` | `prepare` 下载／核验原图及建立 point 审核记录；`convert` 导入已审核实例框生成分组计数 |
| `bash scripts/sft_data/propose_point_windows.sh` | 规则几何或本地 SAM 3 生成待审核 bbox 候选 |
| `bash scripts/sft_data/propose_sft_regions.sh` | 本地缓存 VLM 为 CoSyn 或 LLaVA 提议 ROI 和 interleaved 过程 |
| `bash scripts/sft_data/produce_cosyn.sh` | 导入 CoSyn 的人工／教师标注，转换为标准候选 |
| `bash scripts/sft_data/produce_llava.sh` | 导入 LLaVA 单轮任务的人工／教师标注，转换为标准候选 |
| `bash scripts/sft_data/review_sft.sh` | 少量候选图文审核，或导出显式验收通过的记录 |

每次生产使用新的输出目录，包含 `candidates.jsonl`、`rejections.jsonl` 和 `manifest.json`。候选默认 `pending`，正式训练只接受经过任务、特征和批次审核的 `validated` 数据。SAM 3 路线需本地兼容的 tracker checkpoint；PixMo URL 图片必须通过来源哈希核验。详细审核字段及可执行范围见各数据集说明。

### 交互数据审核网页

编辑 `scripts/sft_data/viewer.sh` 顶部的 JSONL 配置路径、监听地址和端口，运行 `bash scripts/sft_data/viewer.sh`。网页支持数据集切换、sample_id 搜索、双模式轨迹、窗口悬停和前序／引用箭头；JSONL 和图片根目录的配置见 [网页说明](docs/sft_data/viewer.md)。

后续教师默认 `Qwen3.5-9B`，资源受限时可选 `Qwen3.5-4B`；Counting 暂停，等待 SAM 3 权重下载完成。
