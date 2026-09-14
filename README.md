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

Use quoted, literal absolute paths. Spaces are supported; variable references and shell commands are not part of the supported configuration format.

`.env` is private and ignored by Git. `.env_example` contains empty values and generic instructions and can be committed. Keep machine-specific paths in `.env` and experiment settings such as model, cases, batch size, and GPUs in the Bash scripts. Outputs go to `WORK_DIR`; placing it outside the repository keeps generated artifacts separate from source files. Runtime metadata may contain resolved local paths.

Bash entry points load configuration through `scripts/load_env.sh`. Direct Python entry points read the same file through `utils/project_paths.py`. Missing required paths produce an explicit error. The root `.python-version` file selects Python 3.12 for uv; `requires-python` in `pyproject.toml` declares the supported version range. Dependencies are managed with uv and pinned in `uv.lock`; scripts use `uv run --frozen`. Conda activation is unnecessary. Model weights are loaded offline from the local cache.

| Script | Purpose and configuration | Launch command from the repository root |
| --- | --- | --- |
| `scripts/setup_env.sh` | Install the locked environment | `bash scripts/setup_env.sh` |
| `scripts/simple_test.sh` | Configure one model and paired image/prompt lists | `bash scripts/simple_test.sh` |
| `scripts/evaluation.sh` | Configure full-benchmark or subset evaluation | `bash scripts/evaluation.sh` |
| `scripts/smoke_models.sh` | Run paired image/prompt lists on a model array sequentially | `bash scripts/smoke_models.sh` |
| `scripts/prepare_data.sh` | Configure the dataset source and export destination | `bash scripts/prepare_data.sh` |
| `scripts/download.sh` | 配置 Hugging Face 数据集名称、保存路径及下载参数 | `bash scripts/download.sh` |
| `scripts/validate_data.sh` | Configure the dataset and directory to validate | `bash scripts/validate_data.sh` |
| `scripts/check.sh` | Check Bash syntax, unit tests, and Python CLIs | `bash scripts/check.sh` |

For example, from the repository root:

```bash
bash scripts/simple_test.sh
```

## Configuring experiments in Bash

Edit the configuration section in `scripts/simple_test.sh`:

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

## Generation length

Earlier tests with a 512-token limit sometimes stopped before producing a final answer. The three inference Bash scripts and the Python CLI therefore default to **4096 new tokens**. This is an initial budget informed by those observations, not a verified guarantee that every case will finish.

- With the default `THINKING="off"`, start with 4096 and inspect truncation on representative cases or the full benchmark.
- For Qwen3.5 with `THINKING="on"`, an initial budget of 8192 can be evaluated and adjusted based on observed truncation.
- `MAX_NEW_TOKENS` is an upper limit, not a required output length. Generation ends early when the model emits EOS.
- If many cases still have `finish_reason="length"`, inspect their responses and consider a larger consistent budget. Long output can also reflect repeated reasoning or loops.
- Keep generation and image budgets consistent across model comparisons. Report truncation counts and do not treat the earlier 128/512-token interface checks as benchmark performance estimates.

`THINKING` affects the two supported Qwen3.5 models. The Qwen3-VL Instruct checkpoints do not implement this switch. Qwen3.5 currently uses the PyTorch reference implementation of linear attention, without the optional `causal_conv1d` and `flash-linear-attention` acceleration packages; long generation may be slow.

## Simple-test outputs and resuming

Simple test writes one JSONL record per image/prompt pair with `sample_id` (zero-based input index), `image_path`, the exact `prompt`, full `response`, model name, generated token count, stop reason, and batch timing. Its summary reports sample count, truncations, and generated tokens. No accuracy is computed because these inputs have no reference-answer labels.

The companion metadata file records model and generation settings, ordered image/prompt pairs, and SHA256 hashes of image contents. With `RESUME=true`, settings and inputs must match the original run. Changing a prompt, the input order, or image bytes at the same path requires a new output file. Completed records are skipped by `sample_id`; interrupted samples without a saved record are regenerated from the start.

## Full evaluation, outputs, and resuming

Set `MODEL`, `GPU_IDS`, `OUTPUT_FILE`, and other parameters in `scripts/evaluation.sh`. Leave `CASE_IDS=()` for the full benchmark, then run:

```bash
bash scripts/evaluation.sh
```

After an interruption, change `RESUME=false` to `RESUME=true` in the same script, keep the other settings unchanged, and run the same command. Model paths, parameters, selected cases, and the manifest hash must match. If you change the generation budget, choose a new output file; results from different configurations are not merged during resumption.

Each run produces:

- `*.jsonl`: task ID, categories, prompt, full response, prediction and reference answer, generated token count, stop reason, and batch timing.
- `*.jsonl.meta.json`: configuration, model snapshot, manifest hash, dependency versions, device mapping, and processor configuration.
- `*.jsonl.summary.json`: accuracy, unparsed answers, truncations, and breakdowns by category, subtype, and answer type.

Each record is flushed after writing; summaries are updated after each batch. Existing output files are protected from accidental overwrites. Malformed JSONL produces an error rather than being silently skipped. `batch_seconds` measures an entire batch and should not be summed once per case within that batch.

Model inputs contain only the image, question, and options. Reference answers and dataset CoT annotations are excluded. Local strict scoring preferentially extracts `Final answer: ...`. Multiple-choice predictions must be valid option letters. Blank answers are normalized only for Unicode, whitespace, case, and trailing periods; unparsed answers count as incorrect. Semantically equivalent text can fail strict matching, so inspect raw responses. This differs from the [official LLM-judge evaluation](https://github.com/UniPat-AI/BabyVision/tree/main/babyvision_eval).

## Hugging Face 数据集下载

在 `scripts/download.sh` 顶部设置 `REPO_ID`、`LOCAL_DIR`、`REVISION`、并发数和超时，然后执行 `bash scripts/download.sh`。默认下载 `allenai/pixmo-count` 到 `.env` 中 `DATA_DIR` 下的 `pixmo-count` 目录。机器相关路径保存在 `.env`。

Bash 入口继承环境中的 `HF_ENDPOINT`、`http_proxy` 和 `https_proxy`；也可在私有 `.env` 中显式配置这些变量，由 `scripts/load_env.sh` 导出。非交互式 Shell 不保证执行 `.bashrc` 中的代理配置。从任务运行器启动下载时，可通过 `.env` 保证网络设置生效。镜像地址控制 Hub API 入口，实际文件下载仍可能被重定向到其他存储域名。

下载默认使用普通 HTTP，避免 `hf-xet` 的 CAS 响应解码故障；`USE_XET=true` 可启用 Xet。临时 HTTP 网络错误和部分服务端错误最多尝试 `ATTEMPTS` 次，最终失败保留异常并返回非零状态。重复运行可复用已完成文件及目标目录下的 `.cache/huggingface` 元数据；未完成文件能否续传取决于 Hub 版本和下载后端，不保证跨进程续传。下载入口使用项目锁定的 uv 环境。

`snapshot_download` 下载仓库文件。PixMo-Count 的 Parquet 包含图片 URL 和标注，图片需根据 `image_url` 另行下载，并可通过 `image_sha256` 校验。比较下载大小时使用 `du -sb` 或文件字节数；`du -sh` 显示磁盘占用，在支持压缩的文件系统上可能更小。

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
bash scripts/install_flash_attention.sh
```

The script synchronizes the locked environment and reports the package and native
extension paths. Dependencies are installed inside `${ROOT_DIR}/.venv`, normally
under `.venv/lib/python3.12/site-packages/`. The uv cache is a separate download and
installation cache; use `uv cache dir` to locate it. `DATA_DIR` and `WORK_DIR` do not
control where Python dependencies are installed.

To verify GPU execution, set `GPU_ID` at the top of the check script and run:

```bash
bash scripts/check_flash_attention.sh
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

## 一般视觉推理 SFT 数据生产

SFT 生产覆盖 Visual-CoT、Thyme-SFT、PixMo-Count、CoSyn-400K 和 LLaVA-OneVision。方法、字段和小样本结果见 [生产说明](docs/sft_data/sft_production.md)。先编辑脚本顶部参数，再运行对应 Bash 入口；默认仅处理少量样本，`LIMIT=0` 在生产 adapter 中表示全部已完成来源记录。本地教师入口要求正的显式数量上限。

共享下载源可在私有 `.env` 中设置 `DATA_DIR_PUBLIC`；未设置时使用 `DATA_DIR`。源文件只读，提取或派生图片写到独立目录。候选与日志写入 `WORK_DIR`，标准图片写入 `DATA_DIR/sft_images`。所有来源应共用图片父目录下的 split 注册表，避免同图跨划分。

| 入口 | 用途 |
| --- | --- |
| `bash scripts/sft_data/produce_visual_cot.sh` | 从原始 metadata 与原图构造 ROI 过程；可选限量提取已完成 tar 分片 |
| `bash scripts/sft_data/produce_thyme.sh` | 安全恢复裁剪轨迹，核对派生图与原图窗口 |
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
