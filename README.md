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
| `scripts/simple_test.sh` | Configure one model and selected cases at the top | `bash scripts/simple_test.sh` |
| `scripts/evaluation.sh` | Configure full-benchmark or subset evaluation | `bash scripts/evaluation.sh` |
| `scripts/smoke_models.sh` | Configure a model array and test models sequentially | `bash scripts/smoke_models.sh` |
| `scripts/prepare_data.sh` | Configure the dataset source and export destination | `bash scripts/prepare_data.sh` |
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
CASE_IDS=(445 462)  # Use (445) for a single case.
OUTPUT_FILE="$WORK_DIR/qwen35_4b_cases_4096.jsonl"
BENCH="babyvision"
SPLIT="train"
DATA_ROOT="$DATA_DIR"
CACHE_ROOT="$HF_HUB_CACHE"
BATCHSIZE=1
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

`CASE_IDS` is a Bash array of original `taskId` values, not row indices. Use `(445)` for one case or `(445 462 507)` for multiple cases. In `evaluation.sh`, `CASE_IDS=()` selects all 388 cases in the split. The arrays in `simple_test.sh` and `smoke_models.sh` must not be empty.

`GPU_IDS="0,3"` makes two GPUs visible for automatic placement of a single model. This is model placement, not data parallelism with one model replica per GPU. `BATCHSIZE` controls the number of cases passed to each generation call. Its default of 1 provides a starting point for checking memory requirements with longer outputs. Multiple selected cases can still run sequentially with batch size 1.

`MAX_PIXELS=""` preserves the model processor's image-resolution settings. Setting a numeric limit changes the visual input budget; keep it consistent when comparing models. `DRY_RUN=true` validates cases and prompts without loading weights. Use a separate output file for dry runs.

## Generation length

Earlier tests with a 512-token limit sometimes stopped before producing a final answer. The three inference Bash scripts and the Python CLI therefore default to **4096 new tokens**. This is an initial budget informed by those observations, not a verified guarantee that every case will finish.

- With the default `THINKING="off"`, start with 4096 and inspect truncation on representative cases or the full benchmark.
- For Qwen3.5 with `THINKING="on"`, an initial budget of 8192 can be evaluated and adjusted based on observed truncation.
- `MAX_NEW_TOKENS` is an upper limit, not a required output length. Generation ends early when the model emits EOS.
- If many cases still have `finish_reason="length"`, inspect their responses and consider a larger consistent budget. Long output can also reflect repeated reasoning or loops.
- Keep generation and image budgets consistent across model comparisons. Report truncation counts and do not treat the earlier 128/512-token interface checks as benchmark performance estimates.

`THINKING` affects the two supported Qwen3.5 models. The Qwen3-VL Instruct checkpoints do not implement this switch. Qwen3.5 currently uses the PyTorch reference implementation of linear attention, without the optional `causal_conv1d` and `flash-linear-attention` acceleration packages; long generation may be slow.

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

`utils/project_paths.py` reads and validates private path configuration. `data/` handles export and validation. `eva/benchmark.py` implements case selection, prompt construction, and scoring. `eva/inference.py` shares offline model loading and batched generation. `eva/simple_test.py` and `eva/evaluation.py` are Python entry points; use the scripts in `scripts/` for routine execution.

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
