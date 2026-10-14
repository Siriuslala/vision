# 测试目录

测试按被测功能分类，分析实验进一步按研究问题划分。

| 目录 | 测试范围 |
| --- | --- |
| `analysis/common/` | 共享合成刺激的模板、属性变化、画布对齐与确定性 |
| `analysis/representation/` | 表征距离、局部改动、实例布局与高范数处理 |
| `analysis/position/` | 位置标签、网格顺序与坐标 probe |
| `analysis/token_semantics/` | 词汇读出、范数组划分与中文显示 |
| `analysis/behavior/` | Grounding 数据构造、答案解析与几何评分 |
| `evaluation/` | Benchmark、解码配置、图像输入、推理与结果查看 |
| `sft_data/` | SFT 数据转换、标注约束、自动生成、计数、窗口与数据查看 |
| `infrastructure/` | 路径配置与下载、校验、断点恢复 |

在仓库根目录执行 `bash scripts/testing/check.sh`，会递归运行全部测试，同时检查 Bash 语法和推理 CLI。各层 `__init__.py` 用于支持 Python 3.12 的 unittest 递归发现；新测试文件使用 `test_*.py` 命名，新增子目录也应包含该文件。

单独检查一个类别时，例如位置信息测试，可在仓库根目录执行：

```bash
uv run --frozen python -m unittest discover -s tests/analysis/position -t . -v
```

其中 `-t .` 指定仓库根目录为导入起点，使测试以完整包名加载。
