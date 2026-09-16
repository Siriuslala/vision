# SFT 数据交互可视化

网页从后台配置的 JSONL 文件读取真实样本，将原图、两种推理模式及完整字段并排展示。窗口标记与图像上的几何关联来自原始 `trajectory` 和长边归一化坐标，用于逐条检查观察、引用、语言及最终答案。

## 1. 配置与启动

配置文件位于 `scripts/sft_data/viewer_config.json`，为每个数据集配置名称、JSONL 文件和图片根目录。当前试制数据保存在 `/archive/public/tmp`，配置中直接记录实际数据路径，不包含 `.env` 中的私密配置或凭据。以下为配置结构示例；JSON 不展开环境变量。

```json
{
  "datasets": [
    {
      "name": "Visual-CoT",
      "jsonl_path": "/data/work/sft/visual_cot/candidates.jsonl",
      "data_root": "/data/training"
    },
    {
      "name": "CoSyn-400K",
      "jsonl_paths": [
        "/data/work/sft/cosyn/interleaved.jsonl",
        "/data/work/sft/cosyn/visual_only.jsonl"
      ],
      "data_root": "/data/training"
    }
  ]
}
```

配置顶层可选 `font_path`，指定本地中文 OTF 字体文件；后台将其提供给浏览器，适用于缺少中文字库的情况。未配置时使用浏览器系统字体。

同一 JSONL 可以混合两种模式，也可使用 `jsonl_paths` 提供多个文件。图片实际路径由 `data_root` 与记录中的 `image.path` 拼接。数据集没有可用候选时暂不加入配置；损坏 JSON、重复 sample_id 和不支持的坐标规范在启动时明确报错。

编辑 `scripts/sft_data/viewer.sh` 顶部的 `CONFIG`、`HOST`、`PORT`，执行：

```bash
bash scripts/sft_data/viewer.sh
```

默认仅监听本机。远程服务器可使用 SSH 端口转发访问；需要直接通过服务器地址访问时，将 `HOST` 配置为 `0.0.0.0`。服务只提供已配置数据集及其图片，不提供任意文件路径浏览。

启动时扫描 JSONL 建立文件偏移索引，浏览时按需读取样本，原图按需加载。修改或追加 JSONL 后需要重启服务重建索引；检测到文件变化时明确提示，避免旧索引读取错误记录。

## 2. 页面交互

- 顶部下拉菜单选择数据集；左右箭头按题目切换，键盘左右键同样有效。
- 搜索框接受当前数据集中的完整 `sample_id`，两种模式的 ID 都可定位对应题目。
- 左栏显示原图，右栏分别展示 interleaved 与 visual_only，再展示可展开的字段表。
- 自然语言为黑色文字，视觉读取为绿色 `<window_t>`，其中 t 使用真实 `region_id`。
- 悬停或键盘聚焦窗口标记，显示亮绿色当前框及左上角编号。浅绿色虚框标示前一窗口，红色虚框标示参考窗口。
- 亮绿色箭头从前一窗口指向当前窗口，亮红色箭头从当前窗口指向参考窗口。初始区域 0 表示整图；两个箭头方向分别保留。相同中心使用曲线避免箭头退化。
- 点击窗口标记固定显示，点击“清除选中”或按 Escape 取消固定。
- 字段表可切换所属模式，并支持下载当前原始 JSON 记录。

两种模式按来源名称、来源版本、原始样本 ID、图片身份、问题及答案共同配对。缺少另一种模式时显示“未提供”，不在浏览器中自动删除语言。多条同模式示范独立保留。同图不同问题不合并。

图像上的绘制先将长边归一化坐标还原为原图像素，再随原图等比例显示；非正方形图像不会被强行拉伸。窗口与完整字段展示不修改源 JSONL，也不自动改变质量状态。

## 3. 代码位置与开发检查

- `data/sft/viewer/server.py`：配置、JSONL 索引、样本搜索与图片服务。
- `data/sft/viewer/static/`：网页布局、窗口与箭头、交互逻辑。
- `tests/test_sft_viewer.py`：模式配对、不同题目、输入变化、重复 ID 和图片目录边界。

服务使用 Python 标准库，前端使用原生 HTML、CSS、JavaScript 与 SVG，无新增项目依赖。
