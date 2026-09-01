<div align="center">
  <h1 align="center">第三方组件与模型说明</h1>
</div>

## 1. 源码依赖

PanelTone 使用的 Python 与前端依赖由 `pyproject.toml`、`package.json` 和锁文件声明；发行者需要在发布二进制包前根据实际打包内容生成完整依赖清单并复核许可证

主要运行组件包括 FastAPI、Uvicorn、Pillow、NumPy、OpenCV、PyMuPDF、React、TypeScript 与 Vite；本文件不替代各项目的许可证原文

## 2. 模型

仓库不分发模型权重；`src/manga_repaint/model_catalog.json` 只保存模型标识、固定修订号、下载过滤规则和许可证链接

当前模型清单：

| 模型 | 用途 | 清单中的许可证 | 发布边界 |
|---|---|---|---|
| FLUX.2 Klein 4B | 快速画风编辑与多参考图编辑 | Apache-2.0 | 用户单独下载，使用前核对模型页面 |

表 2.1 当前模型清单

模型许可证不授予源漫画、目标画风、训练数据、参考图或输出结果的版权；用户必须自行确认处理与发布权利

## 3. 可选组件

ComfyUI、ControlNet、LoRA、MangaNinja、Cobra、Qwen-Image-Edit 与其他研究模型不随当前仓库发布；接入时必须单独记录源码、权重、训练数据与商用边界

