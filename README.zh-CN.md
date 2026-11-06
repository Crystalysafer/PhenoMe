# <img src="pheno/icons/icon.png" width="32"> PhenoMe

[English](README.md) | **简体中文**

一款面向植物器官的标准化高通量表型分析软件，操作简单易用。
一张照片进去——校准好的逐目标性状表格出来。

PhenoMe 从普通 RGB 照片中分割植物器官，利用画面中的色卡与刻度尺完成
颜色与尺度的校准，并为每个目标计算可配置的性状表——全程无需训练任何
模型。它在设计上与器官种类无关（种子、荚果、叶片、花朵……均可）；随附
的性状描述符集以豆科种子与荚果为示例，更多物种/器官无需重新训练即可
接入。

## 功能特性

- <img src="pheno/icons/text_detect.png" width="18"> **零样本器官分割** —
  Grounding DINO 将自由文本提示
  （"pod"、"seed"、"leaf"、"flower"）解析为检测框；SAM / SAM2 再把框
  （或用户点击/笔刷）转换为掩膜。无需针对物种训练。
- <img src="pheno/icons/color_checker.png" width="18"> **可信的校准** —
  MCC24 色卡校正会报告校正前后的 ΔE，并
  *拒绝*拟合不佳的结果；RulerNet 读取刻度尺并自检刻度模式，错误的尺度
  会被直接拒绝，而不是静默污染每一条测量。
- <img src="pheno/icons/calc.png" width="18"> **定量性状，可追溯** —
  每个目标的形状、尺寸与颜色，以及
  器官特异性状（荚果弯曲度与尖端锐度、种子种阜面积与曲率……）。每个
  数字都可回溯到掩膜、提示词以及保存在 `.pheno.json` 附属文件中的校准
  信息。
- <img src="pheno/icons/file.png" width="18"> **离线桌面应用** —
  基于 PyQt5，支持批量处理、掩膜编辑与复核窗口。
- <img src="pheno/icons/manageAlgorithm.png" width="18"> **无需编程即可扩展** —
  在节点图编辑器中定义新性状，自动进行
  量纲分析；或把一个只含两个函数的 Python 文件放进 `algorithms/`
  （编写指南见 `algorithms/README_EN.md`）。

## 获取应用

旧版 v1 安装包存档于
<https://doi.org/10.5281/zenodo.19936151>，它**不是**当前的 v2
源码，也不是经过验证的 v2 安装包。本源代码快照单独提供；新的 v2
版本化安装包尚未发布。

> 模型权重不在本仓库中——安装包已内置全部权重；源码运行请按下表单独
> 获取（或从上方 Zenodo 记录下载 `models.zip`）。

Web 前端的在线演示见 <https://webpheno.me>。

## 源码运行

需要 Python 3.10；依赖见 `requirements.txt`。有 CUDA GPU 时会自动启用，
但没有也能运行。

```bash
pip install -r requirements.txt
# 另需：models/sam_vit_h_4b8939.pth 和 models/grounding-dino-tiny/（见下表）

python main.py                      # 桌面应用
```


源码树所需的模型资源：

| 路径 | 内容 | 来源 |
| --- | --- | --- |
| `models/sam_vit_h_4b8939.pth` | SAM ViT-H 权重 | [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything) |
| `models/sam2/sam2.1_hiera_tiny.pt` | SAM2.1 Hiera-Tiny 权重 | [facebookresearch/sam2](https://github.com/facebookresearch/sam2) |
| `models/sam2/sam2.1_hiera_large.pt` | SAM2.1 Hiera-Large 权重 | [facebookresearch/sam2](https://github.com/facebookresearch/sam2) |
| `models/grounding-dino-tiny/` | Grounding DINO Tiny（配置 + 权重 + 分词器） | [IDEA-Research/grounding-dino-tiny](https://huggingface.co/IDEA-Research/grounding-dino-tiny) |
| `models/rulernet.onnx` | RulerNet 刻度检测器 | 随安装包 / Zenodo `models.zip` 提供 |

`build_v2.bat` + `PhenoMe.spec` 可构建 Windows 打包版本。

## 目录结构

| 路径 | 内容 |
| --- | --- |
| `main.py` | 桌面端入口 |
| `pheno/` | 应用程序 + 共享分析内核（SAM、校准、性状、UI） |
| `algorithms/` | 性状算法插件 + 编写指南 |

## 文档

- `algorithms/README_EN.md` — 插件编写指南

## 引用

如果你在研究中使用了 PhenoMe，请引用 Zenodo 上存档的软件版本（DOI
见上文）——以及修订版发表后刊登于 *Plant Phenomics* 的配套论文。

## 许可证

MIT — 见 `LICENSE`。
