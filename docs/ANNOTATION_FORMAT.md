# PhenoMe 标注文件格式（`.pheno.json`）

本文件是桌面端（`pheno/`）与网页端（`web/`）之间的数据契约。桌面端的读写实现在
`pheno/annotation.py`，改动格式必须同步更新本文档。

- 格式名：`phenome-annotation`
- 当前版本：`1`
- 编码：UTF-8，紧凑 JSON（无缩进、无多余空格）

## 1. 文件位置与命名

每张图像对应一个同名 sidecar 文件，后缀 `.pheno.json`：

```
input/photo.jpg
input/photo.pheno.json
```

批处理时 sidecar 写入用户选定的输出目录，而不是图像所在目录。

写入是原子的：先写 `<path>.tmp`，再 `os.replace` 覆盖目标，避免中途崩溃留下半个文件。

## 2. 顶层结构

```json
{
  "format": "phenome-annotation",
  "version": 1,
  "image": {
    "path": "photo.jpg",
    "width": 4624,
    "height": 3472,
    "md5": "1e18364edbae033c00c1f638142fdbeb"
  },
  "scale": { "per_pixel": 1.0, "unit": "pixel" },
  "processing": { "color_correction": false, "ruler_detection": false },
  "objects": [ /* 见第 3 节 */ ]
}
```

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `format` | string | 固定 `"phenome-annotation"`。读取方应先校验此字段，不匹配则视为无效文件 |
| `version` | int | 结构版本。新增可选字段不升版本；语义变更才升 |
| `image.path` | string | 图像文件名（basename，不含目录），仅作参考 |
| `image.width` / `height` | int | 图像像素尺寸。所有坐标都以此为基准 |
| `image.md5` | string \| null | 图像文件字节的 MD5。用于检测 sidecar 与图像是否配套 |
| `scale.per_pixel` | float | 1 像素对应的真实长度 |
| `scale.unit` | string | 长度单位，`"pixel"` 表示未标定，标尺检测后通常为 `"mm"` |
| `processing.color_correction` | bool | 该图是否经过色卡校正 |
| `processing.ruler_detection` | bool | 该图是否跑过标尺检测 |
| `objects` | array | 对象列表，顺序即创建顺序 |

### 关于 `image.md5`

桌面端载入时会比对当前图像的 MD5：不一致则弹窗询问是否仍要载入，因为像素坐标可能
已经错位。网页端建议做同样的校验；若无法读取原图字节，可以退化为只校验
`image.width` / `height`。

`md5` 可能为 `null`（写入时读不到图像文件），此时跳过校验。

## 3. 对象结构

```json
{
  "label": "pod01",
  "name_prefix": "pod",
  "source_prompt": "bean pod",
  "detection_score": 0.7212,
  "polygon": [[2868.0, 2247.5], [2860.0, 2247.5], ...],
  "prompt_points": [{ "x": 2800, "y": 1500, "label": 1 }],
  "mask": {
    "encoding": "rle-zlib-base64",
    "image_size": [4624, 3472],
    "bbox": [2475, 841, 3061, 2246],
    "counts": "eNq913tojmEYx/Fn++4lrWWGzbGZ85xmaMth..."
  }
}
```

| 字段 | 类型 | 必需 | 说明 |
| --- | --- | --- | --- |
| `label` | string | 是 | **显示名**，用户可自由改名，也是导出 CSV 的 `Name` 列。同一张图内唯一，但**不是**跨会话的稳定标识 |
| `name_prefix` | string \| null | 是 | 自动编号用的词干。`null` 表示通用前缀 `Object` |
| `source_prompt` | string \| null | 是 | 该对象来自文本检测时，模型返回的标签（如 `bean pod`）；其他来源为 `null` |
| `detection_score` | float | 否 | 文本检测置信度，仅文本检测对象有 |
| `polygon` | array | 是 | 轮廓点 `[[x, y], ...]`，1 位小数。可能为空数组 |
| `prompt_points` | array | 否 | Click 分割用的提示点，`label` 为 `1`（前景）或 `0`（背景） |
| `mask` | object | 否 | 二值掩膜，见第 4 节。缺失时只能依赖 `polygon` |

### `polygon` 的时效性

桌面端支持笔刷编辑掩膜；每次编辑结束会重算并覆写 `polygon`，因此两者保持一致。
但如果**任何一方**（网页端、外部脚本）单独改了 `mask` 或 `polygon` 而没有同步另一方，
读取方必须以 `mask` 为准：`polygon` 只是掩膜轮廓的近似（容差 `0.001 × 最大跨度`），
仅供快速预览。`mask` 缺失时才允许退回 `polygon`。

### `label` 与内部标识

桌面端内部另有一个 `Shape.uid`（单调递增、永不复用），但它**不写入文件**，因为跨会话
没有意义。文件里没有稳定 ID：需要跨会话追踪同一对象时，请自行约定（例如用
`label`，但要接受用户改名的可能）。

### 命名规则（供网页端保持一致）

新建对象的名字是 `<前缀><两位序号>`，序号取当前该前缀下**最大序号 + 1**：

- `Object01..Object05`，删掉 `Object03` → 下一个是 `Object06`
- `Object01..Object05`，删掉 `Object05` → 下一个是 `Object05`
- 全部删除 → 下一个是 `Object01`

不同前缀独立计数：`pod01` 与 `Object01` 互不影响。文本检测的对象用用户输入的提示词
作前缀（输入 `pod` → `pod01`、`pod02`），而不是模型返回的 `bean pod`。

## 4. 掩膜编码

掩膜是本格式的**主数据**。`polygon` 只是掩膜轮廓的简化近似（简化容差
`0.001 × 最大跨度`），仅适合快速预览；面积、周长、solidity 等性状必须由掩膜计算，
否则会与桌面端结果不一致。

### 编码流程

1. 掩膜二值化（`> 0` 为前景）。
2. 求前景的紧致外接框 `bbox = [x1, y1, x2, y2]`（**闭区间**，包含右下角像素）。
3. 裁剪到 bbox。
4. **列优先**（column-major / Fortran order）展平，做游程编码。
5. 游程长度存为 `uint32` **小端**数组。
6. `zlib.compress(..., level=9)`，再 `base64` 标准编码（含 `+` `/` `=`）。

### 游程约定

游程交替表示 背景 / 前景，且**总是从背景开始**。若首像素是前景，则首个游程长度为
`0`。这与 COCO RLE 一致，解码端无需分支判断。

```
counts = [13, 3, 3, 3, 3, 3, 14]
         ↑背景 ↑前景 ↑背景 ...
```

所有游程之和必须等于 `裁剪高 × 裁剪宽`。解码端应校验这一点。

### `mask` 字段

| 字段 | 说明 |
| --- | --- |
| `encoding` | 固定 `"rle-zlib-base64"`。遇到未知值应报错而非猜测 |
| `image_size` | `[width, height]`，**注意是宽在前**，与 `bbox` 所在的全图坐标系一致 |
| `bbox` | `[x1, y1, x2, y2]` 闭区间，裁剪尺寸为 `(y2-y1+1, x2-x1+1)` |
| `counts` | base64 字符串 |

### JavaScript 解码参考

```js
// pako 用于 zlib 解压（浏览器端）
import pako from "pako";

function decodeMask(payload) {
  if (payload.encoding !== "rle-zlib-base64") {
    throw new Error(`unsupported encoding: ${payload.encoding}`);
  }
  const [imgW, imgH] = payload.image_size;
  const [x1, y1, x2, y2] = payload.bbox;
  const cropW = x2 - x1 + 1;
  const cropH = y2 - y1 + 1;

  // base64 -> 字节 -> zlib 解压 -> uint32 小端
  const raw = Uint8Array.from(atob(payload.counts), (c) => c.charCodeAt(0));
  const bytes = pako.inflate(raw);
  const counts = new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.length / 4);

  // 展开游程：偶数下标是背景，奇数下标是前景
  const crop = new Uint8Array(cropW * cropH);
  let pos = 0;
  for (let i = 0; i < counts.length; i++) {
    const run = counts[i];
    if (i % 2 === 1) crop.fill(1, pos, pos + run);
    pos += run;
  }
  if (pos !== cropW * cropH) {
    throw new Error(`run lengths sum to ${pos}, expected ${cropW * cropH}`);
  }

  // 列优先 -> 行优先，并放回全图坐标
  const full = new Uint8Array(imgW * imgH);
  for (let cx = 0; cx < cropW; cx++) {
    const colBase = cx * cropH;
    for (let cy = 0; cy < cropH; cy++) {
      if (crop[colBase + cy]) {
        full[(y1 + cy) * imgW + (x1 + cx)] = 1;
      }
    }
  }
  return { data: full, width: imgW, height: imgH };
}
```

编码方向相同：按列扫描裁剪区域累计游程，`uint32` 小端打包后 zlib + base64。

> 注意 `new Uint32Array(bytes.buffer, ...)` 依赖字节对齐。若 `pako` 返回的
> `byteOffset` 不是 4 的倍数，需先 `bytes.slice()` 拷贝一份。

## 5. 实测体积与耗时

数据来自一张 16 MPx（4624×3472）的豆荚照片，文本检测出 3 个对象，
每个前景约 44–50 万像素：

| 项目 | 大小 |
| --- | --- |
| 单个对象的 `mask` 字段 | 1.1 KB |
| 3 个对象掩膜合计 | 3.3 KB |
| 3 个对象多边形合计（1 位小数） | 21.2 KB |
| 完整 sidecar 文件 | 25.2 KB |

对比参照：同样的掩膜导出为 8bit PNG 是 19.9 KB/个，1bit PNG 是 4.5 KB/个。

**多边形反而是体积主因**（每个对象约 470 个点）。如果网页端不需要预览轮廓，可以在
生成时省略 `polygon` 字段，文件会降到 4 KB 左右；读取端必须容忍该字段缺失或为空。

耗时（RTX A2000，3 个对象）：

| 操作 | 耗时 |
| --- | --- |
| `saveAnnotation()` 全流程 | 79 ms |
| `loadFile()` 含标注载入 | 101 ms |
| 单对象 RLE 编码（裁剪后） | 0.9 ms |
| 单对象 RLE 解码 | 0.9 ms |
| zlib + base64 编解码 | < 0.1 ms |
| 参照：单次 SAM `predict_box` | ~250 ms |
| 参照：从掩膜重算多边形 | ~240 ms |

从掩膜重算多边形（240 ms/对象）比解码掩膜（1 ms）慢两个数量级，这也是选择存
`polygon` 而非载入时重算的原因；已验证重算结果与保存值在 0.011 px 内一致。

## 6. 兼容性约定

- **读取方必须容忍未知字段**，不要因为多了字段就报错。
- 可选字段缺失时应有合理默认：`scale` 缺失按 `1.0 / "pixel"`，`processing` 缺失按
  全 `false`，`polygon` 缺失按空数组。
- `format` 不匹配或 JSON 解析失败时，桌面端的做法是**静默忽略**（当作没有标注），
  而不是弹错误；网页端建议一致，避免坏文件阻塞流程。
- 新增字段请保持可选，并在本文档登记；破坏性变更需要提升 `version`，读取方应拒绝
  高于自己支持的版本。

## 7. 桌面端行为参考

- 打开图像时自动尝试载入同名 sidecar（批处理模式下跳过）。
- `文件 → 保存标注`（`Ctrl+S`）手动保存。未保存提醒里的「保存」按钮调用的是同一个
  `saveAnnotation()`，所以快捷键和对话框的语义一致。
- 有未保存对象时切图或关闭，会弹出 保存 / 丢弃 / 取消 三选。
- 工具栏「导出掩膜」（`Ctrl+Shift+M`）把**勾选**对象的掩膜写成 1-bit PNG，给人或
  工具用；它是导出而非保存——没有任何代码读回这些 PNG，而 sidecar 里同样带着掩膜
  （RLE，约 1.1 KB/对象，对比 1-bit PNG 4.5 KB、8-bit PNG 19.9 KB）。
- 批处理每张图在写掩膜 PNG 的同时写一份 sidecar 到输出目录。
- 载入的对象保留文件中的名字，不会被重新编号；后续新建对象按第 3 节规则接续。
- 笔刷编辑（B 键）结束后重算并覆写被编辑对象的 `polygon`，随后保存即落盘。
- 「检阅批处理结果」（`Ctrl+R`）独立窗口只读 sidecar：解码 `mask` 重算性状、
  按 `name_prefix` 分组标面积离群（0.5x–2.0x 中位数），导出宽表与缩略图总览；
  找不到原图时几何性状仍可计算，色彩列报 `image not found`。
