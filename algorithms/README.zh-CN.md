# 自定义表型算法

[English](README.md) | **简体中文**

这个目录存放 PhenoMe 的自定义表型算法。在桌面端 `Manage Algorithms →
Load Custom Algorithms` 时，对话框会默认定位到这里。

## 算法文件格式

每个 `.py` 文件是一个独立算法，需要暴露两个函数：

```python
def getHeaders():
    """返回特征名列表，顺序与 calcTraits 的返回值一一对应。"""
    return ["Area", "Perimeter", ...]


def calcTraits(rgb_img, mask, scale_per_pixel=1.0):
    """
    参数:
    - rgb_img: 原始图像 (numpy 数组, HxWx4 BGRA)。alpha 通道标记当前对象
      像素（mask 内为 255，背景为 0）；如需 BGR 三通道，取 rgb_img[:,:,:3]。
    - mask: 二值掩码 (HxW, >0 为前景)
    - scale_per_pixel: 每像素对应的真实世界尺度

    返回: 特征值列表，顺序与 getHeaders() 对应。
    """
    return [area, perimeter, ...]
```

`scale_per_pixel` 表示当前测量单位下一个像素的尺寸。未标定时使用像素单位；
自动标尺检测使用毫米，手动 Set Scale 使用所选单位。长度类特征乘此比例因子，
面积类特征乘它的平方。

## 自带算法

| 文件 | 对象 | 特征 |
| --- | --- | --- |
| `beanSeedPheno.py` | 豆类种子 | 形态 10 项 + MeanB/G/R + 种脐（caruncle）面积与曲率，共 15 项 |
| `beanPodPheno.py` | 豆荚 | 面积、弯曲度、宽、长、两端尖锐度、边缘长度、骨架长 + 颜色，共 12 项 |

## 依赖

- `beanSeedPheno.py`: cv2, numpy, scikit-learn, scikit-image
- `beanPodPheno.py`: cv2, numpy, scipy, scikit-image, shapely

以上均已在 `requirements.txt` 中。

## 自己写一个

复制任意一个现有文件改起即可；最简模板：

```python
import cv2

def getHeaders():
    return ["Area"]

def calcTraits(rgb_img, mask, scale_per_pixel=1.0):
    area = cv2.countNonZero(mask) * (scale_per_pixel ** 2)
    return [area]
```

## 不想写代码：公式性状（节点图）

如果只是想在基础测量值上做算术组合（如果形指数、颜色比值），完全不用写
Python。`Manage Algorithms → New Formula Traits...` 打开节点图编辑器：

- 从节点右侧端口拖到另一节点左侧端口连线；右键空白处添加节点，
  右键节点重命名或删除；点击已有连线可断开。
- 节点分三类：测量值（蓝）、运算（紫，`+ - * / ^ %`、`sqrt`、`abs`、
  取负、`round`）、输出/性状（绿）。
- **单位自动推导**：按量纲分析得出——长度×长度=面积单位、面积开方=长度
  单位、比率无量纲；单位随当前图像的测量尺度（标尺检测或 Set Scale 设置）
  变化，无需手输。无意义的组合（如长度开方）自动显示为无单位。
- 右侧面板实时显示编译出的表达式和试算值；保存为 `.phenoFormula.json`
  （连节点布局一起存，重开可恢复图的样子），与 `.py` 算法完全同级：
  导入、加载、批处理、检阅窗口都能直接用。

变量：`Area, Perimeter, MajorAxis, MinorAxis, AspectRatio, Solidity,
Circularity, ConvexArea, ConvexPerimeter, Convexity, MeanB, MeanG, MeanR`。
这些基础值来自内置测量集，**始终可用**——在 Manage Algorithms 中取消勾选
DEFAULT 只是不在表格中显示其列，不会使公式失输入。除零与缺失值产出
`nan`，不会中断批处理。

公式文件是 JSON，也可手工编辑。实现见 `pheno/formulaTraits.py`（求值）、
`pheno/formulaGraph.py`（图模型、量纲分析）、`pheno/formulaCanvas.py`
（画布）、`pheno/formulaEditorDialog.py`（编辑器）。
