"""按颜色分割绿色叶片：剪叶摆在灯箱/白纸这类高对比场景。

Grounding DINO 把 4096x3072 压到千像素级再推理，细长叶片只剩几个像素宽，挨得近
的几片会被并成一个框 —— 实测某张"三片叶片挨得较近"的图只检出 1 片，换措辞、多
提示词并集都救不回来。而这类场景用颜色就能干净地分开：同一张图上，颜色法分出的
三个连通域与人工标注的包围框几乎逐像素重合，耗时 0.05 秒（语义检测要 1-3 秒），
而且直接产出掩膜，连 SAM 都不用跑。

适用前提：叶片是绿色，背景明显更亮/更淡（灯箱背光、白纸、白盘）。彩色背景、
枯黄叶片、或者叶片互相重叠粘连成一片时不适用 —— 那时请用语义检测或手动标注。
"""

import cv2
import numpy as np

# 绿色范围（HSV；OpenCV 的 H 取值 0-179）。上限放到 95 是为了吃下灯箱背光下偏
# 黄绿的透光叶片；下限 25 避开纸箱与木质桌面的暖黄。
DEFAULT_HSV_LOWER = (25, 40, 40)
DEFAULT_HSV_UPPER = (95, 255, 255)

# 面积下限：小于画幅 0.25% 的绿斑是噪点、叶片碎屑或远处的环境杂物。真实样本在
# 实测的 12 张图里占 0.40%-1.04% 画幅，而画面角落的环境绿斑只有 0.15%，这个
# 门槛把后者挡在外面、又不碰真样本。
MIN_AREA_RATIO = 0.0025
# 面积上限：大于画幅 25% 的绿色区域是背景或环境，不是摆好的叶片。
MAX_AREA_RATIO = 0.25
# 外接框填充率上限：细长叶片（尤其斜放时）填充率很低（实测 10%-33%），而块状
# 杂物（网袋、纸盒、桌面上的绿色物件）能到 70% 以上，用这个把它们挡掉。
MAX_BOX_FILL = 0.55


def segment_leaves(image, hsv_lower=DEFAULT_HSV_LOWER, hsv_upper=DEFAULT_HSV_UPPER,
                   min_area_ratio=MIN_AREA_RATIO, max_area_ratio=MAX_AREA_RATIO,
                   max_box_fill=MAX_BOX_FILL, drop_border=True,
                   close_iterations=3, open_iterations=2):
    """分割绿色叶片，返回 ``[{'mask', 'box', 'area'}, ...]``，按面积从大到小。

    ``image`` 是 OpenCV 的 BGR 数组（灰度/四通道也能处理）。``mask`` 是整图尺寸的
    uint8 0/255 掩膜，可以直接当对象掩膜用；``box`` 是 ``(x1, y1, x2, y2)``。
    """
    if image is None or getattr(image, "size", 0) == 0:
        return []

    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        bgr = np.asarray(image)

    height, width = bgr.shape[:2]
    frame_area = float(height * width)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, tuple(hsv_lower), tuple(hsv_upper))

    # 闭合补上叶脉/反光造成的镂空，开运算去掉零散噪点
    kernel = np.ones((7, 7), np.uint8)
    if close_iterations:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,
                                iterations=close_iterations)
    if open_iterations:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel,
                                iterations=open_iterations)

    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    results = []
    for index in range(1, count):
        x, y, w, h, area = (int(value) for value in stats[index])
        if area < min_area_ratio * frame_area:
            continue
        if area > max_area_ratio * frame_area:
            continue
        if max_box_fill and area / float(max(w * h, 1)) > max_box_fill:
            continue
        if drop_border and (x <= 0 or y <= 0
                            or x + w >= width or y + h >= height):
            # 贴着画面边缘的绿色多半是灯箱外的环境，不是摆好的样本
            continue

        full = np.zeros((height, width), dtype=np.uint8)
        full[y:y + h, x:x + w] = np.where(
            labels[y:y + h, x:x + w] == index, 255, 0).astype(np.uint8)
        results.append({'mask': full, 'box': (x, y, x + w, y + h), 'area': area})

    results.sort(key=lambda item: -item['area'])
    return results
