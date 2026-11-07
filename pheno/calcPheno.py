
import time
import random
import cv2
import numpy as np
from pheno.lib import qimage_to_cv2
from pheno.outliers import contract_mask


class PhenomicsCalculator:
    @staticmethod
    def getAlgorithmNames():
        """
        返回当前注册的算法名称列表。
        """
        return list(PhenomicsCalculator.custom_algorithms.keys())

    custom_algorithms = {}
    custom_headers = {}

    # 每个默认性状的量纲。表头单位由此生成，而不是写死在字符串里——
    # 因为单位取决于当前比例：没标定时是像素，标定后是 mm/cm。
    #
    # 自定义算法（无代码性状）的列**故意不列在这里**：它们的量纲未知，
    # 猜一个单位比不写单位更糟。未登记的列返回空后缀，表头保持原样。
    TRAIT_DIMENSIONS = {
        # 长度：随比例线性变化
        "Perimeter": "length",
        "Major Axis": "length",
        "Minor Axis": "length",
        "Convex Perimeter": "length",
        # 面积：随比例平方变化
        "Area": "area",
        "Convex Area": "area",
        # 无量纲比值，加单位反而是错的
        "Aspect Ratio": "ratio",
        "Solidity": "ratio",
        "Circularity": "ratio",
        "Convexity": "ratio",
        # 8 位通道均值。标注取值范围，否则读者无法判断是 0-255 还是 0-1
        "MeanB": "channel",
        "MeanG": "channel",
        "MeanR": "channel",
    }

    # 未标定时的单位符号。写 px 而不是 pixel，表头才不会过宽。
    PIXEL_SYMBOL = "px"

    @staticmethod
    def unitSuffix(header, unit=None):
        """一列表头该带的单位后缀，例如 " (mm²)"；无单位的列返回空串。"""
        dimension = PhenomicsCalculator.TRAIT_DIMENSIONS.get(str(header).strip())
        if dimension is None or dimension == "ratio":
            return ""
        if dimension == "channel":
            return " (0-255)"

        symbol = str(unit or "").strip()
        if not symbol or symbol.lower() == "pixel":
            symbol = PhenomicsCalculator.PIXEL_SYMBOL
        return " ({}\u00b2)".format(symbol) if dimension == "area" else " ({})".format(symbol)

    @staticmethod
    def getHeaders(unit=None):
        """表头列表。``unit`` 是当前比例的单位（"mm"/"cm"/"pixel"）。

        默认按像素标注，因为未标定时结果确实是像素——留空反而让人以为
        那些数字已经是毫米。
        """

        headers = ["Name"]

        for name in PhenomicsCalculator.custom_headers.keys():
            head = PhenomicsCalculator.custom_headers[name]
            if isinstance(head, list):
                headers.extend(head)
            elif isinstance(head, str):
                headers.append(head)
            else:
                continue # None

        return [
            column + PhenomicsCalculator.unitSuffix(column, unit)
            for column in headers
        ]

    @staticmethod
    def getBareHeaders():
        """不带单位的列名，供需要稳定标识符的地方使用。"""
        headers = ["Name"]
        for name in PhenomicsCalculator.custom_headers.keys():
            head = PhenomicsCalculator.custom_headers[name]
            if isinstance(head, list):
                headers.extend(head)
            elif isinstance(head, str):
                headers.append(head)
        return headers

    @classmethod
    def calcPhenomics(cls, shape, pixmap, scale_per_pixel=1.0, image=None):

        # ``image`` lets a batch caller convert the frame once and share it
        # across all shapes instead of re-converting the pixmap per object.
        rgbImg = image if image is not None else qimage_to_cv2(pixmap.toImage())
        mask = shape.mask  # BW image (binary mask)
        mask = (mask > 0).astype(np.uint8)

        name = shape.label
        results = [name]

        for name, func in cls.custom_algorithms.items():
            if cls.custom_headers.get(name) is None:
                print(f"Warning: No headers defined for algorithm '{name}'")
                continue
            try:
                # 传递尺度信息给自定义算法
                result = func(rgbImg, mask, scale_per_pixel)
                if isinstance(result, list):
                    # 转换numpy类型为Python标准类型
                    converted_result = []
                    for item in result:
                        if isinstance(item, (np.integer, np.floating)):
                            converted_result.append(float(item) if isinstance(item, np.floating) else int(item))
                        else:
                            converted_result.append(item)
                    results.extend(converted_result)
                else:
                    # 单个值也转换类型
                    if isinstance(result, (np.integer, np.floating)):
                        results.append(float(result) if isinstance(result, np.floating) else int(result))
                    else:
                        results.append(result)
            except Exception as e:
                print(f"Error in {name}: {e}")
                results.extend([None] * len(cls.custom_headers[name]))

        
        # time.sleep(random.randint(1, 3))
        return results 

    @staticmethod
    def defaultPhenomics(rgbImg, mask, scale_per_pixel=1.0):
        """
        Extract default morphological features from the given shape.
        
        Features:
        - Area (in real-world units if scale is provided)
        - Perimeter (in real-world units if scale is provided)
        - Major Axis Length (in real-world units if scale is provided)
        - Minor Axis Length (in real-world units if scale is provided)
        - Aspect Ratio
        - Solidity
        - Circularity
        - Convex Area (in real-world units if scale is provided)
        - Convex Perimeter (in real-world units if scale is provided)
        - Convexity
        """

        # Contract the mask by the shared boundary-rim amount before
        # measuring morphology (see pheno.outliers.contract_mask); tiny
        # objects that would erode away keep the original mask. Colour stats
        # also use the contracted mask -- edge pixels are anti-aliased
        # object/background blends, so excluding them gives purer means.
        mask_m = contract_mask(mask)

        # Find contours
        contours, _ = cv2.findContours(mask_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # A fragment can survive erosion as a <5-point contour -- no ellipse
        # fit possible. Measure the uncontracted mask instead; a slightly
        # loose number beats a zero (see contract_mask docstring).
        contour = max(contours, key=cv2.contourArea) if contours else None
        if (contour is None or len(contour) < 5) and mask_m is not mask:
            raw_contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if raw_contours:
                raw_best = max(raw_contours, key=cv2.contourArea)
                if len(raw_best) >= 5:
                    mask_m = mask
                    contours = raw_contours
                    contour = raw_best

        if not contours:
            return [0] * 10  # Return zeros if no contour is found

        # Compute area (convert to real-world units)
        area = cv2.contourArea(contour) * (scale_per_pixel ** 2)

        # Compute perimeter (convert to real-world units)
        perimeter = cv2.arcLength(contour, closed=True) * scale_per_pixel

        # Compute circularity (unitless ratio)
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter != 0 else 0

        # Fit an ellipse if possible (requires at least 5 points)
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            major_axis = max(ellipse[1]) * scale_per_pixel  # Convert to real-world units
            minor_axis = min(ellipse[1]) * scale_per_pixel  # Convert to real-world units
        else:
            major_axis, minor_axis = 0, 0  # Default values if not enough points

        # Aspect Ratio (unitless ratio)
        aspect_ratio = major_axis / minor_axis if minor_axis != 0 else 0

        # Compute convex hull
        hull = cv2.convexHull(contour)
        convex_area = cv2.contourArea(hull) * (scale_per_pixel ** 2)  # Convert to real-world units
        convex_perimeter = cv2.arcLength(hull, closed=True) * scale_per_pixel  # Convert to real-world units

        # Solidity (unitless ratio - how much the shape fills its convex hull)
        solidity = area / convex_area if convex_area != 0 else 0

        # Convexity (unitless ratio - perimeter ratio between convex hull and contour)
        convexity = convex_perimeter / perimeter if perimeter != 0 else 0

        channel_mean = cv2.mean(rgbImg, mask=mask_m)[:3]  # B G R

        # 安全取均值，缺通道填充nan
        meanB = channel_mean[0] if len(channel_mean) > 0 else np.nan
        meanG = channel_mean[1] if len(channel_mean) > 1 else np.nan
        meanR = channel_mean[2] if len(channel_mean) > 2 else np.nan

        # Store results
        result = [
            area, perimeter, major_axis, minor_axis, aspect_ratio,
            solidity, circularity, convex_area, convex_perimeter, convexity,
            meanB, meanG, meanR
        ]

        return result

    @staticmethod
    def defaultHeaders():
        """
        Return the default headers for the phenomics table.
        """
        return [
            "Area", "Perimeter", "Major Axis", "Minor Axis", "Aspect Ratio",
            "Solidity", "Circularity", "Convex Area", "Convex Perimeter", "Convexity",
            "MeanB", "MeanG", "MeanR"
        ]
    
    @classmethod
    def registerAlgorithm(cls, selected_algorithms, selected_headers):
        cls.custom_algorithms = selected_algorithms
        cls.custom_headers = selected_headers


PhenomicsCalculator.custom_algorithms = {'DEFAULT': PhenomicsCalculator.defaultPhenomics}
PhenomicsCalculator.custom_headers = {'DEFAULT': PhenomicsCalculator.defaultHeaders()}
# The built-in measurement set, bound directly so formula traits can always
# reach it -- unchecking DEFAULT in Manage Algorithms only hides its columns
# and must never make the base measurements unavailable.
PhenomicsCalculator.default_algorithm = PhenomicsCalculator.defaultPhenomics