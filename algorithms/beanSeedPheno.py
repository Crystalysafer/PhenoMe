def getHeaders():
    """返回豆类种子分析的特征标题"""
    return [
        "Area", "Perimeter", "Major Axis", "Minor Axis", "Aspect Ratio",
        "Solidity", "Circularity", "Convex Area", "Convex Perimeter", "Convexity",
        "MeanB", "MeanG", "MeanR", "CaruncleArea", "CaruncleCurvature"
    ]

def calcTraits(rgb_img, mask, scale_per_pixel=1.0):
    """
    计算豆类种子的形态和颜色特征
    参数:
    - rgb_img: RGB图像
    - mask: 二值掩码
    - scale_per_pixel: 每像素对应的真实世界尺度
    返回:
    - 特征值列表，顺序与 getHeaders() 对应
    """
    try:
        if rgb_img is None or mask is None or rgb_img.size == 0 or mask.size == 0:
            return [np.nan] * 15
        original_mask = mask.copy()
        morphological_traits = get_seed_morphological_traits(rgb_img, mask, scale_per_pixel)
        masked_rgb = cv2.bitwise_and(rgb_img, rgb_img, mask=mask)
        try:
            color_features = quantify_seed_color(masked_rgb, original_mask)
        except Exception as e:
            print(f"Error in color quantification: {e}")
            color_features = None
        extracted_caruncle_region = get_caruncle_area(masked_rgb)
        if extracted_caruncle_region is not None:
            caruncleCurvature, caruncleArea = get_caruncle_curvature_and_area(extracted_caruncle_region)
        else:
            caruncleCurvature, caruncleArea = np.nan, np.nan
        # 组装结果，顺序与 getHeaders 保持一致
        meanB = color_features["MeanB"] if color_features else np.nan
        meanG = color_features["MeanG"] if color_features else np.nan
        meanR = color_features["MeanR"] if color_features else np.nan

        return list(morphological_traits)[:10] + [meanB, meanG, meanR, caruncleArea, caruncleCurvature]
    except Exception as e:
        print(f"Error in seed trait analysis: {e}")
        import traceback
        traceback.print_exc()
        return [np.nan] * 15
import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage.measure import label

try:
    from pheno.outliers import contract_mask
except ImportError:
    def contract_mask(mask):
        binary = np.asarray(mask, np.uint8)
        eroded = cv2.erode(binary, np.ones((5, 5), np.uint8))
        return eroded if np.any(eroded) else binary

def quantify_seed_color(masked_rgb, mask):
    """
    对给定的种子图像进行颜色量化，并返回颜色特征
    参数:
    - masked_rgb: 带掩码的 RGB/BGR 图像 (H,W,C)
    - mask: 二值掩码 (H,W)
    返回:
    - 颜色特征字典，包含 meanB, meanG, meanR；无有效像素返回None
    """
    # 1. 提取掩码区域像素
    seed_pixels = masked_rgb[mask > 0]

    # 容错1：没有任何种子像素，直接返回
    if seed_pixels.size == 0:
        return None

    # 容错2：剔除含NaN的像素，避免均值为nan
    seed_pixels = seed_pixels[~np.isnan(seed_pixels).any(axis=-1)]
    if seed_pixels.size == 0:
        return None

    # 计算各通道均值，不再直接解包，用索引取值（最保守）
    channel_mean = np.mean(seed_pixels, axis=0)

    color_features = {}
    # 容错：通道不足3时，缺失通道填充0，防止索引越界
    color_features['MeanB'] = channel_mean[0] if channel_mean.shape[0] > 0 else np.nan
    color_features['MeanG'] = channel_mean[1] if channel_mean.shape[0] > 1 else np.nan
    color_features['MeanR'] = channel_mean[2] if channel_mean.shape[0] > 2 else np.nan

    return color_features
def get_caruncle_area(masked_rgb):
    # cv2.imwrite("masked_rgb.png", masked_rgb)
    h, w = masked_rgb.shape[:2]

    # 判断是否有透明通道
    if masked_rgb.shape[2] == 4:
        alpha = masked_rgb[:, :, 3]
        foreground_mask = (alpha > 0).astype(np.uint8)
        flat_img = masked_rgb[:, :, :3].reshape((-1, 3))
        foreground_mask_flat = foreground_mask.flatten()
        non_transparent_idx = np.where(foreground_mask_flat > 0)[0]
        non_transparent_pixels = flat_img[non_transparent_idx]
    else:
        # 没有透明通道时，全部视为前景
        flat_img = masked_rgb.reshape((-1, 3))
        non_transparent_idx = np.arange(h * w)
        non_transparent_pixels = flat_img

    if len(non_transparent_pixels) == 0:
        return None

    # 聚类
    kmeans = KMeans(n_clusters=2, n_init='auto', random_state=0)
    labels = kmeans.fit_predict(non_transparent_pixels)

    # 将标签还原成图像形式
    clustered_map = np.zeros((h * w), dtype=np.int32)
    clustered_map[non_transparent_idx] = labels
    clustered_map = clustered_map.reshape((h, w))

    # 获取两个类的面积和连通性
    mask0 = (clustered_map == 0).astype(np.uint8)
    mask1 = (clustered_map == 1).astype(np.uint8)

    # 只保留最大连通区域
    def get_largest_conn_region(mask):
        labeled = label(mask, connectivity=2)
        if labeled.max() == 0:
            return np.zeros_like(mask)
        max_label = np.argmax(np.bincount(labeled.flat)[1:]) + 1
        return (labeled == max_label).astype(np.uint8)

    mask0_conn = get_largest_conn_region(mask0)
    mask1_conn = get_largest_conn_region(mask1)

    area0 = np.sum(mask0_conn)
    area1 = np.sum(mask1_conn)

    if area0 < area1:
        minor_mask = mask0_conn
    else:
        minor_mask = mask1_conn

    # 如果面积太大，认为无 caruncle
    if np.sum(minor_mask) > len(non_transparent_idx) / 4 or np.sum(minor_mask) < len(non_transparent_idx) / 20:
        return None

    # 构造返回图像
    masked_caruncle = np.zeros_like(masked_rgb[:, :, :3])
    masked_caruncle[minor_mask > 0] = masked_rgb[:, :, :3][minor_mask > 0]

    return masked_caruncle

def get_caruncle_curvature_and_area(caruncle_area):
    gray = cv2.cvtColor(caruncle_area, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.nan, np.nan
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    max_dist = 0
    pt1 = pt2 = None
    for i in range(len(cnt)):
        for j in range(i + 1, len(cnt)):
            dist = np.linalg.norm(cnt[i][0] - cnt[j][0])
            if dist > max_dist:
                max_dist = dist
                pt1, pt2 = cnt[i][0], cnt[j][0]
    if pt1 is None or pt2 is None or max_dist == 0:
        return np.nan, area
    chord_vec = pt2 - pt1
    chord_unit = chord_vec / np.linalg.norm(chord_vec)
    normal = np.array([-chord_unit[1], chord_unit[0]])
    max_dev = 0
    for pt in cnt:
        vec = pt[0] - pt1
        dev = abs(np.dot(vec, normal))
        if dev > max_dev:
            max_dev = dev
    curvature = max_dev / max_dist
    return curvature, area

def get_seed_morphological_traits(rgbImg, mask, scale_per_pixel=1.0):
    mask = contract_mask(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [np.nan] * 10
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour) * (scale_per_pixel ** 2)
    perimeter = cv2.arcLength(contour, closed=True) * scale_per_pixel
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter != 0 else np.nan
    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        major_axis = max(ellipse[1]) * scale_per_pixel
        minor_axis = min(ellipse[1]) * scale_per_pixel
    else:
        major_axis, minor_axis = np.nan, np.nan
    aspect_ratio = major_axis / minor_axis if minor_axis != 0 else np.nan
    hull = cv2.convexHull(contour)
    convex_area = cv2.contourArea(hull) * (scale_per_pixel ** 2)
    convex_perimeter = cv2.arcLength(hull, closed=True) * scale_per_pixel
    solidity = area / convex_area if convex_area != 0 else np.nan
    convexity = convex_perimeter / perimeter if perimeter != 0 else np.nan
    return [
        area, perimeter, major_axis, minor_axis, aspect_ratio,
        solidity, circularity, convex_area, convex_perimeter, convexity
    ]

def analyze_seed_image(rgb_path, mask_path, scale_per_pixel=1.0):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    rgb = cv2.imread(rgb_path)
    if mask is None or rgb is None:
        raise FileNotFoundError(f"Cannot read {rgb_path} or {mask_path}")
    original_mask = mask.copy()
    masked_rgb = cv2.bitwise_and(rgb, rgb, mask=mask)
    extracted_caruncle_region = get_caruncle_area(masked_rgb)
    if extracted_caruncle_region is not None:
        caruncleCurvature, caruncleArea = get_caruncle_curvature_and_area(extracted_caruncle_region)
    else:
        caruncleCurvature, caruncleArea = np.nan, np.nan
    morphological_traits = get_seed_morphological_traits(rgb, mask, scale_per_pixel)
    color_features = quantify_seed_color(masked_rgb, original_mask)
    result = {
        "morphological_traits": morphological_traits,
        "color_features": color_features,
        "caruncle_area": caruncleArea,
        "caruncle_curvature": caruncleCurvature,
    }
    return result

# 用法示例：
# result = analyze_seed_image('seed.jpg', 'seed_Object01.png')
# print(result)
