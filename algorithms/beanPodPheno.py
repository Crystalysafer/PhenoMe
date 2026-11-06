import cv2
import math
import numpy as np
from skimage.morphology import skeletonize, medial_axis
from skimage.graph import route_through_array
from skimage.measure import label, regionprops
from scipy.spatial import ConvexHull
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import euclidean
from shapely.geometry import Polygon


def getHeaders():
    """返回豆类分析的特征标题"""
    return [
        "Area", "Curvedness", "Width", "Length", "Sharpness1", "Sharpness2",
        "Edge1", "Edge2", "SkeletonLength", "MeanB", "MeanG", "MeanR"
    ]


def calcTraits(rgb_img, mask, scale_per_pixel=1.0):
    """
    计算豆类的形态和颜色特征
    
    参数:
    - rgb_img: RGB图像
    - mask: 二值掩码
    - scale_per_pixel: 每像素对应的真实世界尺度
    
    返回:
    - 特征值列表
    """
    try:
        # 输入验证
        if rgb_img is None or mask is None:
            return [np.nan] * 11
            
        if rgb_img.size == 0 or mask.size == 0:
            return [np.nan] * 11
        
        # 获取掩码区域的图像
        masked_img = rgb_img.copy()
        masked_img[mask == 0] = 0
        
        # 分析结果 - 直接使用mask进行形态分析
        results = analyze_image_internal(masked_img, mask, scale_per_pixel)
        
        # 按照headers的顺序返回结果，确保所有值都是标准Python数字类型
        def safe_get(d, key, default=np.nan):
            val = d.get(key, default)
            if val is None or (isinstance(val, (int, float, np.integer, np.floating)) and (np.isnan(val) or np.isinf(val))):
                return default
            # 转换numpy类型为Python标准类型
            if isinstance(val, (np.integer, np.floating)):
                return float(val) if isinstance(val, np.floating) else int(val)
            return val
        
        return [
            safe_get(results, "Area", np.nan),
            safe_get(results, "Curvedness", np.nan),
            safe_get(results, "Width", np.nan),
            safe_get(results, "Length", np.nan),
            safe_get(results, "Sharpness1", np.nan),
            safe_get(results, "Sharpness2", np.nan),
            safe_get(results, "Edge1", np.nan),
            safe_get(results, "Edge2", np.nan),
            safe_get(results, "SkeletonLength", np.nan),
            safe_get(results, "MeanB", np.nan),
            safe_get(results, "MeanG", np.nan),
            safe_get(results, "MeanR", np.nan)
        ]
    except Exception as e:
        print(f"Error in bean analysis: {e}")
        import traceback
        traceback.print_exc()
        # 返回默认值
        return [np.nan] * 11


# ==================== 内部分析函数 ====================

def keep_largest_path(skeleton):
    labeled = label(skeleton)
    regions = sorted(regionprops(labeled), key=lambda r: r.area, reverse=True)
    if not regions:
        return skeleton
    largest_label = regions[0].label
    return (labeled == largest_label).astype(np.uint8)


def compute_path_length(path):
    if len(path) < 2:
        return np.nan
    diffs = np.diff(path, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    return np.sum(segment_lengths)


def get_normals_improved(path, step=1, smooth_sigma=None):
    path = np.array(path)
    if len(path) < 3:
        return []
    
    if smooth_sigma is not None:
        try:
            path = np.column_stack([
                gaussian_filter1d(path[:, 0], sigma=smooth_sigma),
                gaussian_filter1d(path[:, 1], sigma=smooth_sigma),
            ])
        except:
            pass  # 如果平滑失败，使用原始路径
            
    normals = []
    for i in range(step, len(path) - step):
        p_prev = path[i - step]
        p_next = path[i + step]
        tangent = p_next - p_prev
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm > 1e-6:
            tangent = tangent / tangent_norm
            normal = np.array([-tangent[1], tangent[0]])
            normals.append(normal)
        else:
            normals.append(np.array([0, 1]))  # 默认法线
            
    if len(normals) > 0:
        normals = [normals[0]] * step + normals + [normals[-1]] * step
    return normals


def find_circle_contour_intersections(center, contour_points, radius):
    """找到以center为中心，半径为radius的圆与轮廓的交点"""
    cx, cy = center
    intersections = []
    for i, (x, y) in enumerate(contour_points):
        if abs(np.hypot(x - cx, y - cy) - radius) < 1.5:
            intersections.append((i, (x, y)))
    if len(intersections) > 1:
        if intersections[0][0] + len(contour_points) - intersections[-1][0] < len(contour_points) // 2:
            for i in range(len(intersections) - 1):
                if intersections[i][0] < len(contour_points) // 2 and intersections[i + 1][0] > len(contour_points) // 2:
                    return [intersections[i], intersections[i + 1]]
        else:
            return [intersections[-1], intersections[0]]
    return [None, None]


def find_region_by_area(contour_points, path, normals, binary, area_threshold, max_steps=200):
    """通过面积阈值找到区域"""
    def find_contour_intersections(center, normal, binary, max_scan=200):
        x0, y0 = int(center[0]), int(center[1])
        found_a = found_b = None
        for d in range(1, max_scan):
            dx, dy = int(round(normal[0] * d)), int(round(normal[1] * d))
            xa, ya = x0 + dx, y0 + dy
            xb, yb = x0 - dx, y0 - dy
            if 0 <= xa < binary.shape[1] and 0 <= ya < binary.shape[0]:
                if binary[ya, xa] == 0 and found_a is None:
                    found_a = (xa, ya)
            if 0 <= xb < binary.shape[1] and 0 <= yb < binary.shape[0]:
                if binary[yb, xb] == 0 and found_b is None:
                    found_b = (xb, yb)
            if found_a and found_b:
                break
        return found_a, found_b

    for i in range(1, min(len(path), max_steps)):
        if i >= len(normals):
            break
        left, right = find_contour_intersections(path[i], normals[i], binary)
        if left is None or right is None:
            continue
        left_idx = np.argmin(np.linalg.norm(contour_points - np.array(left), axis=1))
        right_idx = np.argmin(np.linalg.norm(contour_points - np.array(right), axis=1))
        if abs(left_idx - right_idx) < len(contour_points) / 2:
            if left_idx < right_idx:
                contour_segment = contour_points[left_idx:right_idx + 1]
            else:
                contour_segment = contour_points[right_idx:left_idx + 1]
        else:
            if left_idx < right_idx:
                contour_segment = np.concatenate([contour_points[right_idx:], contour_points[:left_idx + 1]])
            else:
                contour_segment = np.concatenate([contour_points[left_idx:], contour_points[:right_idx + 1]])
        region_pts = [left] + contour_segment.tolist() + [right]
        if len(region_pts) >= 3:
            try:
                polygon = Polygon(region_pts)
                if polygon.area >= area_threshold:
                    return left, right, compute_path_length(path[:i])
            except:
                continue
    return None, None, None


def analyze_BW(mask: np.ndarray, scale_per_pixel=1.0):
    """分析二值掩码的形态特征"""
    # mask已经是二值图像了，直接使用
    binary = mask.astype(np.uint8)
    
    if np.sum(binary) < 100:  # 面积太小
        return {
            "Curvedness": np.nan, "Area": np.nan, "Width": np.nan, "Length": np.nan,
            "Sharpness1": np.nan, "Sharpness2": np.nan, "SkeletonLength": np.nan,
            "Edge1": np.nan, "Edge2": np.nan, 
        }, [], []

    try:
        # 骨架化并保留主路径
        skeleton = medial_axis(binary)
        skeleton = keep_largest_path(skeleton).astype(np.uint8)

        ys, xs = np.where(skeleton == 1)
        if len(xs) < 2:
            return {
            "Curvedness": np.nan, "Area": np.nan, "Width": np.nan, "Length": np.nan,
            "Sharpness1": np.nan, "Sharpness2": np.nan, "SkeletonLength": np.nan,
            "Edge1": np.nan, "Edge2": np.nan, 
        }, [], []
            
        skeleton_points = list(zip(xs, ys))
        skel_pts = np.array(skeleton_points)

        # 获取轮廓点
        contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return {
            "Curvedness": np.nan, "Area": np.nan, "Width": np.nan, "Length": np.nan,
            "Sharpness1": np.nan, "Sharpness2": np.nan, "SkeletonLength": np.nan,
            "Edge1": np.nan, "Edge2": np.nan, 
        }, [], []
            
        contour_points = np.vstack(contours).squeeze()

        # 骨架最长点对
        if len(skel_pts) < 3:
            hull_pts = skel_pts
        else:
            hull = ConvexHull(skel_pts)
            hull_pts = skel_pts[hull.vertices]
            
        max_dist = 0
        skeleton_start = skeleton_end = hull_pts[0]
        for i in range(len(hull_pts)):
            for j in range(i + 1, len(hull_pts)):
                d = np.linalg.norm(hull_pts[i] - hull_pts[j])
                if d > max_dist:
                    max_dist = d
                    skeleton_start = tuple(hull_pts[i])
                    skeleton_end = tuple(hull_pts[j])

        if skeleton_start[1] > skeleton_end[1]:
            skeleton_start, skeleton_end = skeleton_end, skeleton_start

        start_idx = np.argmin(np.linalg.norm(contour_points - np.array(skeleton_start), axis=1))
        contour_points = np.roll(contour_points, -start_idx, axis=0)
        start = contour_points[0]
        end_idx = np.argmin(np.linalg.norm(contour_points - np.array(skeleton_end), axis=1))
        end = contour_points[end_idx]

        # 路径搜索 - 修复边界检查
        skeleton_dilated = cv2.dilate(skeleton, np.ones((3, 3), np.uint8), iterations=1)
        cost = np.where(skeleton_dilated == 1, 1.0, 1000.0)
        ys, xs = np.where(skeleton_dilated)
        if len(xs) == 0 or len(ys) == 0:
            return {
            "Curvedness": np.nan, "Area": np.nan, "Width": np.nan, "Length": np.nan,
            "Sharpness1": np.nan, "Sharpness2": np.nan, "SkeletonLength": np.nan,
            "Edge1": np.nan, "Edge2": np.nan, 
        }, [], []
            
        min_x, max_x = xs.min(), xs.max()
        min_y, max_y = ys.min(), ys.max()
        cost_crop = cost[min_y:max_y + 1, min_x:max_x + 1]
        
        # 确保起点和终点在裁剪后的数组边界内
        start_crop_y = max(0, min(start[1] - min_y, cost_crop.shape[0] - 1))
        start_crop_x = max(0, min(start[0] - min_x, cost_crop.shape[1] - 1))
        end_crop_y = max(0, min(end[1] - min_y, cost_crop.shape[0] - 1))
        end_crop_x = max(0, min(end[0] - min_x, cost_crop.shape[1] - 1))
        
        start_crop = (start_crop_y, start_crop_x)
        end_crop = (end_crop_y, end_crop_x)
        
        # 检查起点和终点是否有效
        if (start_crop[0] < 0 or start_crop[0] >= cost_crop.shape[0] or 
            start_crop[1] < 0 or start_crop[1] >= cost_crop.shape[1] or
            end_crop[0] < 0 or end_crop[0] >= cost_crop.shape[0] or 
            end_crop[1] < 0 or end_crop[1] >= cost_crop.shape[1]):
            # 如果边界检查失败，返回基本的测量值
            path_length = euclidean(skeleton_start, skeleton_end)
            return {
                "Curvedness": np.nan, "Area": cv2.contourArea(contour_points) * (scale_per_pixel ** 2),
                "Sharpness1": np.nan, "Sharpness2": np.nan,
                "Width": np.nan, "Length": np.nan, "Edge1": np.nan, "Edge2": np.nan,
                "SkeletonLength": path_length * scale_per_pixel
            }, [], []
        
        # 如果起点和终点相同，直接返回
        if start_crop == end_crop:
            return {
                "Curvedness": np.nan, "Area": cv2.contourArea(contour_points) * (scale_per_pixel ** 2),
                "Sharpness1": np.nan, "Sharpness2": np.nan, "Width": np.nan, "Length": np.nan,
                "Edge1": np.nan, "Edge2": np.nan, "SkeletonLength": np.nan
            }, [], []
        
        path, _ = route_through_array(cost_crop, start_crop, end_crop, fully_connected=True)
        path = [(x + min_x, y + min_y) for y, x in path]

        # 法线与宽度计算
        step_size = max(1, len(path)//100)
        smooth_sigma = min(100, max(1, len(path)//10))
        normals = get_normals_improved(path, step=step_size, smooth_sigma=smooth_sigma)
        
        max_width = np.nan
        max_info = None
        half_scan = max(10, int(compute_path_length(path) // 3))
        
        # 确保normals列表长度与path匹配
        if len(normals) != len(path):
            # 如果法线长度不匹配，重新计算或补齐
            if len(normals) == 0:
                normals = [np.array([0, 1])] * len(path)  # 默认法线向上
            elif len(normals) < len(path):
                # 补齐法线
                last_normal = normals[-1] if normals else np.array([0, 1])
                normals.extend([last_normal] * (len(path) - len(normals)))
        
        for i in range(1, min(len(path) - 1, len(normals))):
            x0, y0 = path[i]
            if i >= len(normals):
                continue
            normal = normals[i]
            found_a = found_b = None
            for d in range(1, half_scan):
                dx, dy = int(normal[0] * d), int(normal[1] * d)
                xa, ya = int(x0 + dx), int(y0 + dy)
                xb, yb = int(x0 - dx), int(y0 - dy)
                if 0 <= ya < binary.shape[0] and 0 <= xa < binary.shape[1] and binary[ya, xa] == 0 and found_a is None:
                    found_a = (xa, ya)
                if 0 <= yb < binary.shape[0] and 0 <= xb < binary.shape[1] and binary[yb, xb] == 0 and found_b is None:
                    found_b = (xb, yb)
                if found_a and found_b:
                    width = np.linalg.norm(np.array(found_a) - np.array(found_b))
                    if np.isnan(max_width) or width > max_width:
                        max_width = width
                        max_info = {"width": width, "center": (x0, y0), "left": found_a, "right": found_b}
                    break
        
        # 如果没有找到有效宽度，使用备选方法
        if np.isnan(max_width) and len(contour_points) > 0:
            # 使用轮廓的边界框作为宽度估计
            x_coords = contour_points[:, 0]
            y_coords = contour_points[:, 1]
            max_width = max(np.max(x_coords) - np.min(x_coords), np.max(y_coords) - np.min(y_coords)) * 0.5

        # 面积计算
        try:
            area = Polygon(contour_points).area
        except:
            area = cv2.contourArea(contour_points)
        
        path_length = compute_path_length(path)
        straight_distance = euclidean(path[0], path[-1])
        curvedness = path_length / straight_distance if straight_distance > 0 else 1
        

        # 面积与角度分析
        area_threshold = area * 0.01
        left_start, right_start, depth_start = find_region_by_area(contour_points, path, normals, binary, area_threshold)
        left_end, right_end, depth_end = find_region_by_area(contour_points, path[::-1], normals[::-1], binary, area_threshold)
        
        # 确保深度值不为None
        if depth_start is None:
            depth_start = np.nan
        if depth_end is None:
            depth_end = np.nan

        # 找圆与轮廓交点并绘制路径片段
        radius = 10
        start1, end2 = find_circle_contour_intersections(start, contour_points, radius)
        start2, end1 = find_circle_contour_intersections(end, contour_points, radius)

        path_cw = []
        path_ccw = []
        if start1 and end1 and start1[0] is not None and end1[0] is not None:
            # 确保索引有效
            start_idx = max(0, min(start1[0], len(contour_points) - 1))
            end_idx = max(0, min(end1[0], len(contour_points) - 1))
            if start_idx <= end_idx:
                path_cw = contour_points[start_idx:end_idx+1]
            else:
                path_cw = np.concatenate([contour_points[start_idx:], contour_points[:end_idx+1]])

        if start2 and end2 and start2[0] is not None and end2[0] is not None:
            # 确保索引有效
            start_idx = max(0, min(start2[0], len(contour_points) - 1))
            end_idx = max(0, min(end2[0], len(contour_points) - 1))
            if start_idx <= end_idx:
                path_ccw = contour_points[start_idx:end_idx+1]
            else:
                path_ccw = np.concatenate([contour_points[start_idx:], contour_points[:end_idx+1]])


        # 计算轮廓长度
        length_cw = compute_path_length(path_cw) if len(path_cw) > 1 else np.nan
        length_ccw = compute_path_length(path_ccw) if len(path_ccw) > 1 else np.nan

        # 应用尺度转换并确保所有值都是标准Python数字类型
        results = {
            "Curvedness": float(max(1.0, curvedness)) if curvedness and curvedness > 0 else 1.0,
            "Area": float(area * (scale_per_pixel ** 2)) if area and area > 0 else np.nan,
            "Sharpness1": float(depth_start * scale_per_pixel) if not np.isnan(depth_start) and depth_start > 0 else np.nan,
            "Sharpness2": float(depth_end * scale_per_pixel) if not np.isnan(depth_end) and depth_end > 0 else np.nan,
            "Width": float(max_info["width"] * scale_per_pixel) if max_info and max_info.get("width", 0) > 0 else (float(max_width * scale_per_pixel) if not np.isnan(max_width) else np.nan),
            "Edge1": float(length_cw * scale_per_pixel) if not np.isnan(length_cw) and length_cw > 0 else np.nan,
            "Edge2": float(length_ccw * scale_per_pixel) if not np.isnan(length_ccw) and length_ccw > 0 else np.nan,
            "SkeletonLength": float(path_length * scale_per_pixel) if not np.isnan(path_length) and path_length > 0 else np.nan,
            "Length": float(straight_distance * scale_per_pixel) if not np.isnan(straight_distance) and straight_distance > 0 else np.nan,
        }

        return results, path_cw, path_ccw

    except Exception as e:
        print(f"Error in BW analysis: {e}")
        return {
            "Curvedness": np.nan, "Area": np.nan, "Width": np.nan, "Length": np.nan,
            "Sharpness1": np.nan, "Sharpness2": np.nan, "SkeletonLength": np.nan,
            "Edge1": np.nan, "Edge2": np.nan, 
        }, [], []



def path_to_circle_mask(path: np.ndarray, shape: tuple, radius: int = 10) -> np.ndarray:
    """将路径点每个点画圆构成掩膜，用于颜色分析"""
    mask = np.zeros(shape, dtype=np.uint8)
    for pt in path:
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < shape[1] and 0 <= y < shape[0]:
            cv2.circle(mask, (x, y), radius, 1, -1)
    return mask

def analyze_RGB(rgb_img: np.ndarray, mask: np.ndarray):

    # 1. 提取掩码区域像素
    pod_pixels = rgb_img[mask > 0]

    # 容错1：没有任何种子像素，直接返回
    if pod_pixels.size == 0:
        return None

    # 容错2：剔除含NaN的像素，避免均值为nan
    pod_pixels = pod_pixels[~np.isnan(pod_pixels).any(axis=-1)]
    if pod_pixels.size == 0:
        return None

    # 计算各通道均值，不再直接解包，用索引取值（最保守）
    channel_mean = np.mean(pod_pixels, axis=0)

    color_features = {}
    # 容错：通道不足3时，缺失通道填充0，防止索引越界
    color_features['MeanB'] = channel_mean[0] if channel_mean.shape[0] > 0 else np.nan
    color_features['MeanG'] = channel_mean[1] if channel_mean.shape[0] > 1 else np.nan
    color_features['MeanR'] = channel_mean[2] if channel_mean.shape[0] > 2 else np.nan

    return color_features


def analyze_image_internal(rgb_img: np.ndarray, mask: np.ndarray, scale_per_pixel=1.0) -> dict:
    """内部分析函数"""
    # 形态分析 - 直接使用mask
    results_bw, _, _ = analyze_BW(mask, scale_per_pixel)
    
    # 颜色分析
    results_rgb = analyze_RGB(rgb_img, mask)
    # 合并结果
    results = {}
    results.update(results_bw)
    results.update(results_rgb)

    
    return results
