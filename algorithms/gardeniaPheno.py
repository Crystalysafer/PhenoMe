import cv2
import numpy as np

def getHeaders():
    return ["Area", "Length", "Width", "L", "A", "B"]

def remove_outliers_iqr(data):
    """使用 IQR 方法去除离群值"""
    # 计算第25百分位数 (Q1) 和第75百分位数 (Q3)
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    
    # 计算上下限
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # 过滤掉离群值
    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
    return filtered_data

def calcTraits(rgb, mask, scale_per_pixel=1.0):
    """
    自定义算法示例，现在支持尺度参数
    
    参数:
    - rgb: RGB图像
    - mask: 二值掩码
    - scale_per_pixel: 每像素对应的真实世界尺度
    
    返回:
    - 特征值列表
    """
    # 计算区域面积
    area_pixels = mask.sum()
    area_real = area_pixels * (scale_per_pixel ** 2)  # 转换为真实世界单位
    
    # 计算最小外接矩形的长和宽
    # 1. 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # 2. 选择最大轮廓
        contour = max(contours, key=cv2.contourArea)
        
        # 3. 获取最小外接矩形
        rect = cv2.minAreaRect(contour)
        
        # 4. 获取矩形的长和宽
        width, height = rect[1]  # rect[1] 包含长和宽
        
        # 5. 确保 width 是最小值，height 是最大值
        length = max(width, height)
        width = min(width, height)
    else:
        # 如果没有找到轮廓，长和宽为 0
        length = 0
        width = 0
    
    # 转换为真实世界单位
    length_real = length * scale_per_pixel
    width_real = width * scale_per_pixel

    # 转换为 CIELAB 色度空间
    lab_image = cv2.cvtColor(rgb, cv2.COLOR_BGR2LAB)
    
    # 计算掩码区域的 L*, A*, B* 均值
    masked_lab_image = cv2.bitwise_and(lab_image, lab_image, mask=mask)
    
    # 提取 L*, a*, b* 通道
    L = masked_lab_image[:, :, 0]
    A = masked_lab_image[:, :, 1]
    B = masked_lab_image[:, :, 2]
    
    # 去除离群值并计算 L*, A*, B* 的均值
    L_filtered = remove_outliers_iqr(L[L > 0])  # 去除掩码外的部分
    A_filtered = remove_outliers_iqr(A[A > 0])
    B_filtered = remove_outliers_iqr(B[B > 0])
    
    # 计算均值
    L_mean = np.mean(L_filtered)
    A_mean = np.mean(A_filtered)
    B_mean = np.mean(B_filtered)
    
    # 返回特征值
    return [area_real, length_real, width_real, L_mean, A_mean, B_mean]
