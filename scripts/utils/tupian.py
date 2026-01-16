import cv2
import numpy as np

def extract_red_mask(original_path, modified_path, output_path):
    # 1. 读取图片
    img1 = cv2.imread(original_path)
    img2 = cv2.imread(modified_path)

    if img1 is None or img2 is None:
        print("错误：无法加载图片，请检查路径。")
        return

    # --- 新增：处理尺寸不一致的问题 ---
    if img1.shape != img2.shape:
        print(f"检测到尺寸不匹配：图1 {img1.shape} vs 图2 {img2.shape}")
        # 将 img2 缩放到 img1 的尺寸 (宽, 高)
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # --- 新增：统一通道数（防止 PNG 的 Alpha 通道干扰） ---
    if img1.shape[2] == 4:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGRA2BGR)
    if img2.shape[2] == 4:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGRA2BGR)

    # 2. 计算两张图片的绝对差异
    diff = cv2.absdiff(img1, img2)

    # 3. 转到 HSV 空间进行颜色过滤
    hsv_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2HSV)

    # 红色范围
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv_diff, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_diff, lower_red2, upper_red2)
    full_mask = cv2.bitwise_or(mask1, mask2)

    # 4. 形态学处理：去除极小的噪点
    kernel = np.ones((3,3), np.uint8)
    full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_OPEN, kernel)

    # 5. 保存结果
    cv2.imwrite(output_path, full_mask)
    print(f"掩码已成功提取并保存至: {output_path}")

# 路径使用你报错信息中的路径
extract_red_mask('/root/co-tracker/assets/ge1.png', '/root/co-tracker/assets/ge2.jpg', '/root/co-tracker/mask_result.png')