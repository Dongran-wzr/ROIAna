import argparse
import cv2
import numpy as np
import sys
import os
import time
import json

from src.utils import load_image_with_exif, resize_to_1080p
from src.detector import HandDetector
from src.processor import PalmLineExtractor

def main():
    parser = argparse.ArgumentParser(description="掌纹识别项目 (Palm Recognition Project)")
    parser.add_argument("image_path", help="输入图片的路径 (JPG/PNG)")
    parser.add_argument("--output", help="输出结果图片的路径", default="output.jpg")
    parser.add_argument("--save-data", help="保存检测数据供人工矫正 (JSON)", action="store_true")
    args = parser.parse_args()

    image_path = args.image_path
    
    # 开始总计时
    total_start_time = time.time()
    
    # Step 1: 加载与预处理
    t0 = time.time()
    print(f"正在加载图片: {image_path}")
    original_img = load_image_with_exif(image_path)
    if original_img is None:
        print("错误: 无法加载图片")
        sys.exit(1)
        
    # 缩放到 1080p
    processed_img = resize_to_1080p(original_img)
    t1 = time.time()
    print(f"图片已缩放至: {processed_img.shape} (耗时: {t1 - t0:.4f}s)")

    # Step 2: 手掌检测
    print("正在检测手掌...")
    detector = HandDetector()
    detect_result = detector.detect_and_crop(processed_img)
    t2 = time.time()
    print(f"手掌检测完成 (耗时: {t2 - t1:.4f}s)")
    
    if 'error_code' in detect_result:
        print(f"检测失败 (代码 {detect_result['error_code']}): {detect_result['suggestion']}")
        # 即使失败也保存（原图），或直接退出
        sys.exit(1)
        
    roi = detect_result['roi']
    landmarks_roi = detect_result['landmarks_roi']
    bbox = detect_result['bbox'] # (x1, y1, x2, y2)
    x1, y1, _, _ = bbox
    
    print(f"检测到手掌: {detect_result['hand_info']}")
    hand_info = detect_result['hand_info']
    
    # Step 3: 掌纹提取
    print("正在提取掌纹主线...")
    extractor = PalmLineExtractor()
    lines_result = extractor.extract_lines(roi, landmarks_roi)
    t3 = time.time()
    print(f"掌纹提取完成 (耗时: {t3 - t2:.4f}s)")
    
    # 准备数据用于保存
    correction_data = {
        "image_path": os.path.abspath(args.output),
        "lines": {}
    }
    
    # Step 4: 结果绘制 (在缩放后的原图上)
    # 创建一个遮罩层用于绘制透明线条
    overlay = processed_img.copy()
    
    # 自适应线条粗细
    h, w = processed_img.shape[:2]
    line_thickness = max(2, int(w / 300))
    
    for line_name, data in lines_result.items():
        contour = data['contour']
        confidence = data['confidence']
        color = data['color']
        
        if contour is not None and confidence > 0.1: # 过滤低置信度
            print(f"  - {line_name}: 置信度 {confidence:.2f}")
            
            # 将 ROI 坐标系的轮廓映射回原图坐标系
            # contour shape is (N, 1, 2)
            contour_original = contour + [x1, y1]
            
            # 保存数据
            epsilon = 0.005 * cv2.arcLength(contour_original, False)
            approx = cv2.approxPolyDP(contour_original, epsilon, False)
            correction_data["lines"][line_name] = approx.tolist()
            
            # 绘制线条 (较粗，以便看清)
            cv2.drawContours(overlay, [contour_original], -1, color, line_thickness)
        else:
            print(f"  - {line_name}: 未检测到或置信度过低")
            correction_data["lines"][line_name] = []

    # 混合原图与线条层，实现透明效果
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, processed_img, 1 - alpha, 0, processed_img)
    
    # 绘制 ROI 框 (可选，方便调试)
    cv2.rectangle(processed_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 255), max(1, line_thickness // 2))
    
    # 显示左右手信息
    label_text = f"{hand_info['label']} Hand ({hand_info['score']:.2f})"
    if not hand_info.get('is_open', True):
        label_text += " [Not Fully Open]"
        
    cv2.putText(processed_img, label_text, (bbox[0], bbox[1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    
    # 保存结果
    output_path = args.output
    cv2.imwrite(output_path, processed_img)
    t4 = time.time()
    print(f"处理完成，结果已保存至: {output_path}")
    print(f"总耗时: {t4 - total_start_time:.4f}s (绘制与保存: {t4 - t3:.4f}s)")

    if args.save_data:
        json_path = os.path.splitext(output_path)[0] + ".json"
        with open(json_path, 'w') as f:
            json.dump(correction_data, f)
        print(f"数据已保存至: {json_path}")

if __name__ == "__main__":
    main()
