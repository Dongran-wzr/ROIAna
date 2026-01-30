import cv2
import mediapipe as mp
import numpy as np

class HandDetector:
    """
    手掌检测器，用于检测手部并提取 ROI。
    """
    def __init__(self, static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def detect_and_crop(self, image: np.ndarray, padding_ratio: float = 0.2):
        """
        检测手掌并裁切 ROI 区域。
        """
        if image is None:
            return {
                'error_code': 1000,
                'suggestion': '输入图像无效，请检查文件路径或格式。'
            }

        height, width = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)

        if not results.multi_hand_landmarks:
            return {
                'error_code': 1001,
                'suggestion': '未检测到手掌，请确保手掌完整出现在画面中，背景尽量干净。'
            }

        # 获取第一个检测到的手
        hand_landmarks = results.multi_hand_landmarks[0]
        handedness = results.multi_handedness[0]
        
        # 提取关键点坐标
        landmark_indices = [0, 1, 5, 9, 13, 17] 
        points = []
        all_landmarks = [] 

        for idx, lm in enumerate(hand_landmarks.landmark):
            cx, cy = int(lm.x * width), int(lm.y * height)
            all_landmarks.append((cx, cy))
            if idx in landmark_indices:
                points.append((cx, cy))
        
        
        # 检查手掌占比
        points_np = np.array(points)
        x, y, w, h = cv2.boundingRect(points_np)
        roi_area = w * h
        img_area = width * height
        ratio = roi_area / img_area
        
        if ratio < 0.03: 
            return {
                'error_code': 1002,
                'suggestion': '检测到的手掌过小，请将手掌移近镜头。'
            }

        # 展开状态检测
        wrist = np.array(all_landmarks[0])
        middle_mcp = np.array(all_landmarks[9]) # 中指指根
        middle_tip = np.array(all_landmarks[12]) # 中指指尖
        
        dist_palm = np.linalg.norm(middle_mcp - wrist)
        dist_finger = np.linalg.norm(middle_tip - wrist)
        
        if dist_finger < dist_palm * 1.2: # 手指弯曲
             return {
                'error_code': 1003,
                'suggestion': '手掌似乎未完全展开，请张开手掌以获得最佳效果。'
            }

        # 边界检查
        margin = 10
        is_near_border = False
        for px, py in points:
            if px < margin or px > width - margin or py < margin or py > height - margin:
                is_near_border = True
                break

        # ROI 裁切
        pad_w = int(w * padding_ratio)
        pad_h = int(h * padding_ratio)
        
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(width, x + w + pad_w)
        y2 = min(height, y + h + pad_h)
        
        # 获取左右手信息
        original_label = handedness.classification[0].label
        label = "Right" if original_label == "Left" else "Left"
        score = handedness.classification[0].score
        
        # 裁切 ROI
        roi = image[y1:y2, x1:x2]
        
        # 映射到 ROI 坐标系
        roi_landmarks = []
        for (lx, ly) in all_landmarks:
            roi_landmarks.append((lx - x1, ly - y1))

        return {
            'roi': roi,
            'landmarks_original': all_landmarks, 
            'landmarks_roi': roi_landmarks,     
            'bbox': (x1, y1, x2, y2),
            'hand_info': {'label': label, 'score': score, 'is_open': True}
        }
