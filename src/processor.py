import cv2
import numpy as np

class PalmLineExtractor:
    """
    掌纹提取器，负责图像增强和主线提取。
    """
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))

    def preprocess_image(self, roi: np.ndarray) -> np.ndarray:
        """
        对 ROI 进行预处理：灰度化 -> 强增强(CLAHE) -> 高斯模糊。
        """
        # 转灰度
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 增强 CLAHE 参数，提高对比度
        # clipLimit 提高到 3.0，tileGridSize 稍微调小以适应细节
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 高斯模糊去除皮肤噪点，保留主线
        h, w = roi.shape[:2]
        ksize = int(w / 120)
        if ksize % 2 == 0: ksize += 1
        ksize = max(3, ksize)
        
        blurred = cv2.GaussianBlur(enhanced, (ksize, ksize), 0)
        
        return blurred

    def extract_lines(self, roi: np.ndarray, landmarks_roi: list) -> dict:
        """
        提取三大主线：生命线、感情线、智慧线。
        采用“自适应阈值 + 骨架提取 + 多段拟合”方案。
        """
        preprocessed = self.preprocess_image(roi)
        height, width = preprocessed.shape
        
        results = {}
        lines_info = {
            'life_line': {'color': (0, 0, 255)},   # 红色
            'heart_line': {'color': (255, 0, 0)},  # 蓝色
            'head_line': {'color': (0, 255, 0)}    # 绿色
        }

        for line_name in lines_info.keys():
            # 1. 生成 Mask (收紧范围，减少杂纹干扰)
            mask = self._generate_mask(line_name, landmarks_roi, (width, height))
            
            # 2. 图像增强与二值化 (针对暗纹路)
            masked_img = cv2.bitwise_and(preprocessed, preprocessed, mask=mask)
            
            block_size = int(width / 25) # 稍微调小 block_size，对细纹更敏感
            if block_size % 2 == 0: block_size += 1
            block_size = max(11, block_size)
            
            binary = cv2.adaptiveThreshold(
                masked_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, block_size, 2
            )
            
            binary = cv2.bitwise_and(binary, binary, mask=mask)
            
            # 3. 形态学去噪与连接 (关键步骤：强化连接)
            # 开运算去除小噪点
            k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open)
            
            # 闭运算连接断裂
            # 对于生命线和智慧线，如果核太大，容易把两条线粘连在一起，导致骨架提取偏差
            # 因此针对不同线调整核大小
            if line_name in ['life_line', 'head_line']:
                 close_ksize = int(width / 100) # 较小，避免粘连
            else:
                 close_ksize = int(width / 60)  # 感情线通常较远，可以使用大核
                 
            if close_ksize < 3: close_ksize = 3
            k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (close_ksize, close_ksize))
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k_close)
            
            # 4. 骨架提取
            skeleton = self._skeletonize(closed)
            
            # 5. 提取最长路径 (只取一条)
            contours, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            valid_contours = []
            max_len = 0
            best_contour = None
            
            # 策略：只找最长的那一条，解决“乱”的问题
            for cnt in contours:
                length = cv2.arcLength(cnt, False)
                if length > max_len:
                    max_len = length
                    best_contour = cnt
            
            # 过滤掉太短的噪音
            min_valid_len = max(width, height) * 0.1 # 至少 10% 长度
            if best_contour is not None and max_len > min_valid_len:
                valid_contours.append(best_contour)
            
            # 6. 计算置信度
            roi_diag = np.sqrt(width**2 + height**2)
            confidence = min(1.0, max_len / (roi_diag * 0.3)) if roi_diag > 0 else 0
            
            results[line_name] = {
                'contours': valid_contours, 
                'confidence': confidence,
                'color': lines_info[line_name]['color']
            }
            
        return results

    def _skeletonize(self, img):
        """
        基于形态学的骨架提取算法 (通用，不依赖 opencv-contrib)。
        重复执行 腐蚀 和 开运算 直到图像为空。
        """
        img = img.copy()
        skel = np.zeros(img.shape, np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        
        while True:
            open_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
            temp = cv2.subtract(img, open_img)
            eroded = cv2.erode(img, element)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded.copy()
            if cv2.countNonZero(img) == 0:
                break
        return skel

    def _generate_mask(self, line_name: str, landmarks: list, shape: tuple) -> np.ndarray:
        """
        根据关键点生成特定主线的掩膜
        """
        mask = np.zeros((shape[1], shape[0]), dtype=np.uint8)
        points = []
        
        try:
            # 通用辅助点
            p0 = landmarks[0]   # Wrist
            p1 = landmarks[1]   # Thumb CMC
            p2 = landmarks[2]   # Thumb MCP
            p5 = landmarks[5]   # Index MCP
            p9 = landmarks[9]   # Middle MCP
            p13 = landmarks[13] # Ring MCP
            p17 = landmarks[17] # Pinky MCP
            
            # 计算虎口中间点 (Web point)
            web_point = ((p2[0] + p5[0]) // 2, (p2[1] + p5[1]) // 2)
            
            # 手掌中心 (近似)
            center_palm = ((p0[0] + p5[0] + p17[0]) // 3, (p0[1] + p5[1] + p17[1]) // 3)

            if line_name == 'life_line':
                # 生命线：围绕大鱼际，避开掌心太远的地方
                # 关键点：Wrist -> Thumb bases -> Web Point -> Slightly towards Center -> Wrist
                
                # 往掌心内收一点点，但不要太多
                inner_point = ((center_palm[0] + p0[0])//2, (center_palm[1] + p0[1])//2)
                
                pts = np.array([
                    p0, p1, p2, 
                    web_point, 
                    inner_point
                ], dtype=np.int32)
                points = [pts]

            elif line_name == 'head_line':
                # 智慧线：横穿手掌中部
                # 起点：虎口附近 (Web Point)
                # 终点：小鱼际上方 (Pinky side)
                
                # 稍微下移一点 web_point 以避开生命线起点
                start_point = web_point
                
                # 终点区域：p17 下方一点
                end_region_top = (p17[0], p17[1] + int(shape[1]*0.1))
                end_region_bottom = (p17[0], p17[1] + int(shape[1]*0.3))
                
                # 掌心下方界限 (避免切到感情线)
                # 感情线通常在 p17-p5 连线下方
                # 智慧线在感情线下方
                
                pts = np.array([
                    start_point,
                    p5, # 包含一点食指根部以防线太高
                    end_region_top,
                    end_region_bottom,
                    center_palm # 回到掌心
                ], dtype=np.int32)
                
                points = [pts]
                
            elif line_name == 'heart_line':
                # 感情线：最上方，横穿
                # 起点：小指下方
                # 终点：食指/中指下方
                
                # 下边界：大概在 p17 下方 20-30% 掌宽处
                palm_width = np.linalg.norm(np.array(p5) - np.array(p17))
                offset = int(palm_width * 0.35) 
                
                pts = np.array([
                    p17, p13, p9, p5, # 指根连线
                    (p5[0], p5[1] + offset), # 食指下方偏移
                    (p17[0], p17[1] + offset) # 小指下方偏移
                ], dtype=np.int32)
                points = [pts]
                
            if points:
                cv2.fillPoly(mask, points, 255)
                
        except IndexError:
            pass
            
        return mask
