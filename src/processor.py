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
        灰度化 -> CLAHE -> 高斯模糊。
        """
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 增强对比度，突出纹理
        enhanced = self.clahe.apply(gray)
        
        # 去噪
        h, w = roi.shape[:2]
        ksize = int(w / 120)
        if ksize % 2 == 0: ksize += 1
        ksize = max(3, ksize)
        
        blurred = cv2.GaussianBlur(enhanced, (ksize, ksize), 0)
        
        return blurred

    def extract_lines(self, roi: np.ndarray, landmarks_roi: list) -> dict:
        """
        提取三大主线
        """
        preprocessed = self.preprocess_image(roi)
        height, width = preprocessed.shape
        
        results = {}
        lines_info = {
            'life_line': {'color': (0, 0, 255)}, 
            'heart_line': {'color': (255, 0, 0)},  
            'head_line': {'color': (0, 255, 0)}  
        }

        for line_name in lines_info.keys():
            mask = self._generate_mask(line_name, landmarks_roi, (width, height))
            
            # 边缘提取
            masked_img = cv2.bitwise_and(preprocessed, preprocessed, mask=mask)
            
            # 中值自适应
            v = np.median(masked_img[mask > 0]) if np.sum(mask) > 0 else 127
            sigma = 0.33
            lower = int(max(0, (1.0 - sigma) * v))
            upper = int(min(255, (1.0 + sigma) * v))
            
            lower = max(20, lower)
            upper = max(60, upper)
            
            edges = cv2.Canny(masked_img, lower, upper)
            
            # 形态学操作
            m_ksize = int(width / 200)
            if m_ksize < 3: m_ksize = 3
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (m_ksize, m_ksize))
            dilated = cv2.dilate(edges, kernel, iterations=1)
            closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
            
            # 骨架提取
            try:
                thinned = cv2.ximgproc.thinning(closed, thinningType=cv2.ximgproc.THINNING_GUOHALL)
            except Exception as e:
                print(f"Warning: Thinning failed, using original mask. {e}")
                thinned = closed
            
            # 主线拟合
            # 在骨架图上寻找轮廓，得到的将是贴合中心的线条
            contours, _ = cv2.findContours(thinned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            best_contour = None
            max_len = 0
            
            for cnt in contours:
                length = cv2.arcLength(cnt, False) # 不闭合
                if length > max_len:
                    max_len = length
                    best_contour = cnt
            
            # 计算置信度。长度 / ROI对角线长度
            roi_diag = np.sqrt(width**2 + height**2)
            confidence = min(1.0, max_len / (roi_diag * 0.5)) if roi_diag > 0 else 0
            
            results[line_name] = {
                'contour': best_contour,
                'confidence': confidence,
                'color': lines_info[line_name]['color']
            }
            
        return results

    def _generate_mask(self, line_name: str, landmarks: list, shape: tuple) -> np.ndarray:
        """
        根据关键点生成特定主线的掩膜
        """
        mask = np.zeros((shape[1], shape[0]), dtype=np.uint8)
        points = []
        
      
        try:
            if line_name == 'life_line':
                # 生命线
                p0 = landmarks[0] 
                p1 = landmarks[1] 
                p2 = landmarks[2]  
                p5 = landmarks[5]  
                
                # 手掌中心
                center_x = (landmarks[0][0] + landmarks[5][0] + landmarks[17][0]) // 3
                center_y = (landmarks[0][1] + landmarks[5][1] + landmarks[17][1]) // 3
                
                # 构建多边形
                pts = np.array([p0, p1, p2, (center_x, center_y)], dtype=np.int32)
                points = [pts]

            elif line_name == 'heart_line':
                # 感情线
                p5 = landmarks[5]
                p9 = landmarks[9]
                p13 = landmarks[13]
                p17 = landmarks[17]
                
                palm_width = np.linalg.norm(np.array(p5) - np.array(p17))
                offset = int(palm_width * 0.4) 
                
                pts = np.array([
                    p5, p9, p13, p17,
                    (p17[0], p17[1] + offset),
                    (p5[0], p5[1] + offset)
                ], dtype=np.int32)
                points = [pts]

            elif line_name == 'head_line':
                # 智慧线
                p2 = landmarks[2] 
                p5 = landmarks[5] 
                p17 = landmarks[17] 
                
                center_x = (landmarks[0][0] + landmarks[9][0]) // 2
                center_y = (landmarks[0][1] + landmarks[9][1]) // 2
                
                pts = np.array([
                    landmarks[2], 
                    landmarks[5],
                    (landmarks[17][0], landmarks[17][1] + 20), 
                    (center_x, center_y + 50) 
                ], dtype=np.int32)
                
                points = [pts]
                
            if points:
                cv2.fillPoly(mask, points, 255)
                
        except IndexError:
            # 若关键点缺失，返回全黑 Mask
            pass
            
        return mask
