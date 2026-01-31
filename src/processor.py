import cv2
import numpy as np

class PalmLineExtractor:
    """
    掌纹提取器，负责图像增强和主线提取
    """
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))

    def preprocess_image(self, roi: np.ndarray) -> np.ndarray:
        """
        对 ROI 进行预处理：灰度化 -> CLAHE -> 双边滤波。
        """
        # 转灰度
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
    
        filtered = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)
        
        return filtered

    def extract_lines(self, roi: np.ndarray, landmarks_roi: list) -> dict:
        """
        提取三大主线
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
           
            mask = self._generate_mask(line_name, landmarks_roi, (width, height))
            
            masked_img = cv2.bitwise_and(preprocessed, preprocessed, mask=mask)
            
            block_size = int(width / 30)
            if block_size % 2 == 0: block_size += 1
            thresh_adapt = cv2.adaptiveThreshold(
                masked_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, block_size, 2
            )
            
            ksize_hat = int(width / 50) 
            kernel_hat = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize_hat, ksize_hat))
            blackhat = cv2.morphologyEx(masked_img, cv2.MORPH_BLACKHAT, kernel_hat)
           
            _, thresh_hat = cv2.threshold(blackhat, 15, 255, cv2.THRESH_BINARY)
            
            combined = cv2.bitwise_or(thresh_adapt, thresh_hat)
            combined = cv2.bitwise_and(combined, combined, mask=mask)
            
            k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            opened = cv2.morphologyEx(combined, cv2.MORPH_OPEN, k_open)
            
            close_ksize = int(width / 80)
            if close_ksize < 3: close_ksize = 3
            k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k_close)
            
            skeleton = self._skeletonize(closed)
            
            contours, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            valid_contours = []
            max_len = 0
            
            sorted_contours = sorted(contours, key=lambda c: cv2.arcLength(c, False), reverse=True)
            min_valid_len = max(width, height) * 0.08 # 最小有效长度阈值
            
            for i, cnt in enumerate(sorted_contours):
                if i >= 3: break # 最多取前3段
                length = cv2.arcLength(cnt, False)
                if length > min_valid_len:
                    valid_contours.append(cnt)
                    if length > max_len: max_len = length
            
            # 连接断裂的线段
            merged_contours = self._merge_contours(valid_contours, width, height)
            
            # 只保留最长的一条
            final_contours = []
            if merged_contours:
                # 按长度排序
                merged_contours.sort(key=lambda c: cv2.arcLength(c, False), reverse=True)
                final_contours = [merged_contours[0]]
            
            # 计算总长度用于置信度
            total_len = sum(cv2.arcLength(c, False) for c in final_contours)
            
            roi_diag = np.sqrt(width**2 + height**2)
            confidence = min(1.0, total_len / (roi_diag * 0.4)) if roi_diag > 0 else 0
            
            results[line_name] = {
                'contours': final_contours, 
                'confidence': confidence,
                'color': lines_info[line_name]['color']
            }
            
        return results

    def _merge_contours(self, contours, width, height):
        """
        连接断裂的线段
        """
        if not contours:
            return []
            
        def dist(p1, p2):
            return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
            
        # 将所有 contour 转换为点列表
        segments = []
        for cnt in contours:
            pts = [pt[0].tolist() for pt in cnt]
            segments.append(pts)
            
        # 按长度降序排列，以最长线为基准
        segments.sort(key=len, reverse=True)
        
        merged_paths = []
        
        while segments:
            # 取出当前最长的一段作为基准路径
            current_path = segments.pop(0)
            
            changed = True
            while changed:
                changed = False
                best_idx = -1
                best_score = float('inf')
                best_action = None # 'append', 'prepend', 'append_rev', 'prepend_rev'
                
                head = np.array(current_path[0])
                tail = np.array(current_path[-1])
                
                # 寻找剩余片段中最近的一个
                for i, seg in enumerate(segments):
                    s_head = np.array(seg[0])
                    s_tail = np.array(seg[-1])
                    
                    # 计算四种连接方式的距离
                    d_tail_head = dist(tail, s_head) 
                    d_tail_tail = dist(tail, s_tail) 
                    d_head_tail = dist(head, s_tail)
                    d_head_head = dist(head, s_head) 
                    
                    # 检查是否满足连接阈值
                    threshold = max(width, height) * 0.15
                    
                    if d_tail_head < best_score and d_tail_head < threshold:
                        best_score = d_tail_head
                        best_idx = i
                        best_action = 'append'
                    if d_tail_tail < best_score and d_tail_tail < threshold:
                        best_score = d_tail_tail
                        best_idx = i
                        best_action = 'append_rev'
                    if d_head_tail < best_score and d_head_tail < threshold:
                        best_score = d_head_tail
                        best_idx = i
                        best_action = 'prepend'
                    if d_head_head < best_score and d_head_head < threshold:
                        best_score = d_head_head
                        best_idx = i
                        best_action = 'prepend_rev'
                
                # 如果找到了合适的连接对象
                if best_idx != -1:
                    seg = segments.pop(best_idx)
                    
                    if best_action == 'append':
                        current_path.extend(seg)
                    elif best_action == 'append_rev':
                        current_path.extend(seg[::-1])
                    elif best_action == 'prepend':
                        current_path = seg + current_path
                    elif best_action == 'prepend_rev':
                        current_path = seg[::-1] + current_path
                        
                    changed = True # 继续尝试连接更多
            
            merged_paths.append(current_path)
            
        # 转换回 numpy
        result_contours = []
        for path in merged_paths:
            if len(path) > 1:
                cnt = np.array(path, dtype=np.int32).reshape((-1, 1, 2))
                result_contours.append(cnt)
                
        return result_contours

    def _skeletonize(self, img):
        """
        基于形态学的骨架提取算法
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
            p0 = landmarks[0]  
            p1 = landmarks[1]   
            p2 = landmarks[2]  
            p5 = landmarks[5]  
            p9 = landmarks[9]  
            p13 = landmarks[13]
            p17 = landmarks[17] 
            
            # 计算虎口中间点
            web_point = ((p2[0] + p5[0]) // 2, (p2[1] + p5[1]) // 2)
            
            # 手掌中心
            center_palm = ((p0[0] + p5[0] + p17[0]) // 3, (p0[1] + p5[1] + p17[1]) // 3)

            if line_name == 'life_line':
                inner_point = ((center_palm[0] + p0[0])//2, (center_palm[1] + p0[1])//2)
                
                pts = np.array([
                    p0, p1, p2, 
                    web_point, 
                    inner_point
                ], dtype=np.int32)
                points = [pts]

            elif line_name == 'head_line':
                start_point = web_point
                
                end_region_top = (p17[0], p17[1] + int(shape[1]*0.1))
                end_region_bottom = (p17[0], p17[1] + int(shape[1]*0.3))
                
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
