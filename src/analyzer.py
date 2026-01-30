import cv2
import numpy as np
import os
import json
from openai import OpenAI

class PalmAnalyzer:
    """
    掌纹特征分析与解读
    """
    
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if self.api_key:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.deepseek.com"
            )
        else:
            print("Warning: DEEPSEEK_API_KEY not set. Using rule-based fallback.")
            self.client = None

    def analyze(self, lines_result: dict, roi_shape: tuple) -> dict:
        """
        分析掌纹并返回解读报告。
        """
        features = self._extract_features(lines_result, roi_shape)
        
        # 如果有 API Key，调用 LLM
        if self.client:
            try:
                return self._analyze_with_llm(features)
            except Exception as e:
                print(f"LLM analysis failed: {e}. Falling back to rule-based.")
                
        # 降级策略
        return self._analyze_rule_based(features)

    def _extract_features(self, lines_result: dict, roi_shape: tuple) -> dict:
        height, width = roi_shape
        features = {}
        
        # 提取生命线
        life_res = {}
        # 兼容旧逻辑，如果有 'contour' 则用，否则用 'contours'
        # 但我们已经更新了 processor.py，现在只有 'contours'
        # 不过为了稳健性，我们需要从 contours 中聚合特征
        
        def aggregate_metrics(contours):
            if not contours: return None
            total_len = sum(cv2.arcLength(cnt, False) for cnt in contours)
            # 取最长的一段计算 curvature / slope 等特征
            longest_cnt = max(contours, key=lambda c: cv2.arcLength(c, False))
            return total_len, longest_cnt

        life_cnts = lines_result.get('life_line', {}).get('contours')
        res = aggregate_metrics(life_cnts)
        
        if res:
            length, longest_cnt = res
            norm_len = length / max(width, height)
            x, y, w, h = cv2.boundingRect(longest_cnt)
            curvature = w / width
            
            life_res['desc'] = f"长度指数 {norm_len:.2f}, 弧度指数 {curvature:.2f}"
            life_res['metrics'] = {'norm_len': float(norm_len), 'curvature': float(curvature)}
            life_res['detected'] = True
        else:
            life_res['desc'] = "未检测到"
            life_res['metrics'] = {}
            life_res['detected'] = False
        features['life_line'] = life_res
            
        # 提取智慧线
        head_res = {}
        head_cnts = lines_result.get('head_line', {}).get('contours')
        res = aggregate_metrics(head_cnts)
        
        if res:
            length, longest_cnt = res
            norm_len = length / width
            
            try:
                line_params = cv2.fitLine(longest_cnt, cv2.DIST_L2, 0, 0.01, 0.01)
                vx, vy, x, y = line_params.flatten()
                slope = vy / vx if vx != 0 else 100
            except Exception as e:
                print(f"Error fitting line for head_line: {e}")
                slope = 0
            
            head_res['desc'] = f"长度指数 {norm_len:.2f}, 斜率 {slope:.2f}"
            head_res['metrics'] = {'norm_len': float(norm_len), 'slope': float(slope)}
            head_res['detected'] = True
        else:
            head_res['desc'] = "未检测到"
            head_res['metrics'] = {}
            head_res['detected'] = False
        features['head_line'] = head_res
            
        # 提取感情线
        heart_res = {}
        heart_cnts = lines_result.get('heart_line', {}).get('contours')
        res = aggregate_metrics(heart_cnts)
        
        if res:
            length, longest_cnt = res
            norm_len = length / width
            # 复杂度：所有轮廓的复杂度之和
            complexity = sum(len(cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, False), False)) for cnt in heart_cnts)
            
            heart_res['desc'] = f"长度指数 {norm_len:.2f}, 复杂度 {complexity}"
            heart_res['metrics'] = {'norm_len': float(norm_len), 'complexity': int(complexity)}
            heart_res['detected'] = True
        else:
            heart_res['desc'] = "未检测到"
            heart_res['metrics'] = {}
            heart_res['detected'] = False
        features['heart_line'] = heart_res
            
        return features

    def _analyze_with_llm(self, features: dict) -> dict:
        prompt = f"""
        你是一位精通中国传统相术的手相大师。请根据以下掌纹特征数据，输出一段详细的性格与运势解读。
        
        [特征数据]
        生命线: {features['life_line']['desc']} (长度指数>0.6为长，弧度>0.3为饱满)
        智慧线: {features['head_line']['desc']} (长度指数>0.5为长，斜率绝对值>0.5为下垂)
        感情线: {features['heart_line']['desc']} (长度指数>0.6为长，复杂度>10为复杂)
        
        [输出要求]
        1. 语言风格：专业、神秘、富有哲理且积极向上。
        2. 必须严格返回合法的 JSON 格式，不要包含 Markdown 标记（如 ```json）。
        3. JSON 结构如下：
        {{
            "life_line": {{"feature": "简短特征描述(4-6字)", "reading": "详细解读..."}},
            "heart_line": {{"feature": "...", "reading": "..."}},
            "head_line": {{"feature": "...", "reading": "..."}}
        }}
        """
        
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that outputs JSON only."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=1.3
        )
        
        content = response.choices[0].message.content
        return json.loads(content)

    #默认规则(No LLM)
    def _analyze_rule_based(self, features: dict) -> dict:
        readings = {}
        
        # 分析生命线
        life = features.get('life_line', {})
        if life.get('detected'):
            m = life['metrics']
            feat_list = []
            read_list = []
            
            if m['norm_len'] > 0.6:
                feat_list.append("纹路深长")
                read_list.append("精力充沛，生命力顽强，抵抗力较好。")
            else:
                feat_list.append("纹路较短")
                read_list.append("平时要注意劳逸结合，避免过度透支体力。")
                
            if m['curvature'] > 0.3:
                feat_list.append("弧度饱满")
                read_list.append("性格开朗热情，社交能力强，生活充满活力。")
            else:
                feat_list.append("弧度平直")
                read_list.append("性格较为内敛冷静，做事谨慎，喜欢安稳的生活。")
            
            readings['life_line'] = {
                'feature': "，".join(feat_list),
                'reading': "".join(read_list)
            }
        else:
            readings['life_line'] = {'feature': '未检测到', 'reading': '暂无数据'}

        # 分析智慧线
        head = features.get('head_line', {})
        if head.get('detected'):
            m = head['metrics']
            feat_list = []
            read_list = []
            
            if m['norm_len'] > 0.5:
                feat_list.append("线条修长")
                read_list.append("思维清晰，逻辑感强，擅长深度思考。")
            else:
                feat_list.append("线条精炼")
                read_list.append("反应迅速，直觉敏锐，决策果断。")
                
            if abs(m['slope']) < 0.5:
                feat_list.append("走向平直")
                read_list.append("注重实际和理性，理财观念强，适合数理逻辑工作。")
            else:
                feat_list.append("末端下垂")
                read_list.append("想象力丰富，富有创意和艺术天分，感性思维活跃。")
                
            readings['head_line'] = {
                'feature': "，".join(feat_list),
                'reading': "".join(read_list)
            }
        else:
            readings['head_line'] = {'feature': '未检测到', 'reading': '暂无数据'}

        # 分析感情线
        heart = features.get('heart_line', {})
        if heart.get('detected'):
            m = heart['metrics']
            feat_list = []
            read_list = []
            
            if m['norm_len'] > 0.6:
                feat_list.append("延伸至指根")
                read_list.append("情感丰富细腻，重情重义，对感情专一且投入。")
            else:
                feat_list.append("中途停止")
                read_list.append("对待感情较为理智，不喜拖泥带水，自我保护意识强。")
                
            if m['complexity'] > 10:
                feat_list.append("纹路复杂")
                read_list.append(" 桃花运较旺，但情感经历可能较为波折。")
            else:
                feat_list.append("纹路清晰")
                read_list.append(" 感情生活简单纯粹，追求心灵的契合。")
                
            readings['heart_line'] = {
                'feature': "，".join(feat_list),
                'reading': "".join(read_list)
            }
        else:
            readings['heart_line'] = {'feature': '未检测到', 'reading': '暂无数据'}
            
        return readings

