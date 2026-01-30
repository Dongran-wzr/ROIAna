from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import shutil
import os
import cv2
import numpy as np
import base64
import uuid
import json
import uvicorn

from src.utils import load_image_with_exif, resize_to_1080p
from src.detector import HandDetector
from src.processor import PalmLineExtractor
from src.analyzer import PalmAnalyzer

# 创建 FastAPI 实例
app = FastAPI(
    title="Recognition API",
    description="掌纹识别与分析 API",
    version="1.0.0"
)

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 临时文件存储目录
TEMP_DIR = "temp_uploads"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

# 启动时初始化
detector = HandDetector()
extractor = PalmLineExtractor()
analyzer = PalmAnalyzer()

class DetectionResult(BaseModel):
    lines: Dict[str, List[List[List[int]]]] # 多段线: 线段列表 -> 点列表 -> [x,y]
    hand_info: Dict
    # reading: Dict[str, Dict[str, str]] # 移除自动解读
    image_url: str # 处理后图片的访问链接
    clean_image_url: str # 原始图片(用于底图)
    data_id: str   # 用于后续矫正的唯一 ID

class AnalyzeRequest(BaseModel):
    data_id: str

class CorrectionRequest(BaseModel):
    data_id: str
    lines: Dict[str, List[List[List[int]]]] # 支持多段线

@app.get("/")
async def root():
    return {"message": "Recognition API is running. Visit /docs for documentation."}

@app.post("/detect", response_model=DetectionResult)
async def detect_palm(file: UploadFile = File(...)):
    """
    上传图片进行掌纹检测。
    返回检测到的主线坐标和处理后的图片链接。
    """
    file_id = str(uuid.uuid4())
    filename = f"{file_id}_{file.filename}"
    file_path = os.path.join(TEMP_DIR, filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件保存失败: {str(e)}")

    original_img = load_image_with_exif(file_path)
    if original_img is None:
        raise HTTPException(status_code=400, detail="无效的图片文件")
        
    processed_img = resize_to_1080p(original_img)
    
    detect_result = detector.detect_and_crop(processed_img)
    
    if 'error_code' in detect_result:
        raise HTTPException(status_code=400, detail=detect_result)
        
    roi = detect_result['roi']
    landmarks_roi = detect_result['landmarks_roi']
    bbox = detect_result['bbox']
    hand_info = detect_result['hand_info']
    x1, y1, _, _ = bbox
    
    # 掌纹提取
    lines_result = extractor.extract_lines(roi, landmarks_roi)
    
    # 保存中间数据供后续解读
    features = analyzer._extract_features(lines_result, roi.shape[:2])
    feature_file = os.path.join(TEMP_DIR, f"{file_id}_features.json")
    with open(feature_file, 'w') as f:
        json.dump(features, f)
    
    # 数据整理
    overlay = processed_img.copy()
    h, w = processed_img.shape[:2]
    line_thickness = max(2, int(w / 300))
    
    export_lines = {}
    
    for line_name, data in lines_result.items():
        # data 现在包含 'contours' 列表
        contours = data.get('contours', [])
        confidence = data['confidence']
        color = data['color']
        
        export_lines[line_name] = []
        
        if contours and confidence > 0.1:
            for contour in contours:
                contour_original = contour + [x1, y1]
                
                # 简化数据
                epsilon = 0.001 * cv2.arcLength(contour_original, False)
                approx = cv2.approxPolyDP(contour_original, epsilon, False)
                
                # approx shape is (N, 1, 2) -> [[x,y], [x,y]...]
                points_list = approx.reshape(-1, 2).tolist()
                export_lines[line_name].append(points_list)
                
                # 绘制到 overlay 用于生成静态预览图 (依然绘制，作为备份)
                cv2.drawContours(overlay, [contour_original], -1, color, line_thickness)

    alpha = 0.7
    cv2.addWeighted(overlay, alpha, processed_img, 1 - alpha, 0, processed_img)
    cv2.rectangle(processed_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 255), max(1, line_thickness // 2))
    
    result_filename = f"result_{filename}"
    result_path = os.path.join(TEMP_DIR, result_filename)
    cv2.imwrite(result_path, processed_img)
    
    clean_filename = f"clean_{filename}"
    clean_path = os.path.join(TEMP_DIR, clean_filename)
    clean_img = resize_to_1080p(original_img)
    cv2.imwrite(clean_path, clean_img)

    return {
        "lines": export_lines,
        "hand_info": hand_info,
        "image_url": f"/images/{result_filename}",
        "clean_image_url": f"/images/{clean_filename}", # 额外返回干净图
        "data_id": file_id
    }

@app.post("/analyze_hand")
async def analyze_hand(request: AnalyzeRequest):
    """
    调用 DeepSeek 进行手相解读。
    """
    file_id = request.data_id
    feature_file = os.path.join(TEMP_DIR, f"{file_id}_features.json")
    
    if not os.path.exists(feature_file):
        raise HTTPException(status_code=404, detail="Analysis data not found")
        
    with open(feature_file, 'r') as f:
        features = json.load(f)
        
    # 调用 LLM
    try:
        if analyzer.client:
            result = analyzer._analyze_with_llm(features)
        else:
            result = analyzer._analyze_rule_based(features)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/correct")
async def correct_palm(request: CorrectionRequest):
    """
    接收人工矫正后的线条数据，重新绘制结果图，并更新特征数据供分析使用。
    """
   
    clean_files = [f for f in os.listdir(TEMP_DIR) if f.startswith(f"clean_{request.data_id}")]
    if not clean_files:
         raise HTTPException(status_code=404, detail="找不到原始检测记录")
         
    clean_filename = clean_files[0]
    clean_path = os.path.join(TEMP_DIR, clean_filename)
    
    img = cv2.imread(clean_path)
    if img is None:
        raise HTTPException(status_code=500, detail="无法加载原始图片")
        
    overlay = img.copy()
    h, w = img.shape[:2]
    line_thickness = max(2, int(w / 300))
    
    colors = {
        'life_line': (0, 0, 255),   
        'heart_line': (255, 0, 0),  
        'head_line': (0, 255, 0)   
    }
    
    # 1. 保存矫正后的线条数据
    corrected_data_path = os.path.join(TEMP_DIR, f"{request.data_id}_corrected.json")
    with open(corrected_data_path, 'w') as f:
        json.dump(request.lines, f)
        
    # 2. 重构数据以进行特征重新提取
    reconstructed_result = {}
    
    for line_name, segments in request.lines.items():
        if not segments: 
            reconstructed_result[line_name] = {'contours': []}
            continue
            
        # 转换回 numpy array 列表，格式 (N, 1, 2)
        contours = []
        
        color = colors.get(line_name, (255, 255, 255))
        
        for segment in segments:
            if not segment: continue
            pts = np.array(segment, np.int32)
            pts = pts.reshape((-1, 1, 2))
            contours.append(pts)
            
            # 绘制
            cv2.polylines(overlay, [pts], False, color, line_thickness)
            
        reconstructed_result[line_name] = {'contours': contours}

    # 3. 重新计算特征并保存
    try:
        features = analyzer._extract_features(reconstructed_result, (h, w))
        feature_file = os.path.join(TEMP_DIR, f"{request.data_id}_features.json")
        with open(feature_file, 'w') as f:
            json.dump(features, f)
    except Exception as e:
        print(f"Feature re-extraction failed: {e}")
        # 不阻断流程，仅打印错误
        
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    # 保存矫正后的结果图片
    corrected_filename = f"corrected_{request.data_id}.jpg"
    corrected_path = os.path.join(TEMP_DIR, corrected_filename)
    cv2.imwrite(corrected_path, img)
    
    return {
        "message": "Correction saved and analysis updated",
        "image_url": f"/images/{corrected_filename}"
    }

from fastapi.staticfiles import StaticFiles
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/images", StaticFiles(directory=TEMP_DIR), name="images")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
