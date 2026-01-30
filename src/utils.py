import cv2
import numpy as np
from PIL import Image, ImageOps
import os

def load_image_with_exif(path: str) -> np.ndarray:
    """
    读取图片并处理 EXIF 旋转信息
    """
    if not os.path.exists(path):
        print(f"错误: 文件不存在 - {path}")
        return None

    try:
        img_pil = Image.open(path)
        
        img_pil = ImageOps.exif_transpose(img_pil)
        
        img_np = np.array(img_pil)
        
        if img_np.ndim == 3:
            if img_np.shape[2] == 3:
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            elif img_np.shape[2] == 4:
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
            else:
                img_bgr = img_np 
        elif img_np.ndim == 2:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = img_np
            
        return img_bgr
    except Exception as e:
        print(f"读取图片失败: {e}")
        return None

def resize_to_1080p(image: np.ndarray) -> np.ndarray:
    """
    将图片高度缩放到 1080p
    """
    if image is None:
        return None
        
    height, width = image.shape[:2]
    target_height = 1080
    
    scale = target_height / float(height)
    new_width = int(width * scale)
    
   
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    
    resized_img = cv2.resize(image, (new_width, target_height), interpolation=interpolation)
    
    return resized_img
