import cv2
import numpy as np
import json
import argparse
import sys
import os

class ManualCorrector:
    def __init__(self, image_path, data_path):
        self.image_path = image_path
        self.data_path = data_path
        
        self.image = cv2.imread(image_path)
        if self.image is None:
            print(f"错误: 无法加载图片 {image_path}")
            sys.exit(1)
        self.display_image = self.image.copy()
            
        try:
            with open(data_path, 'r') as f:
                self.data = json.load(f)
        except Exception as e:
            print(f"错误: 无法加载数据 {data_path}: {e}")
            sys.exit(1)
            
        # 转换点格式为列表
        self.lines = {}
        for name, points in self.data['lines'].items():
            if points:
                self.lines[name] = [p[0] for p in points]
            else:
                self.lines[name] = []
                
        self.selected_point = None 
        self.hover_point = None
        self.drag_active = False
        
        self.colors = {
            'life_line': (0, 0, 255),   
            'heart_line': (255, 0, 0),  
            'head_line': (0, 255, 0)  
        }
        
        self.point_radius = 6
        self.window_name = "Palm Line Corrector (Press 's' to save, 'q' to quit)"
        
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

    def draw(self):
        img = self.image.copy()
        
        
        for name, points in self.lines.items():
            if not points: continue
            
            pts = np.array(points, np.int32)
            color = self.colors.get(name, (255, 255, 255))
            
            cv2.polylines(img, [pts], False, color, 2)
            
            # 绘制关键点
            for i, p in enumerate(points):
                # 高亮悬停或选中的点
                if self.hover_point == (name, i) or self.selected_point == (name, i):
                    cv2.circle(img, (p[0], p[1]), self.point_radius + 2, (0, 255, 255), -1)
                else:
                    cv2.circle(img, (p[0], p[1]), self.point_radius, color, -1)
                    
        self.display_image = img
        cv2.imshow(self.window_name, self.display_image)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.hover_point:
                self.selected_point = self.hover_point
                self.drag_active = True
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.drag_active = False
            self.selected_point = None
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drag_active and self.selected_point:
                name, idx = self.selected_point
                self.lines[name][idx] = [x, y]
            else:
                # 寻找最近的点
                min_dist = float('inf')
                closest = None
                
                for name, points in self.lines.items():
                    for i, p in enumerate(points):
                        dist = np.sqrt((x - p[0])**2 + (y - p[1])**2)
                        if dist < self.point_radius * 2: 
                            if dist < min_dist:
                                min_dist = dist
                                closest = (name, i)
                
                self.hover_point = closest
                
        self.draw()

    def run(self):
        self.draw()
        print("操作指南:")
        print("  - 鼠标左键拖动关键点进行微调")
        print("  - 按 's' 保存修改")
        print("  - 按 'q' 退出")
        
        while True:
            key = cv2.waitKey(20) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save()
                
        cv2.destroyAllWindows()

    def save(self):
        # 转换回原始格式保存
        export_lines = {}
        for name, points in self.lines.items():
            formatted = [[[p[0], p[1]]] for p in points]
            export_lines[name] = formatted
            
        self.data['lines'] = export_lines
        
        try:
            with open(self.data_path, 'w') as f:
                json.dump(self.data, f)
            print(f"修改已保存至: {self.data_path}")
            
            viz_path = os.path.splitext(self.data_path)[0] + "_corrected.jpg"
            cv2.imwrite(viz_path, self.display_image)
            print(f"可视化图已保存至: {viz_path}")
            
        except Exception as e:
            print(f"保存失败: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="掌纹人工矫正工具")
    parser.add_argument("json_path", help="检测生成的 JSON 数据文件路径")
    parser.add_argument("--image", help="图片路径 (如果不指定，将尝试从 JSON 中读取)", default=None)
    args = parser.parse_args()
    
    json_path = args.json_path
    
    if args.image:
        image_path = args.image
    else:
        # 尝试从 JSON 读取
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                image_path = data.get('image_path')
        except:
            image_path = None
            
    if not image_path or not os.path.exists(image_path):
        # 尝试默认命名规则
        # 假设 json 是 output.json, 图片是 output.jpg
        base = os.path.splitext(json_path)[0]
        possible_exts = ['.jpg', '.png', '.jpeg']
        found = False
        for ext in possible_exts:
            if os.path.exists(base + ext):
                image_path = base + ext
                found = True
                break
        
        if not found:
            print("错误: 无法找到对应的图片文件，请使用 --image 指定。")
            sys.exit(1)
            
    print(f"正在加载: {image_path}")
    corrector = ManualCorrector(image_path, json_path)
    corrector.run()
