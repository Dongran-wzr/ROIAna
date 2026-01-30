# 集成LLM掌纹分析系统（赛博手相大师）

这是一个基于CV的掌纹分析系统。它能够自动检测手掌 ROI，提取生命线、智慧线、感情线，并接入LLM，提供详细的性格与运势解读。

## 技术栈

*   **后端**: Python 3.12, FastAPI, Uvicorn
*   **计算机视觉**: OpenCV, MediaPipe, NumPy
*   **LLM**: OpenAI SDK (DeepSeek V3)
*   **前端**: HTML5, CSS3, JavaScript

## 环境准备与安装

### 1. 环境要求
*   Windows / macOS / Linux
*   Python 3.8 或更高版本

### 2. 下载项目
在本项目主页点击 `Code` 按钮，选择 `Download ZIP` 下载项目代码到本地目录。

### 3. 创建虚拟环境
为了避免依赖冲突，建议创建一个 Python 虚拟环境。

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. 安装依赖

```bash
pip install "list_package"
#list_package包括以下全部
package:
    uvicorn
    python-multipart
    opencv-python
    mediapipe
    numpy
    openai
    pydantic
```

> 如果下载速度慢，可以使用国内镜像源：
> `pip install package -i https://pypi.tuna.tsinghua.edu.cn/simple`

## 运行步骤

### 1. 配置 API Key
如果你希望使用 DeepSeek 的 AI 解读功能，需要设置环境变量。如果未设置，系统将采用降级策略，使用内置的规则库进行基础解读。

**CMD:**
```cmd
set DEEPSEEK_API_KEY=sk-api-key
```

**Linux / macOS:**
```bash
export DEEPSEEK_API_KEY="sk-api-key"
```

### 2. 启动后端服务
在项目根目录下运行以下命令启动 FastAPI 服务器：

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

启动成功后，控制台会显示类似如下信息：
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 3. 访问前端页面

打开一个新的终端窗口，进入 `web` 目录：
```bash
cd web
python -m http.server 8080
```
然后在浏览器访问: [http://localhost:8080](http://localhost:8080)

## 测试步骤
`main.py`是测试入口。在终端窗口运行，进行代码测试：

```bash
python main.py img/img01.jpg --output processed.jpg --save-data
```
运行后会在主目录下保存 JSON 数据和标注图像。


## 项目结构

```
ROIAna/
├── src/                # 核心算法模块
│   ├── detector.py     # 手掌检测与 ROI 裁切
│   ├── processor.py    # 掌纹线条提取算法
│   ├── analyzer.py     # 特征计算与 AI 解读接口
│   └── utils.py        # 图像处理工具函数
├── web/                # 前端资源
│   ├── index.html      # 主页
│   ├── result.html     # 结果展示与交互页
│   ├── app.js          # 主页逻辑
│   ├── result.js       # 结果页逻辑
│   └── style.css       # 样式
├── api.py              # 后端入口
├── main.py             # 测试用主程序！！！（代码功能测试专用：可以直接运行，无需启动后端）
├── README.md           # 说明文档
└── temp_uploads/       # 上传or保存的临时文件
```

## 常见问题（本人遇到过！）

**Q: 启动时提示端口被占用 `[Errno 10048]`?**
A: 说明端口 8000 正在被使用。请尝试更改端口，例如 `uvicorn api:app --port 8001`，或者在任务管理器中关闭占用端口的 python 进程。

**Q: 点击“DeepSeek 大师解读”没有反应或报错?**
A: 请检查后台控制台日志。通常是因为 API Key 未正确设置或余额不足（球球别人了，因为我就是😍）。如果没有 Key，系统会非常智能的自动降级到规则解读模式。

**Q: 你是否发现某些图像划分结果与预期不符?**
A: 没错！其实因为我用的是 MediaPipe 的手掌检测模型hhh。

