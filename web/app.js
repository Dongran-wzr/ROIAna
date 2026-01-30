// 全局配置
const API_BASE_URL = 'http://localhost:8000'; 

// DOM 元素引用
const elements = {
    // 标签页切换
    tabBtns: document.querySelectorAll('.tab-btn'),
    panels: document.querySelectorAll('.panel'),
    
    // 上传
    dropZone: document.getElementById('drop-zone'),
    fileInput: document.getElementById('file-input'),
    
    // 摄像头
    video: document.getElementById('video'),
    captureCanvas: document.getElementById('capture-canvas'),
    startCameraBtn: document.getElementById('start-camera-btn'),
    captureBtn: document.getElementById('capture-btn'),
    
    // 主操作
    analyzeBtn: document.getElementById('analyze-btn'),
    
    // 结果展示
    resultImage: document.getElementById('result-image'),
    overlayCanvas: document.getElementById('overlay-canvas'),
    loading: document.getElementById('loading'),
    emptyState: document.getElementById('empty-state'),
    
    // 控制与数据
    controlsBar: document.getElementById('controls-bar'),
    statsPanel: document.getElementById('stats-panel'),
    errorBox: document.getElementById('error-box'),
    
    // 详细数据
    timeTaken: document.getElementById('time-taken'),
    handType: document.getElementById('hand-type'),
    errorReason: document.getElementById('error-reason'),
    errorSuggestion: document.getElementById('error-suggestion'),
    
    // 开关
    toggleLife: document.getElementById('toggle-life'),
    toggleHeart: document.getElementById('toggle-heart'),
    toggleHead: document.getElementById('toggle-head'),
    
    // 进度条
    confLife: document.getElementById('conf-life'),
    confHeart: document.getElementById('conf-heart'),
    confHead: document.getElementById('conf-head'),
    confLifeText: document.getElementById('conf-life-text'),
    confHeartText: document.getElementById('conf-heart-text'),
    confHeadText: document.getElementById('conf-head-text'),
};

// 状态管理
let state = {
    currentFile: null,
    stream: null,
    detectionResult: null,
    cleanImageUrl: null,
    imageNaturalWidth: 0,
    imageNaturalHeight: 0
};

// 初始化
function init() {
    setupTabs();
    setupUpload();
    setupCamera();
    setupAnalysis();
    setupToggles();
    
    // 窗口大小改变时
    window.addEventListener('resize', resizeCanvas);
}

// 标签页切换
function setupTabs() {
    elements.tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            // 切换按钮状态
            elements.tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // 切换面板
            const targetId = btn.dataset.target;
            elements.panels.forEach(p => p.classList.remove('active'));
            document.getElementById(targetId).classList.add('active');
            
            // 如果切走摄像头面板，关闭摄像头
            if (targetId !== 'camera-panel' && state.stream) {
                stopCamera();
            }
        });
    });
}

// 文件上传
function setupUpload() {
    const { dropZone, fileInput } = elements;
    
    dropZone.addEventListener('click', () => fileInput.click());
    
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });
    
    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });
    
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });
    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    });
}

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('请上传图片文件');
        return;
    }
    
    state.currentFile = file;
    elements.analyzeBtn.disabled = false;
    
    // 预览图片
    const reader = new FileReader();
    reader.onload = (e) => {
        displayPreview(e.target.result);
    };
    reader.readAsDataURL(file);
}

// 摄像头
function setupCamera() {
    elements.startCameraBtn.addEventListener('click', startCamera);
    
    elements.captureBtn.addEventListener('click', () => {
        if (!state.stream) return;
        
        const video = elements.video;
        const canvas = elements.captureCanvas;
        
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0);
        
        canvas.toBlob((blob) => {
            const file = new File([blob], "capture.jpg", { type: "image/jpeg" });
            handleFile(file);
            stopCamera();
        }, 'image/jpeg', 0.95);
    });
}

async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { facingMode: 'environment' } 
        });
        state.stream = stream;
        elements.video.srcObject = stream;
        elements.captureBtn.disabled = false;
        elements.startCameraBtn.textContent = "关闭摄像头";
        elements.startCameraBtn.onclick = stopCamera;
    } catch (err) {
        console.error("无法访问摄像头:", err);
        alert("无法访问摄像头，请检查权限或设备连接");
    }
}

function stopCamera() {
    if (state.stream) {
        state.stream.getTracks().forEach(track => track.stop());
        state.stream = null;
        elements.video.srcObject = null;
        elements.captureBtn.disabled = true;
        elements.startCameraBtn.textContent = "开启摄像头";
        elements.startCameraBtn.onclick = startCamera;
    }
}

// API 调用
function setupAnalysis() {
    elements.analyzeBtn.addEventListener('click', async () => {
        if (!state.currentFile) return;
        
        showLoading(true);
        resetResultView();
        
        const formData = new FormData();
        formData.append('file', state.currentFile);
        
        const startTime = performance.now();
        
        try {
            const response = await fetch(`${API_BASE_URL}/detect`, {
                method: 'POST',
                body: formData
            });
            
            const endTime = performance.now();
            const timeElapsed = ((endTime - startTime) / 1000).toFixed(2);
            
            const result = await response.json();
            
            if (!response.ok) {
                // 处理 API 返回的错误
                handleError(result.detail);
            } else {
                handleSuccess(result, timeElapsed);
            }
            
        } catch (err) {
            console.error(err);
            handleError({ 
                error_code: 500, 
                suggestion: "网络请求失败，请检查后端服务是否启动。" 
            });
        } finally {
            showLoading(false);
        }
    });
}

// 结果处理
function handleSuccess(result, timeElapsed) {
    console.log("Detection success:", result);
    state.detectionResult = result;
    
    let cleanUrl = result.clean_image_url;
    if (!cleanUrl.startsWith('/')) cleanUrl = '/' + cleanUrl;
    
    const imgUrl = API_BASE_URL + cleanUrl;
    console.log("Loading image from:", imgUrl);
    
    state.cleanImageUrl = imgUrl;
    
    elements.resultImage.src = imgUrl;
    
    elements.resultImage.removeAttribute('hidden');
    elements.resultImage.style.display = 'block';
    elements.emptyState.setAttribute('hidden', '');
    elements.emptyState.style.display = 'none';
    
    elements.resultImage.onload = () => {
        console.log("Image loaded successfully");
        state.imageNaturalWidth = elements.resultImage.naturalWidth;
        state.imageNaturalHeight = elements.resultImage.naturalHeight;
        
        // 显示相关面板
        elements.overlayCanvas.removeAttribute('hidden');
        elements.overlayCanvas.style.display = 'block';
        
        resizeCanvas();
        drawLines(); 
    };
    
    elements.resultImage.onerror = (e) => {
        console.error("Image load failed:", e);
        alert(`图片加载失败: ${imgUrl}`);
    };
    
    // 更新数据面板
    elements.controlsBar.removeAttribute('hidden');
    elements.controlsBar.style.display = 'block';
    
    elements.statsPanel.removeAttribute('hidden');
    elements.statsPanel.style.display = 'block';
    
    elements.errorBox.setAttribute('hidden', '');
    elements.errorBox.style.display = 'none';
    
    elements.timeTaken.textContent = `${timeElapsed}s`;
    elements.handType.textContent = `${result.hand_info.label} Hand (置信度: ${(result.hand_info.score * 100).toFixed(1)}%)`;
    if (!result.hand_info.is_open) {
        elements.handType.textContent += " [未完全展开]";
        elements.handType.style.color = "orange";
    } else {
        elements.handType.style.color = "inherit";
    }
    
    updateConfidence('life_line', result.lines.life_line);
    updateConfidence('heart_line', result.lines.heart_line);
    updateConfidence('head_line', result.lines.head_line);
    
    // 存储数据并跳转
    localStorage.setItem('palmResult', JSON.stringify(result));
    localStorage.setItem('palmTime', timeElapsed);
    
    // 延迟一点跳转以展示加载完成的状态，或者直接跳转
    setTimeout(() => {
        window.location.href = 'result.html';
    }, 500);
}

function updateConfidence(lineKey, points) {
    const hasLine = points && points.length > 0;
    const percent = hasLine ? '100%' : '0%';
    const text = hasLine ? 'Detected' : 'Not Found';
    
    // 简单的映射
    const map = {
        'life_line': { bar: elements.confLife, text: elements.confLifeText },
        'heart_line': { bar: elements.confHeart, text: elements.confHeartText },
        'head_line': { bar: elements.confHead, text: elements.confHeadText }
    };
    
    const el = map[lineKey];
    el.bar.style.width = percent;
    el.text.textContent = text;
}

function handleError(error) {
    elements.resultImage.hidden = true;
    elements.overlayCanvas.hidden = true;
    elements.controlsBar.hidden = true;
    elements.statsPanel.hidden = false;
    elements.errorBox.hidden = false;
    
    // 显示上传的原图预览，但标记为失败
    // (保持 displayPreview 的内容)
    
    const code = error.error_code || 'Unknown';
    const reasonMap = {
        1000: "无效图片",
        1001: "未检测到手掌",
        1002: "手掌过小",
        1003: "手掌未展开"
    };
    
    elements.errorReason.textContent = reasonMap[code] || `Error ${code}`;
    elements.errorSuggestion.textContent = error.suggestion || "请尝试重新拍摄，保持光线充足，手掌完整。";
}

// 绘图or not
function setupToggles() {
    [elements.toggleLife, elements.toggleHeart, elements.toggleHead].forEach(el => {
        el.addEventListener('change', drawLines);
    });
}

function resizeCanvas() {
    if (!state.cleanImageUrl) return;
    
    // 匹配图片的显示尺寸
    const img = elements.resultImage;
    const canvas = elements.overlayCanvas;
    
    canvas.width = img.clientWidth;
    canvas.height = img.clientHeight;
    
    drawLines();
}

function drawLines() {
    if (!state.detectionResult) return;
    
    const canvas = elements.overlayCanvas;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // 计算缩放比例 (显示尺寸 / 原始尺寸)
    const scaleX = canvas.width / state.imageNaturalWidth;
    const scaleY = canvas.height / state.imageNaturalHeight;
    
    const lines = state.detectionResult.lines;
    
    // 配置颜色
    const styles = {
        'life_line': { color: '#e74c3c', show: elements.toggleLife.checked },
        'heart_line': { color: '#3498db', show: elements.toggleHeart.checked },
        'head_line': { color: '#2ecc71', show: elements.toggleHead.checked }
    };
    
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    
    for (const [name, points] of Object.entries(lines)) {
        if (!points || points.length === 0) continue;
        if (!styles[name].show) continue;
        
        ctx.strokeStyle = styles[name].color;
        ctx.beginPath();
        
        points.forEach((p, i) => {
            const x = p[0] * scaleX;
            const y = p[1] * scaleY;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        
        ctx.stroke();
    }
}

function showLoading(show) {
    if (show) {
        elements.loading.removeAttribute('hidden');
        elements.loading.style.display = 'flex'; // 强制显示
    } else {
        elements.loading.setAttribute('hidden', '');
        elements.loading.style.display = 'none'; // 强制隐藏
    }
    elements.analyzeBtn.disabled = show;
}

function displayPreview(src) {
    elements.resultImage.src = src;
    elements.resultImage.hidden = false;
    elements.emptyState.hidden = true;
    elements.overlayCanvas.hidden = true; 
    elements.controlsBar.hidden = true;
    elements.statsPanel.hidden = true;
}

function resetResultView() {
    // 清空结果状态，准备新的分析
    state.detectionResult = null;
    const ctx = elements.overlayCanvas.getContext('2d');
    ctx.clearRect(0, 0, elements.overlayCanvas.width, elements.overlayCanvas.height);
}

// 启动
document.addEventListener('DOMContentLoaded', init);
