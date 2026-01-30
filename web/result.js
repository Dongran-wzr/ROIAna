// 从 localStorage 获取数据
const API_BASE_URL = 'http://localhost:8000';
const resultData = JSON.parse(localStorage.getItem('palmResult'));
const timeElapsed = localStorage.getItem('palmTime');

const elements = {
    resultImage: document.getElementById('result-image'),
    overlayCanvas: document.getElementById('overlay-canvas'),
    canvasArea: document.getElementById('canvas-area'),
    
    timeTaken: document.getElementById('time-taken'),
    handType: document.getElementById('hand-type'),
    
    toggleLife: document.getElementById('toggle-life'),
    toggleHeart: document.getElementById('toggle-heart'),
    toggleHead: document.getElementById('toggle-head'),
    
    confLife: document.getElementById('conf-life'),
    confHeart: document.getElementById('conf-heart'),
    confHead: document.getElementById('conf-head'),
};

let state = {
    imageNaturalWidth: 0,
    imageNaturalHeight: 0,
    currentScale: 1
};

function init() {
    if (!resultData) {
        alert("无数据，请返回重新识别");
        location.href = 'index.html';
        return;
    }
    
    // 填充文本数据
    elements.timeTaken.textContent = `${timeElapsed}s`;
    
    const info = resultData.hand_info;
    elements.handType.textContent = `${info.label} Hand (${(info.score * 100).toFixed(1)}%)`;
    if (!info.is_open) {
        elements.handType.textContent += " [未展开]";
        elements.handType.style.color = "#f1c40f";
    }
    
    updateConfidence(elements.confLife, resultData.lines.life_line);
    updateConfidence(elements.confHeart, resultData.lines.heart_line);
    updateConfidence(elements.confHead, resultData.lines.head_line);
    
    // 加载图片
    let cleanUrl = resultData.clean_image_url;
    if (!cleanUrl.startsWith('/')) cleanUrl = '/' + cleanUrl;
    const imgUrl = API_BASE_URL + cleanUrl;
    
    elements.resultImage.src = imgUrl;
    
    elements.resultImage.onload = () => {
        state.imageNaturalWidth = elements.resultImage.naturalWidth;
        state.imageNaturalHeight = elements.resultImage.naturalHeight;
        
        // 初始布局计算
        resizeCanvas();
        drawLines();
    };
    
    // 绑定事件
    [elements.toggleLife, elements.toggleHeart, elements.toggleHead].forEach(el => {
        el.addEventListener('change', drawLines);
    });
    
    // 窗口大小改变时重绘
    window.addEventListener('resize', () => {
        resizeCanvas();
        drawLines();
    });
}

function updateConfidence(el, points) {
    if (points && points.length > 0) {
        el.textContent = "已检测";
        el.style.color = "#2ecc71";
    } else {
        el.textContent = "未检测";
        el.style.color = "#666";
    }
}

function resizeCanvas() {
    const img = elements.resultImage;
    const canvas = elements.overlayCanvas;
    
    // 获取图片在容器中的实际显示尺寸
    const ratio = state.imageNaturalWidth / state.imageNaturalHeight;
    const containerW = elements.canvasArea.clientWidth;
    const containerH = elements.canvasArea.clientHeight;
    
    let renderW, renderH;
    
    if (containerW / containerH > ratio) {
        // 容器更宽，高度受限
        renderH = containerH;
        renderW = renderH * ratio;
    } else {
        // 容器更高，宽度受限
        renderW = containerW;
        renderH = renderW / ratio;
    }
    
    
    canvas.width = renderW;
    canvas.height = renderH;
    canvas.style.width = `${renderW}px`;
    canvas.style.height = `${renderH}px`;
}

function drawLines() {
    const canvas = elements.overlayCanvas;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const scaleX = canvas.width / state.imageNaturalWidth;
    const scaleY = canvas.height / state.imageNaturalHeight;
    
    const lines = resultData.lines;
    
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

init();
