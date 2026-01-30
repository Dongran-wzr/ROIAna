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
    currentScale: 1,
    isEditing: false,
    hoverPoint: null, 
    selectedPoint: null
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
    
    updateConfidence(elements.confLife, resultData.confidences.life_line, resultData.lines.life_line);
    updateConfidence(elements.confHeart, resultData.confidences.heart_line, resultData.lines.heart_line);
    updateConfidence(elements.confHead, resultData.confidences.head_line, resultData.lines.head_line);
    
    // 绑定 AI 解读按钮
    const aiBtn = document.getElementById('ai-analyze-btn');
    if (aiBtn) {
        aiBtn.addEventListener('click', async () => {
            aiBtn.disabled = true;
            aiBtn.textContent = "🔮 大师正在冥想中...";
            
            try {
                const response = await fetch(`${API_BASE_URL}/analyze_hand`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ data_id: resultData.data_id })
                });
                
                const readingResult = await response.json();
                
                if (!response.ok) throw new Error(readingResult.detail || 'Analysis failed');
                
                // 渲染结果
                renderReading('life', readingResult.life_line);
                renderReading('heart', readingResult.heart_line);
                renderReading('head', readingResult.head_line);
                
                aiBtn.textContent = "✅ 解读完成";
            } catch (err) {
                console.error(err);
                alert("大师解读失败: " + err.message);
                aiBtn.disabled = false;
                aiBtn.textContent = "🔮 DeepSeek 大师解读";
            }
        });
    }
    
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
    
    // 绑定编辑按钮
    const editBtn = document.getElementById('edit-btn');
    const saveEditBtn = document.getElementById('save-edit-btn');
    
    editBtn.addEventListener('click', () => {
        state.isEditing = !state.isEditing;
        if (state.isEditing) {
            editBtn.textContent = "❌ 退出编辑";
            editBtn.style.background = "#c0392b";
            saveEditBtn.style.display = "block";
            elements.overlayCanvas.style.pointerEvents = "auto";
            elements.overlayCanvas.style.cursor = "crosshair";
        } else {
            editBtn.textContent = "✏️ 人工矫正";
            editBtn.style.background = "#e67e22";
            saveEditBtn.style.display = "none";
            elements.overlayCanvas.style.pointerEvents = "none";
            state.selectedPoint = null;
        }
        drawLines();
    });
    
    saveEditBtn.addEventListener('click', async () => {
        saveEditBtn.disabled = true;
        saveEditBtn.textContent = "保存中...";
        
        try {
            const payload = {
                data_id: resultData.data_id,
                lines: resultData.lines
            };
            
            const response = await fetch(`${API_BASE_URL}/correct`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            
            if (!response.ok) {
                let errorMsg = "保存失败";
                try {
                    const errData = await response.json();
                    if (errData.detail) {
                        errorMsg = typeof errData.detail === 'object' ? JSON.stringify(errData.detail) : errData.detail;
                    }
                } catch (e) {
                    // ignore json parse error
                }
                throw new Error(errorMsg);
            }
            
            await response.json();
            alert("✅ 矫正保存成功！");
            editBtn.click();
            
        } catch (e) {
            alert("❌ " + e.message);
        } finally {
            saveEditBtn.disabled = false;
            saveEditBtn.textContent = "💾 保存修改";
        }
    });
    
    // Canvas 交互事件
    const canvas = elements.overlayCanvas;
    
    canvas.addEventListener('mousemove', (e) => {
        if (!state.isEditing) return;
        
        const rect = canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left) * (canvas.width / rect.width);
        const y = (e.clientY - rect.top) * (canvas.height / rect.height);
        
        const scaleX = state.imageNaturalWidth / canvas.width;
        const scaleY = state.imageNaturalHeight / canvas.height;
        
        const originalX = x * scaleX;
        const originalY = y * scaleY;
        
        let minDist = 20; 
        let found = null;
        
        for (const [name, segments] of Object.entries(resultData.lines)) {
            const toggle = document.getElementById(`toggle-${name.split('_')[0]}`);
            if (toggle && !toggle.checked) continue;
            
            segments.forEach((seg, sIdx) => {
                seg.forEach((p, pIdx) => {
                    const dist = Math.hypot(p[0] - originalX, p[1] - originalY);
                    if (dist < minDist) {
                        minDist = dist;
                        found = { lineName: name, segIndex: sIdx, pointIndex: pIdx };
                    }
                });
            });
        }
        
        state.hoverPoint = found;
        
        if (state.selectedPoint) {
            const { lineName, segIndex, pointIndex } = state.selectedPoint;
            // 确保坐标为整数，避免后端 Pydantic 校验错误
            resultData.lines[lineName][segIndex][pointIndex] = [Math.round(originalX), Math.round(originalY)];
        }
        
        canvas.style.cursor = found ? "pointer" : "crosshair";
        drawLines();
    });
    
    canvas.addEventListener('mousedown', (e) => {
        if (state.isEditing && state.hoverPoint) {
            state.selectedPoint = state.hoverPoint;
        }
    });
    
    canvas.addEventListener('mouseup', () => {
        state.selectedPoint = null;
    });
    
    // 绑定 AI 解读按钮窗口大小改变时重绘
    window.addEventListener('resize', () => {
        resizeCanvas();
        drawLines();
    });
}

function updateConfidence(el, score, points) {
    if (points && points.length > 0) {
        el.textContent = `已检测 (${(score * 100).toFixed(0)}%)`;
        el.style.color = "#2ecc71";
    } else {
        el.textContent = "未检测";
        el.style.color = "#666";
    }
}

function renderReading(type, data) {
    const box = document.getElementById(`reading-${type}`);
    if (!data || data.feature === '未检测到') {
        box.style.display = 'none';
        return;
    }
    
    box.style.display = 'block';
    box.querySelector('.feat').textContent = data.feature;
    box.querySelector('.read').textContent = data.reading;
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
    
    for (const [name, segments] of Object.entries(lines)) {
        if (!segments || segments.length === 0) continue;
        if (!styles[name].show) continue;
        
        const color = styles[name].color;
        
        // 绘制线段
        ctx.lineWidth = 3;
        ctx.strokeStyle = color;
        
        segments.forEach(segment => {
            ctx.beginPath();
            segment.forEach((p, i) => {
                const x = p[0] * scaleX;
                const y = p[1] * scaleY;
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            });
            ctx.stroke();
            
            // 编辑模式下绘制关键点
            if (state.isEditing) {
                segment.forEach((p, i) => {
                    const x = p[0] * scaleX;
                    const y = p[1] * scaleY;
                    
                    ctx.beginPath();
                    ctx.fillStyle = color;
                    ctx.arc(x, y, 3, 0, Math.PI * 2);
                    ctx.fill();
                });
            }
        });
    }
    
    // 绘制 Hover 高亮
    if (state.isEditing && state.hoverPoint) {
        const { lineName, segIndex, pointIndex } = state.hoverPoint;
        const p = lines[lineName][segIndex][pointIndex];
        const x = p[0] * scaleX;
        const y = p[1] * scaleY;
        
        ctx.beginPath();
        ctx.fillStyle = "#fff";
        ctx.strokeStyle = styles[lineName].color;
        ctx.lineWidth = 2;
        ctx.arc(x, y, 6, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
    }
}

init();
