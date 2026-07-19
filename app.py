"""
鹏翔驾校 AI 助手 - Flask 主应用
================================
提供 Web 聊天界面和 API 接口。
支持照片展示、会话管理、图片一键渲染。
"""

import os
import sys
import json
import uuid
import threading
from flask import Flask, request, jsonify, render_template_string

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from config import (
    DOUBAO_API_KEY, DOUBAO_MODEL_NAME, DOUBAO_BASE_URL,
    KNOWLEDGE_BASE_DIR, VECTOR_DB_DIR,
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, PORT
)
from knowledge_loader import load_knowledge_files, chunk_documents
from vector_store import VectorStore
from rag_engine import RAGEngine

app = Flask(__name__)

# 全局状态
rag_engine = None
init_status = {"ready": False, "message": "正在初始化...", "doc_count": 0}

# ============================================
# 照片映射表
# ============================================

IMAGE_MAP = {
    # 北校区
    "beiyuan_01": "/static/images/beiyuan/beiyuan_01.jpg",
    "beiyuan_02": "/static/images/beiyuan/beiyuan_02.jpg",
    "beiyuan_03": "/static/images/beiyuan/beiyuan_03.jpg",
    "beiyuan_04": "/static/images/beiyuan/beiyuan_04.jpg",
    "beiyuan_05": "/static/images/beiyuan/beiyuan_05.jpg",
    # 南校区
    "nanyuan_01": "/static/images/nanyuan/nanyuan_01.jpg",
    "nanyuan_02": "/static/images/nanyuan/nanyuan_02.jpg",
    "nanyuan_03": "/static/images/nanyuan/nanyuan_03.jpg",
    "nanyuan_04": "/static/images/nanyuan/nanyuan_04.jpg",
    "nanyuan_05": "/static/images/nanyuan/nanyuan_05.jpg",
    # 秦汉考务中心
    "qinhan_01": "/static/images/qinhan/qinhan_01.jpg",
    "qinhan_02": "/static/images/qinhan/qinhan_02.jpg",
    "qinhan_03": "/static/images/qinhan/qinhan_03.jpg",
}

# 照片分组（用于首次展示时展示一组照片）
IMAGE_GROUPS = {
    "beiyuan": ["beiyuan_01", "beiyuan_02", "beiyuan_03"],
    "nanyuan": ["nanyuan_01", "nanyuan_02", "nanyuan_03"],
    "qinhan": ["qinhan_01", "qinhan_02", "qinhan_03"],
}

# 会话状态管理
conversation_states = {}

def get_or_create_session(session_id):
    """获取或创建会话状态。"""
    if session_id not in conversation_states:
        conversation_states[session_id] = {
            "photos_shown": False,
            "turn_count": 0,
            "history": []
        }
    return conversation_states[session_id]

def process_photo_tags(answer, session_state):
    """处理回答中的 [图片:xxx] 或 【图片:xxx】 标记。"""
    import re
    # 匹配两种括号格式： [图片:xxx] 和 【图片:xxx】
    photo_pattern = r'[\[【]图片:([^\]】]+)[\]】]'

    if session_state["photos_shown"]:
        # 已展示过，移除所有图片标记
        answer = re.sub(photo_pattern, '', answer).strip()
        return answer, False

    # 首次展示，替换图片标记为 HTML
    def replace_photo_tag(match):
        tag = match.group(1)
        url = IMAGE_MAP.get(tag)
        if url:
            return f'<img src="{url}" alt="驾校实景" style="max-width:100%;border-radius:8px;margin:8px 0;cursor:pointer;" onclick="window.open(\'{url}\')">'
        return ""

    new_answer = re.sub(photo_pattern, replace_photo_tag, answer)
    showed = new_answer != answer
    if showed:
        session_state["photos_shown"] = True
    return new_answer, showed

# ============================================
# HTML 聊天界面
# ============================================

CHAT_HTML = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>鹏翔驾校 AI 助手</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f5f7fa; min-height: 100vh; }
.header { background: linear-gradient(135deg, #1a73e8, #0d47a1); color: #fff; padding: 16px 20px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,.15); }
.header h1 { font-size: 20px; font-weight: 600; }
.header p { font-size: 13px; opacity: .85; margin-top: 4px; }
.chat-container { max-width: 700px; margin: 0 auto; padding: 16px; display: flex; flex-direction: column; height: calc(100vh - 80px); }
.messages { flex: 1; overflow-y: auto; padding: 8px 0; }
.message { margin-bottom: 14px; display: flex; }
.message.user { justify-content: flex-end; }
.message.assistant { justify-content: flex-start; }
.bubble { max-width: 85%; padding: 12px 16px; border-radius: 16px; font-size: 14px; line-height: 1.6; word-break: break-word; }
.message.user .bubble { background: #1a73e8; color: #fff; border-bottom-right-radius: 4px; }
.message.assistant .bubble { background: #fff; color: #333; border-bottom-left-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,.08); }
.bubble img { max-width: 100%; border-radius: 8px; margin: 8px 0; cursor: pointer; }
.typing { color: #999; font-style: italic; padding: 8px 16px; }
.input-area { display: flex; gap: 10px; padding: 12px 0; background: #f5f7fa; }
.input-area input { flex: 1; padding: 12px 16px; border: 1px solid #ddd; border-radius: 24px; font-size: 14px; outline: none; transition: border-color .2s; }
.input-area input:focus { border-color: #1a73e8; }
.input-area button { padding: 10px 24px; background: #1a73e8; color: #fff; border: none; border-radius: 24px; font-size: 14px; cursor: pointer; font-weight: 500; transition: background .2s; }
.input-area button:hover { background: #1557b0; }
.input-area button:disabled { background: #ccc; cursor: not-allowed; }
.status-bar { text-align: center; padding: 6px; font-size: 12px; color: #999; }
.status-bar.ready { color: #4caf50; }
</style>
</head>
<body>
<div class="header">
  <h1>🚗 鹏翔驾校 AI 助手</h1>
  <p>我是小影，您的专属驾校顾问，随时为您解答！</p>
</div>
<div class="chat-container">
  <div class="messages" id="messages">
    <div class="message assistant">
      <div class="bubble">
        您好！我是鹏翔驾校的招生顾问小影 🎉<br><br>
        不管您是想了解价格、课程、考试流程，还是想预约参观我们的训练场，都可以随时问我！<br><br>
        有什么我可以帮您的吗？
      </div>
    </div>
  </div>
  <div class="input-area">
    <input type="text" id="question" placeholder="输入您的问题..." onkeypress="if(event.key==='Enter') sendMessage()">
    <button id="sendBtn" onclick="sendMessage()">发送</button>
  </div>
  <div class="status-bar" id="statusBar">正在初始化...</div>
</div>
<script>
// 会话 ID 管理
function getSessionId() {
    let sid = localStorage.getItem('pengxiang_session_id');
    if (!sid) {
        sid = 'sess_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        localStorage.setItem('pengxiang_session_id', sid);
    }
    return sid;
}

const messages = document.getElementById('messages');
const question = document.getElementById('question');
const sendBtn = document.getElementById('sendBtn');
const statusBar = document.getElementById('statusBar');
const sessionId = getSessionId();

async function checkStatus() {
    try {
        const resp = await fetch('/status');
        const data = await resp.json();
        if (data.ready) {
            statusBar.textContent = '✅ 知识库已就绪 | ' + data.doc_count + ' 个文档块';
            statusBar.className = 'status-bar ready';
        } else {
            statusBar.textContent = data.message || '正在初始化...';
            setTimeout(checkStatus, 2000);
        }
    } catch(e) {
        statusBar.textContent = '连接中...';
        setTimeout(checkStatus, 2000);
    }
}
checkStatus();

function addMessage(role, text) {
    const div = document.createElement('div');
    div.className = 'message ' + role;
    div.innerHTML = '<div class="bubble">' + text.replace(/\\n/g, '<br>') + '</div>';
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
}

function showTyping() {
    const div = document.createElement('div');
    div.className = 'typing';
    div.id = 'typing';
    div.textContent = '小影正在思考...';
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
}

function hideTyping() {
    const el = document.getElementById('typing');
    if (el) el.remove();
}

async function sendMessage() {
    const q = question.value.trim();
    if (!q) return;

    addMessage('user', q);
    question.value = '';
    sendBtn.disabled = true;
    showTyping();

    try {
        const resp = await fetch('/api/chat', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({question: q, session_id: sessionId})
        });
        const data = await resp.json();
        hideTyping();
        addMessage('assistant', data.answer || '抱歉，暂时无法回答您的问题，请稍后再试。');
    } catch(e) {
        hideTyping();
        addMessage('assistant', '抱歉，网络出了点问题，请稍后再试～');
    }
    sendBtn.disabled = false;
}
</script>
</body>
</html>'''

# ============================================
# 路由
# ============================================

@app.route("/")
def index():
    return render_template_string(CHAT_HTML)

@app.route("/healthz")
def healthz():
    return "OK", 200

@app.route("/status")
def status():
    return jsonify(init_status)

@app.route("/api/chat", methods=["POST"])
def chat():
    if not init_status["ready"]:
        return jsonify({"answer": "系统正在初始化，请稍后再试～"}), 503

    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "请提供 question 字段"}), 400

    question = data["question"].strip()
    if not question:
        return jsonify({"error": "问题不能为空"}), 400

    # 会话管理
    session_id = data.get("session_id", "default")
    session_state = get_or_create_session(session_id)
    session_state["turn_count"] += 1
    session_state["history"].append({"role": "user", "content": question})

    try:
        # 构建历史上下文
        history_text = ""
        if len(session_state["history"]) > 1:
            recent = session_state["history"][-5:]  # 最近5轮
            for h in recent[:-1]:
                prefix = "学员" if h["role"] == "user" else "小影"
                history_text += f"{prefix}: {h['content']}\n"

        result = rag_engine.answer(question, history=history_text)

        # 处理照片标记
        answer_text, showed = process_photo_tags(result["answer"], session_state)

        # 记录历史
        session_state["history"].append({"role": "assistant", "content": answer_text})

        return jsonify({
            "answer": answer_text,
            "sources": result["sources"],
            "photos_shown": session_state["photos_shown"]
        })
    except Exception as e:
        return jsonify({"answer": f"抱歉，处理您的问题时出错了：{str(e)}"}), 500

# ============================================
# 初始化
# ============================================

def initialize_rag():
    """后台初始化 RAG 引擎。"""
    global rag_engine, init_status

    try:
        init_status["message"] = "正在加载知识库..."
        print("📚 加载知识库文件...")
        documents = load_knowledge_files(KNOWLEDGE_BASE_DIR)
        if not documents:
            init_status["message"] = "知识库为空，请检查 knowledge_base/ 目录"
            return

        init_status["message"] = "正在处理文档..."
        print("🔧 切分文档...")
        chunks = chunk_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

        init_status["message"] = "正在构建向量索引..."
        print("🧠 构建向量数据库...")
        vector_store = VectorStore(VECTOR_DB_DIR)
        if vector_store.count() == 0:
            vector_store.add_documents(chunks)

        init_status["message"] = "正在初始化 AI 引擎..."
        print("🤖 初始化 RAG 引擎...")
        config = {
            "doubao_api_key": DOUBAO_API_KEY,
            "doubao_model_name": DOUBAO_MODEL_NAME,
            "doubao_base_url": DOUBAO_BASE_URL,
        }
        rag_engine = RAGEngine(vector_store, config)

        init_status = {
            "ready": True,
            "message": "初始化完成",
            "doc_count": vector_store.count()
        }
        print(f"✅ 初始化完成！文档块数: {vector_store.count()}")

    except Exception as e:
        init_status["message"] = f"初始化失败: {str(e)}"
        print(f"❌ 初始化失败: {e}")

# 启动后台初始化线程
init_thread = threading.Thread(target=initialize_rag, daemon=True)
init_thread.start()

# ============================================
# 主入口
# ============================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)