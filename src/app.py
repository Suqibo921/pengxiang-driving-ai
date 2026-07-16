"""
鹏翔驾校 AI 助手 - Flask Web 应用
====================================
提供 Web 测试界面和 REST API 接口。
"""

import os
import sys
import threading
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, request, jsonify

# 延迟导入（在后台初始化线程中导入，不阻塞 Web 启动）

from config import (
        DASHSCOPE_API_KEY, DEEPSEEK_API_KEY, DOUBAO_API_KEY,
        LLM_PROVIDER, QWEN_MODEL_NAME, DEEPSEEK_MODEL_NAME,
        DOUBAO_MODEL_NAME, DOUBAO_BASE_URL,
        KNOWLEDGE_BASE_DIR, VECTOR_DB_DIR, CHUNK_SIZE, CHUNK_OVERLAP,
        TOP_K, PORT
    )

app = Flask(__name__)

# 全局变量：RAG 引擎和向量数据库
rag_engine = None
vector_store = None
docs_loaded = False


def init_rag_system():
    """
    初始化 RAG 系统（延迟导入，不阻塞 Web 启动）：
    1. 加载知识库文件
    2. 切分为文本块
    3. 向量化并存入数据库
    4. 初始化 RAG 引擎
    """
    global rag_engine, vector_store, docs_loaded

    # 延迟导入（避免阻塞 gunicorn 启动）
    from knowledge_loader import load_knowledge_files, chunk_documents
    from vector_store import VectorStore
    from rag_engine import RAGEngine

    print("=" * 50)
    print("🚗 鹏翔驾校 AI 助手 - 初始化中...")
    print("=" * 50)

    # 1. 加载知识库
    print("\n📖 步骤 1/4: 加载知识库文件...")
    documents = load_knowledge_files(KNOWLEDGE_BASE_DIR)
    if not documents:
        print("❌ 未找到知识库文件，请确认 knowledge_base 目录存在")
        return False

    # 2. 文本切片
    print("\n✂️ 步骤 2/4: 文本切片...")
    chunks = chunk_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    # 3. 初始化向量数据库并添加文档
    print("\n🗄️ 步骤 3/4: 初始化向量数据库...")
    vector_store = VectorStore(
        persist_dir=VECTOR_DB_DIR
    )

    # 检查是否已有数据
    existing_count = vector_store.count()
    if existing_count > 0:
        print(f"🔄 向量数据库中已有 {existing_count} 条记录，跳过重新索引")
    else:
        vector_store.add_documents(chunks)

    print(f"📊 向量数据库状态: {vector_store.count()} 条文档块")

    # 4. 初始化 RAG 引擎
    print("\n🤖 步骤 4/4: 初始化 RAG 引擎...")
    config = {
        "llm_provider": LLM_PROVIDER,
        "dashscope_api_key": DASHSCOPE_API_KEY,
        "deepseek_api_key": DEEPSEEK_API_KEY,
        "doubao_api_key": DOUBAO_API_KEY,
        "qwen_model_name": QWEN_MODEL_NAME,
        "deepseek_model_name": DEEPSEEK_MODEL_NAME,
        "doubao_model_name": DOUBAO_MODEL_NAME,
        "doubao_base_url": DOUBAO_BASE_URL,
    }
    rag_engine = RAGEngine(vector_store, config)

    docs_loaded = True
    print("\n" + "=" * 50)
    print("✅ 初始化完成！")
    if not rag_engine.llm_ready:
        print("⚠️ 注意: API Key 未配置，将使用模拟回答模式")
        print("   请编辑 src/config.py 填入你的 API Key")
    print(f"🔧 当前模型: {LLM_PROVIDER}")
    print("=" * 50)
    return True


# ============================================
# 路由
# ============================================

@app.route("/healthz")
def healthz():
    """健康检查端点，用于 Railway 的负载均衡器。"""
    return "OK", 200


@app.route("/")
def index():
    """Web 测试界面首页。"""
    return render_template("index.html", llm_ready=rag_engine.llm_ready if rag_engine else False)


@app.route("/api/ask", methods=["POST"])
def ask():
    """
    RAG 问答接口。
    POST 参数: {"question": "用户问题", "history": "历史对话(可选)"}
    返回: {"answer": "回答", "sources": ["来源文件"], "context": "参考原文"}
    """
    if not docs_loaded:
        return jsonify({"error": "系统未初始化，请先初始化知识库"}), 500

    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "请提供问题"}), 400

    question = data["question"].strip()
    if not question:
        return jsonify({"error": "问题不能为空"}), 400

    history = data.get("history", "")

    result = rag_engine.answer(question, history=history, top_k=TOP_K)
    return jsonify(result)


@app.route("/api/status", methods=["GET"])
def status():
    """系统状态接口。"""
    return jsonify({
        "ready": docs_loaded,
        "llm_ready": rag_engine.llm_ready if rag_engine else False,
        "vector_count": vector_store.count() if vector_store else 0,
        "model": LLM_PROVIDER
    })


@app.route("/api/rebuild", methods=["POST"])
def rebuild():
    """重建向量数据库（知识库更新后调用）。"""
    global vector_store, rag_engine
    try:
        # 重新加载知识库
        documents = load_knowledge_files(KNOWLEDGE_BASE_DIR)
        chunks = chunk_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

        # 清空并重建向量库
        vector_store.clear()
        vector_store.add_documents(chunks)

        # 重建 RAG 引擎
        config = {
            "llm_provider": LLM_PROVIDER,
            "dashscope_api_key": DASHSCOPE_API_KEY,
            "deepseek_api_key": DEEPSEEK_API_KEY,
            "doubao_api_key": DOUBAO_API_KEY,
            "qwen_model_name": QWEN_MODEL_NAME,
            "deepseek_model_name": DEEPSEEK_MODEL_NAME,
            "doubao_model_name": DOUBAO_MODEL_NAME,
            "doubao_base_url": DOUBAO_BASE_URL,
        }
        rag_engine = RAGEngine(vector_store, config)

        return jsonify({"status": "ok", "count": vector_store.count()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================
# 启动（后台初始化，不阻塞 Web 服务启动）
# ============================================

def _init_background():
    """后台线程中执行初始化。"""
    global rag_engine, vector_store, docs_loaded
    try:
        init_rag_system()
        print(f"\n🌐 服务就绪，端口: {PORT}")
    except Exception as e:
        print(f"⚠️ 后台初始化异常: {e}")

# 启动后台初始化线程，Web 服务立即就绪
init_thread = threading.Thread(target=_init_background, daemon=True)
init_thread.start()
print("⏳ 系统正在后台初始化中，Web 服务已启动...")

if __name__ == "__main__":
    # 本地开发：直接运行
    print(f"\n🌐 启动 Web 服务，端口: {PORT} (开发模式)")
    app.run(host="0.0.0.0", port=PORT, debug=False)