"""
鹏翔驾校 AI 助手 - Flask Web 应用
====================================
提供 Web 测试界面和 REST API 接口。
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, request, jsonify

from config import (
        DASHSCOPE_API_KEY, DEEPSEEK_API_KEY, DOUBAO_API_KEY,
        LLM_PROVIDER, QWEN_MODEL_NAME, DEEPSEEK_MODEL_NAME,
        DOUBAO_MODEL_NAME, DOUBAO_BASE_URL,
        KNOWLEDGE_BASE_DIR, VECTOR_DB_DIR, CHUNK_SIZE, CHUNK_OVERLAP,
        TOP_K, PORT
    )

app = Flask(__name__)

# 全局变量
rag_engine = None
vector_store = None
docs_loaded = False
initialized = False


def init_rag_system():
    """初始化 RAG 系统。"""
    global rag_engine, vector_store, docs_loaded, initialized

    from knowledge_loader import load_knowledge_files, chunk_documents
    from vector_store import VectorStore
    from rag_engine import RAGEngine

    print("=" * 50)
    print("鹏翔驾校 AI 助手 - 初始化中...")
    print("=" * 50)

    # 1. 加载知识库
    print("步骤 1/4: 加载知识库文件...")
    documents = load_knowledge_files(KNOWLEDGE_BASE_DIR)
    if not documents:
        print("未找到知识库文件")
        return False

    # 2. 文本切片
    print("步骤 2/4: 文本切片...")
    chunks = chunk_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    # 3. 初始化向量数据库
    print("步骤 3/4: 初始化向量数据库...")
    vector_store = VectorStore(persist_dir=VECTOR_DB_DIR)

    existing_count = vector_store.count()
    if existing_count == 0:
        vector_store.add_documents(chunks)

    print(f"向量数据库状态: {vector_store.count()} 条文档块")

    # 4. 初始化 RAG 引擎
    print("步骤 4/4: 初始化 RAG 引擎...")
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
    initialized = True

    print("=" * 50)
    print("初始化完成！")
    if not rag_engine.llm_ready:
        print("注意: API Key 未配置，将使用模拟回答模式")
    print(f"当前模型: {LLM_PROVIDER}")
    print("=" * 50)
    return True


# ============================================
# 路由
# ============================================

# 健康检查端点（在主模块中通过 add_url_rule 注册，确保 Railway 路由正常工作）
def healthz():
    """健康检查端点。"""
    return "OK", 200, {"Content-Type": "text/plain"}


@app.route("/")
def index():
    """Web 测试界面首页。"""
    if rag_engine:
        return render_template("index.html", llm_ready=rag_engine.llm_ready)
    return render_template("index.html", llm_ready=False)


@app.route("/api/ask", methods=["POST"])
def ask():
    """RAG 问答接口。"""
    if not docs_loaded:
        return jsonify({"error": "系统初始化中，请稍后再试"}), 503

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
        "initialized": initialized,
        "llm_ready": rag_engine.llm_ready if rag_engine else False,
        "vector_count": vector_store.count() if vector_store else 0,
        "model": LLM_PROVIDER
    })


@app.route("/api/rebuild", methods=["POST"])
def rebuild():
    """重建向量数据库。"""
    global vector_store, rag_engine
    try:
        from knowledge_loader import load_knowledge_files, chunk_documents
        from rag_engine import RAGEngine

        documents = load_knowledge_files(KNOWLEDGE_BASE_DIR)
        chunks = chunk_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

        vector_store.clear()
        vector_store.add_documents(chunks)

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
# 启动
# ============================================

if __name__ == "__main__":
    # 先初始化，再启动 Web 服务
    init_rag_system()

    # 手动注册健康检查路由（确保 Railway 健康检查正常工作）
    app.add_url_rule("/healthz", "healthz", healthz, methods=["GET"])
    print("✅ /healthz 路由已注册")

    print(f"启动 Web 服务，端口: {PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False)