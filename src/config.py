"""
鹏翔驾校 AI 助手 - 配置文件
================================
支持环境变量覆盖（云端部署），方便在 Railway/Zeabur 等平台设置。

优先级：环境变量 > .env 文件 > 硬编码默认值

=== 环境变量列表 ===
DOUBAO_API_KEY      - 豆包/火山引擎 API Key（必填）
DOUBAO_MODEL_NAME   - 模型名（默认 doubao-seed-2-0-mini-260428）
DOUBAO_BASE_URL     - API 地址（默认 https://ark.cn-beijing.volces.com/api/v3）
LLM_PROVIDER        - 模型提供商（doubao / qwen / deepseek）
PORT                - 服务端口（默认 5000）
"""

import os
import sys

# ============================================
# 0. 加载 .env 文件（本地开发用）
# ============================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dotenv_path = os.path.join(BASE_DIR, ".env")
if os.path.exists(dotenv_path):
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path)
        print(f"✅ 已加载 .env 文件: {dotenv_path}")
    except ImportError:
        # dotenv 未安装时静默忽略
        pass


# ============================================
# 1. LLM 配置 - 优先读环境变量，再读 .env，再读硬编码默认值
# ============================================

# 豆包/火山引擎 API Key（云端部署时在平台设置环境变量）
DOUBAO_API_KEY = os.environ.get(
    "DOUBAO_API_KEY",
    "YOUR_DOUBAO_API_KEY_HERE"
)

# 通义千问 API Key
DASHSCOPE_API_KEY = os.environ.get(
    "DASHSCOPE_API_KEY",
    "YOUR_DASHSCOPE_API_KEY_HERE"
)

# DeepSeek API Key
DEEPSEEK_API_KEY = os.environ.get(
    "DEEPSEEK_API_KEY",
    "YOUR_DEEPSEEK_API_KEY_HERE"
)


# ============================================
# 2. 模型选择
# ============================================

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "doubao")

QWEN_MODEL_NAME = os.environ.get("QWEN_MODEL_NAME", "qwen3-max")
DEEPSEEK_MODEL_NAME = os.environ.get("DEEPSEEK_MODEL_NAME", "deepseek-chat")
DOUBAO_MODEL_NAME = os.environ.get("DOUBAO_MODEL_NAME", "doubao-seed-2-0-mini-260428")
DOUBAO_BASE_URL = os.environ.get("DOUBAO_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")


# ============================================
# 3. 知识库配置
# ============================================

KNOWLEDGE_BASE_DIR = os.path.join(BASE_DIR, "knowledge_base")
VECTOR_DB_DIR = os.path.join(BASE_DIR, "vector_db")

# 文本切片参数
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80

# 检索参数
TOP_K = 5

# 服务端口
PORT = int(os.environ.get("PORT", 5000))