FROM python:3.11-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制源码
COPY . .

# 创建必要目录
RUN mkdir -p vector_db

# 设置环境变量
ENV PORT=8080

# 启动命令
CMD ["python", "-m", "src.app"]