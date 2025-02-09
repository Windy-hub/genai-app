# 使用 Python 3.11 作为基础镜像
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 复制所有文件到容器中
COPY . /app

# 安装 Python 依赖（如果有 requirements.txt）
RUN pip install --no-cache-dir -r requirements.txt || true

# 运行 Python 代码
CMD ["python", "chatbotlate.py"]

