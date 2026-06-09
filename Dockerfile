# Sử dụng base image Python gọn nhẹ
FROM python:3.10-slim

# Thiết lập thư mục làm việc trong container
WORKDIR /app

# Cài đặt các thư viện hệ thống cần thiết (để build llama-cpp và psycopg2)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy file requirements và cài đặt thư viện
# (Bạn cần đảm bảo đã tạo file requirements.txt chứa fastapi, uvicorn, llama-cpp-python, sentence-transformers, psycopg2-binary...)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn, cấu hình và thư mục models vào container
COPY configs/ ./configs/
COPY src/ ./src/

# Đảm bảo Python có thể import được các module trong thư mục src/
ENV PYTHONPATH="${PYTHONPATH}:/app/src"

# Mở cổng 8000 cho Web API
EXPOSE 8000

# Lệnh chạy server FastAPI khi container khởi động
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]