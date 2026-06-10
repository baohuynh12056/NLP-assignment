import json
from data_pipeline.chunker import attach_embeddings
from database.db_manager import DatabaseManager
from utils.config_loader import GLOBAL_CONFIG

def quick_ingest(file_path: str):
    # 1. Đọc dữ liệu từ file JSONL
    print(f"--- Đang đọc dữ liệu từ {file_path} ---")
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    # 2. Tạo Embedding cho dữ liệu (Bước này bắt buộc để Semantic Search hoạt động)
    # attach_embeddings sẽ tự gọi mô hình BGE để tạo vector cho từng hàm
    print("--- Đang tạo Embedding (vui lòng đợi...) ---")
    processed_records = attach_embeddings(records)
    
    # 3. Đẩy vào Database
    print("--- Đang đẩy vào PostgreSQL ---")
    db_config = GLOBAL_CONFIG.get("database", {})
    db = DatabaseManager(db_config=db_config)
    
    count = db.upsert_functions(processed_records)
    print(f"--- Xong! Đã nạp thành công {count} hàm vào Database. ---")

if __name__ == "__main__":
    quick_ingest("data/chunks/functions.jsonl")