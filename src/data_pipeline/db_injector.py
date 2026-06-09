import json
import psycopg2
from sentence_transformers import SentenceTransformer
from utils.config_loader import GLOBAL_CONFIG
from utils.logger import get_logger

logger = get_logger(__name__)

class DatabaseInjector:
    """Embeds the search_text and inserts records into PostgreSQL with pgvector."""

    def __init__(self, embed_model_name: str = "BAAI/bge-small-en-v1.5"):
        logger.info(f"Loading embedding model: {embed_model_name}")
        self.embed_model = SentenceTransformer(embed_model_name)
        
        db_config = GLOBAL_CONFIG.get("database", {})
        self.conn = psycopg2.connect(**db_config)
        self.cursor = self.conn.cursor()

    def insert_records(self, raw_data_path: str):
        with open(raw_data_path, 'r', encoding='utf-8') as f:
            records = json.load(f)

        logger.info(f"Embedding and inserting {len(records)} records into Database...")
        
        insert_query = """
            INSERT INTO functions 
            (library_name, module_name, func_name, signature, docstring, parameters, search_text, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """

        for i, record in enumerate(records):
            # 1. Encode text to vector
            search_text = record["search_text"]
            embedding = self.embed_model.encode(search_text, normalize_embeddings=True).tolist()

            # 2. Execute SQL Insert
            try:
                self.cursor.execute(insert_query, (
                    record["library_name"],
                    record["module_name"],
                    record["func_name"],
                    record["signature"],
                    record["docstring"],
                    json.dumps(record["parameters"]), # Convert dict to JSONB string
                    search_text,
                    embedding
                ))
            except Exception as e:
                logger.error(f"DB Insert error for {record['func_name']}: {e}")
                self.conn.rollback()
                continue

            # Commit every 100 records
            if (i + 1) % 100 == 0:
                self.conn.commit()
                logger.info(f"Inserted {i + 1} records...")

        self.conn.commit()
        logger.info("Database injection completed successfully.")
        self.close()

    def close(self):
            """Safely close database connections"""
            if hasattr(self, 'conn') and self.conn:
                self.cursor.close()
                self.conn.close()
                logger.info("Database connections closed.")


if __name__ == "__main__":
    # Khởi tạo máy bơm dữ liệu
    injector = DatabaseInjector()
    
    # Bấm nút bơm dữ liệu từ file JSON vào Database
    # LƯU Ý: Đổi "data/functions.json" thành đường dẫn thực tế chứa file dữ liệu JSON của bạn
    data_path = "data/raw/pytorch_deep_raw.json" 
    
    try:
        injector.insert_records(data_path)
    except Exception as e:
        logger.error(f"Lỗi khi nạp dữ liệu: {e}")
    finally:
        # Đảm bảo luôn đóng kết nối khi xong việc
        injector.close()