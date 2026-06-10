import argparse
import json
from pathlib import Path
from data_pipeline.chunker import attach_embeddings
from database.db_manager import DatabaseManager
from utils.config_loader import GLOBAL_CONFIG
from utils.logger import get_logger

logger = get_logger(__name__)

def load_jsonl(path: str) -> list:
    """Safely read a JSONL file."""
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def run_ingest(file_path: str, embedding_model: str):
    logger.info("=== STARTING DATA INGESTION PROCESS ===")
    
    # 1. Load available JSONL data into memory
    logger.info(f"Reading data from: {file_path}")
    try:
        records = load_jsonl(file_path)
        logger.info(f"Loaded {len(records)} raw text records.")
    except Exception as e:
        logger.error(f"Failed to read data file: {str(e)}")
        return

    # 2. Generate Vector Embeddings (Required for Semantic Search to work)
    logger.info(f"Generating Vector Embeddings using the '{embedding_model}' model...")
    # This function uses SentenceTransformer to embed text into a 384-dimensional vector
    processed_records = attach_embeddings(records, model_name=embedding_model)

    # 3. Connect and Initialize the Database
    logger.info("Connecting to PostgreSQL and checking Schema...")
    db_config = GLOBAL_CONFIG.get("database", {})
    db = DatabaseManager(db_config=db_config)
    
    # Ensure the 'functions' table and search indexes always exist
    db.init_schema() 

    # 4. Perform Data Upsert (Update if chunk_id exists, insert if new)
    logger.info("Pushing data to the Database...")
    count = db.upsert_functions(processed_records)
    
    logger.info(f"=== SUCCESS: Ingested {count}/{len(records)} functions into PostgreSQL! ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick tool to ingest RAG data into the Database")
    parser.add_argument("--file", default="data/chunks/functions.jsonl", help="Path to the JSONL data file")
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5", help="The embedding model to use")
    
    args = parser.parse_args()
    run_ingest(file_path=args.file, embedding_model=args.model)