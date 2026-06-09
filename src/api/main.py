import os

from models.query_parser import MicroParserLLM
from models.retriever import PGHybridRetriever
from models.reranker import BGEReranker
from models.llm import QwenLocalLLM
from pipeline.rag_orchestrator import RAGPipeline

def main():
    print("Loading AI Models into memory...")
    parser_model_path = os.getenv(
        "PARSER_MODEL_PATH",
        "models/final_gguf/qwen3-0.6b-instruct-q4_k_m.gguf",
    )
    generator_model_path = os.getenv(
        "GENERATOR_MODEL_PATH",
        "models/final_gguf/qwen3-4b-instruct-q4_k_m.gguf",
    )
    
    # 1. Khởi tạo Database Config
    db_config = {
        "dbname": os.getenv("POSTGRES_DB", "rag_database"),
        "user": os.getenv("POSTGRES_USER", "admin"),
        "password": os.getenv("POSTGRES_PASSWORD", "secretpassword"),
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT", "5432")),
    }
    
    # 2. Bơm (Inject) các module thực tế vào Pipeline
    # Lưu ý: Có thể override đường dẫn bằng PARSER_MODEL_PATH và GENERATOR_MODEL_PATH.
    pipeline = RAGPipeline(
        parser_llm=MicroParserLLM(model_path=parser_model_path),
        retriever=PGHybridRetriever(db_config=db_config),
        reranker=BGEReranker(),
        generator_llm=QwenLocalLLM(model_path=generator_model_path)
    )
    
    # 3. Chạy thử nghiệm
    query = "Làm sao để kết hợp 2 dataframes dựa trên một cột chung trong thư viện pandas?"
    answer = pipeline.run(query)
    
    print("🤖 ANSWER:")
    print(answer)

if __name__ == "__main__":
    main()
