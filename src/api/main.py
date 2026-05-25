from models.micro_parser_llm import MicroParserLLM
from models.pgvector_search import PGHybridRetriever
from models.bge_reranker import BGEReranker
from models.qwen_llm import QwenLocalLLM
from pipeline.rag_orchestrator import RAGPipeline

def main():
    print("Loading AI Models into memory...")
    
    # 1. Khởi tạo Database Config
    db_config = {
        "dbname": "rag_database",
        "user": "admin",
        "password": "secretpassword",
        "host": "localhost",
        "port": 5432
    }
    
    # 2. Bơm (Inject) các module thực tế vào Pipeline
    # Lưu ý: Sửa đường dẫn model_path trỏ tới file GGUF bạn đã tải
    pipeline = RAGPipeline(
        parser_llm=MicroParserLLM(model_path="models/final_gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf"),
        retriever=PGHybridRetriever(db_config=db_config),
        reranker=BGEReranker(),
        generator_llm=QwenLocalLLM(model_path="models/final_gguf/qwen2.5-1.5b-instruct-q4_k_m.gguf")
    )
    
    # 3. Chạy thử nghiệm
    query = "Làm sao để kết hợp 2 dataframes dựa trên một cột chung trong thư viện pandas?"
    answer = pipeline.run(query)
    
    print("🤖 ANSWER:")
    print(answer)

if __name__ == "__main__":
    main()