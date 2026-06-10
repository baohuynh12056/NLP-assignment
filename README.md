# NLP Assignment: Mini Code-Assistance RAG

Repo này xây dựng một hệ thống hỏi đáp tài liệu API cho các thư viện Python phổ biến.
Người dùng hỏi bằng ngôn ngữ tự nhiên, hệ thống sẽ rewrite query, retrieve tài liệu
liên quan từ PostgreSQL/pgvector, rerank kết quả, rồi trả lời qua giao diện web chat.

## Tính năng chính

- Hybrid search: kết hợp keyword search của PostgreSQL và semantic search bằng embedding.
- Reranking: dùng BGE reranker để chọn context tốt hơn trước khi trả lời.
- Local LLM: dùng Qwen GGUF qua `llama-cpp-python`, không cần gọi API ngoài khi demo.
- Fast mode: trả lời nhanh dựa trên retrieved docs và các rule cho câu hỏi phổ biến.
- Full mode: dùng local Qwen model để sinh câu trả lời tự nhiên hơn.
- Web chat UI: giao diện chat gọn, có quick prompts, code block, nút copy code, drawer nguồn tài liệu.
- Deploy demo nhanh: có thể public bằng Cloudflare Tunnel hoặc ngrok.

## Thư viện trong benchmark

Dataset hiện tập trung vào các thư viện:

`pandas`, `numpy`, `sklearn`, `torch`, `tensorflow`, `matplotlib`, `scipy`,
`seaborn`, `requests`, `fastapi`.

## Kiến trúc pipeline

1. `QueryProcessor`
   - Rewrite câu hỏi thành truy vấn rõ hơn cho vector search.
   - Extract filter như `library_name`.
   - Có rule-based fallback cho một số câu phổ biến như `pandas.merge`,
     `pandas.read_csv`, `sklearn.model_selection.train_test_split`.

2. `PGHybridRetriever`
   - Gọi PostgreSQL full-text search cho keyword route.
   - Gọi pgvector cho semantic route.

3. `PostgreSQLHybridSearch`
   - Chuẩn hóa điểm keyword/semantic.
   - Fusion kết quả hai route.
   - Tăng `ivfflat.probes` để retrieval ổn định hơn.

4. `BGEReranker`
   - Rerank top candidates bằng cross-encoder.

5. `PromptBuilder`
   - Đóng gói context docs ngắn gọn.
   - Thêm yêu cầu trả lời theo ngôn ngữ người dùng.
   - Giảm nguy cơ prompt quá dài bằng cách truncate docs.

6. `QwenLocalLLM`
   - Sinh câu trả lời trong Full mode.
   - Dùng `/no_think` và lọc `<think>...</think>` nếu model sinh ra.

7. `FastAPI`
   - Serve API `/ask`, `/health`.
   - Serve web chat UI ở `/`.

## Web Chat Demo

Giao diện web nằm ở:

- `src/api/chat_ui.html`
- `src/api/main.py`

Trang web được thiết kế như một chatbot đơn giản:

- Header cố định ở trên.
- Vùng hội thoại cuộn riêng.
- Ô nhập dính ở cuối màn hình.
- Nhấn `Enter` để gửi, `Shift+Enter` để xuống dòng.
- Có quick prompts cho các câu hỏi mẫu.
- Câu trả lời có render code block và nút `Copy`.
- Nguồn tài liệu và setting nằm trong drawer bên phải.
- Không có password/auth, phù hợp demo nhanh trong mạng local hoặc qua tunnel.

Hai chế độ trả lời:

- `Fast`: nhanh hơn, dùng retrieval và response template/rule.
- `Full`: chậm hơn, dùng local Qwen model để sinh câu trả lời tự nhiên.

Ví dụ câu hỏi:

- `How do I merge two pandas dataframes on a common column?`
- `How do I read a CSV file with pandas?`
- `How do I split data into train and test sets with sklearn?`
- `Cách đọc file CSV bằng pandas như thế nào?`

## Cài đặt

Tạo môi trường Python:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Trên Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Khởi động database

Repo dùng PostgreSQL kèm pgvector trong Docker:

```bash
cd docker
docker compose up -d
cd ..
```

Thông tin mặc định:

- Database: `rag_database`
- User: `admin`
- Password: `secretpassword`
- Port: `5432`

Schema nằm ở:

- `src/database/schema.sql`

## Build dataset

Dataset được build từ docstring của các thư viện Python đã cài.

Cài các thư viện nguồn data:

```bash
pip install -r requirements-data-libs.txt
```

Build chunks và training data:

```bash
PYTHONPATH=src python src/data_pipeline/dataset_builder.py
```

Output chính:

- `data/chunks/functions.jsonl`
- `data/parsed/function_docs.jsonl`
- `data/training/retriever_train.jsonl`
- `data/training/retriever_test.jsonl`

Ingest vào PostgreSQL:

```bash
PYTHONPATH=src python src/data_pipeline/dataset_builder.py --ingest
```

Trên Windows PowerShell:

```powershell
$env:PYTHONPATH='src'
python src\data_pipeline\dataset_builder.py --ingest
```

## Model GGUF

Đặt model local vào:

- `models/final_gguf/qwen3-0.6b-instruct-q4_k_m.gguf`
- `models/final_gguf/qwen3-4b-instruct-q4_k_m.gguf`

Các file `.gguf` không được commit lên GitHub vì dung lượng lớn. Có thể override path bằng biến môi trường:

```bash
export PARSER_MODEL_PATH=/path/to/parser.gguf
export GENERATOR_MODEL_PATH=/path/to/qwen3-4b.gguf
```

Windows PowerShell:

```powershell
$env:PARSER_MODEL_PATH='D:\path\to\parser.gguf'
$env:GENERATOR_MODEL_PATH='D:\path\to\qwen3-4b.gguf'
```

## Chạy API và web local

```bash
PYTHONPATH=src uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Windows PowerShell:

```powershell
$env:PYTHONPATH='src'
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Mở trình duyệt:

- Web chat: `http://127.0.0.1:8000/`
- API docs: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

## Public demo bằng Cloudflare Tunnel

Cách nhanh để máy khác truy cập được web đang chạy local:

```bash
cloudflared tunnel --url http://127.0.0.1:8000
```

Cloudflare sẽ trả về URL dạng:

```text
https://something.trycloudflare.com
```

Gửi URL này cho người khác để demo. Đây là quick tunnel nên URL có thể thay đổi sau mỗi lần chạy lại.

## API

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Ask endpoint:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"How do I read a CSV file with pandas?","mode":"fast"}'
```

Request body:

```json
{
  "query": "How do I read a CSV file with pandas?",
  "mode": "fast"
}
```

`mode` nhận một trong hai giá trị:

- `fast`
- `full`

Response gồm:

- `query`: câu hỏi gốc.
- `optimized_query`: câu query sau rewrite.
- `filters`: filter được áp dụng, ví dụ `library_name`.
- `answer`: câu trả lời.
- `sources`: danh sách chunks được dùng làm nguồn.

## Ghi chú hiệu năng

- Request đầu tiên sau khi restart có thể chậm vì phải load embedding model/reranker/LLM.
- Các request sau thường nhanh hơn.
- Fast mode phù hợp demo vì không cần sinh bằng model lớn.
- Full mode trả lời tự nhiên hơn nhưng chậm hơn vì dùng local Qwen.

## Fine-tune retriever

Training data mặc định nằm ở:

- `data/training/retriever_train.jsonl`
- `data/training/retriever_test.jsonl`

Chạy fine-tune:

```bash
PYTHONPATH=src python src/fine_tuning/train_retriver.py
```

Model output mặc định:

- `models/checkpoints/bge-small-code-assistant`

## Các file quan trọng

- `src/api/main.py`: FastAPI app, schema request, endpoint `/ask`.
- `src/api/chat_ui.html`: giao diện web chat.
- `src/pipeline/rag_orchestrator.py`: điều phối RAG pipeline.
- `src/pipeline/query_processor.py`: rewrite query và filter.
- `src/pipeline/prompt_builder.py`: build prompt/context cho LLM.
- `src/database/hybrid_search.py`: hybrid search PostgreSQL/pgvector.
- `src/models/llm.py`: local Qwen GGUF wrapper.
- `src/core/schemas.py`: schema dữ liệu request/response/chunk.
