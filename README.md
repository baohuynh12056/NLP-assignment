# NLP Assignment: Mini Code-Assistance RAG

Mục tiêu repo là xây dựng một pipeline code assistance nhỏ gọn: người dùng hỏi về
function trong các Python library có trong database, hệ thống retrieve đúng tài
liệu API rồi dùng LLM local để trả lời có context.

## Kiến trúc

1. `QueryProcessor` gọi mini LLM để rewrite query và extract metadata filter.
2. `PGHybridRetriever` chạy hai route:
   - Keyword route: PostgreSQL full-text search.
   - Semantic route: `sentence-transformers` embedding + pgvector.
3. `PostgreSQLHybridSearch` normalize score hai route và fusion kết quả.
4. `BGEReranker` rerank top candidates bằng cross-encoder nhỏ.
5. `PromptBuilder` đóng gói top-k chunks thành RAG context.
6. `QwenLocalLLM` sinh câu trả lời cuối qua `llama-cpp-python`.

## 10 library benchmark ban đầu

`pandas`, `numpy`, `sklearn`, `torch`, `tensorflow`, `matplotlib`, `scipy`,
`seaborn`, `requests`, `fastapi`.

## Cài đặt

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Khởi động database

```bash
cd docker
docker compose up -d
cd ..
```

Schema nằm ở `src/database/schema.sql` và được mount vào container PostgreSQL.

## Build dataset từ thư viện đã cài

Baseline chính của project lấy data trực tiếp từ docstring của các Python
library đã cài. Cách này rẻ, sạch, reproducible và hợp với bài toán hỏi đáp về
function trong library.

Nếu muốn tải riêng các package làm nguồn data trước:

```bash
bash scripts/download_data_libs.sh
```

Lệnh này chỉ tải wheel/source vào `data/package_cache/`, chưa cài vào môi
trường Python. Để cài trực tiếp:

```bash
pip install -r requirements-data-libs.txt
```

```bash
PYTHONPATH=src python3 src/data_pipeline/dataset_builder.py
```

Lệnh này introspect các library mục tiêu, tạo function chunks và sinh vài query
rule-based cho mỗi function.

Output:

- `data/chunks/functions.jsonl`
- `data/training/retriever_train.jsonl`
- `data/training/retriever_test.jsonl`
- `data/benchmark/benchmark_queries.jsonl`

Nếu muốn nhiều query rule-based hơn cho mỗi function:

```bash
PYTHONPATH=src python3 src/data_pipeline/dataset_builder.py --examples-per-function 4
```

## Build thêm từ raw docs

Raw PDF/HTML/DOCX chỉ là nguồn phụ trợ khi cần mở rộng data từ official docs.
Đặt tài liệu library vào dạng:

```text
data/raw/
  pandas/
    api.html
    user_guide.pdf
  numpy/
    reference.docx
```

Parser hiện hỗ trợ `.pdf`, `.html`, `.htm`, `.docx`, `.md`, `.rst`, `.txt`.

```bash
PYTHONPATH=src python3 src/data_pipeline/dataset_builder.py --use-raw-docs
```

Để build embedding và nạp vào PostgreSQL:

```bash
PYTHONPATH=src python3 src/data_pipeline/dataset_builder.py --ingest
```

Embedding mặc định dùng `BAAI/bge-small-en-v1.5` nên schema đang để
`embedding vector(384)`. Nếu đổi embedding model khác, sửa dimension trong
`src/database/schema.sql`.

## Sinh training data bằng GPT-4o

Script này đọc `data/chunks/functions.jsonl`, gọi OpenAI API, và sinh query,
answer, train/test/benchmark bằng Structured Outputs.

```bash
export OPENAI_API_KEY=...
PYTHONPATH=src python3 src/data_pipeline/llm_data_generator.py \
  --functions data/chunks/functions.jsonl \
  --model gpt-4o \
  --limit 200 \
  --examples-per-function 3
```

Output:

- `data/training/llm_retriever_train.jsonl`
- `data/training/llm_retriever_test.jsonl`
- `data/benchmark/llm_benchmark.jsonl`

Nạp dataset examples vào database:

```bash
PYTHONPATH=src python3 src/data_pipeline/dataset_builder.py \
  --ingest-datasets \
  --dataset-files data/training/llm_retriever_train.jsonl data/training/llm_retriever_test.jsonl data/benchmark/llm_benchmark.jsonl
```

## Chạy demo RAG

Đặt file GGUF vào:

- `models/final_gguf/qwen3-0.6b-instruct-q4_k_m.gguf`
- `models/final_gguf/qwen3-4b-instruct-q4_k_m.gguf`

Hoặc override bằng env vars:

```bash
export PARSER_MODEL_PATH=/path/to/parser.gguf
export GENERATOR_MODEL_PATH=/path/to/qwen3-4b.gguf
```

Sau đó chạy:

```bash
PYTHONPATH=src python3 src/api/main.py
```

## Fine-tune retriever

Dataset mặc định là `data/training/retriever_train.jsonl`, được sinh từ
docstring các thư viện.

```json
{"query": "How do I join two dataframes?", "positive": "pandas.merge documentation text..."}
```

Chạy:

```bash
PYTHONPATH=src python3 src/fine_tuning/train_retriver.py
```

Model output mặc định: `models/checkpoints/bge-small-code-assistant`.
