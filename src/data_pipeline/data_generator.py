import json
import random
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
from llama_cpp import Llama

# === CẤU HÌNH ĐƯỜNG DẪN ===
CHUNKS_PATH = "data/chunks/functions.jsonl"
OUTPUT_DIR = Path("data/")
MODEL_PATH = "models/final_gguf/qwen2.5-1.5b-instruct-q4_k_m.gguf"

class RAGDatasetGenerator:
    def __init__(self, model_path: str, chunks_path: str):
        self.chunks_path = chunks_path
        self.chunks = self._load_jsonl(chunks_path)
        
        self.library_chunks = {}
        for c in self.chunks:
            lib = c.get("library_name", "unknown")
            if lib not in self.library_chunks:
                self.library_chunks[lib] = []
            self.library_chunks[lib].append(c)
            
        print(f"Loaded {len(self.chunks)} chunks. Loading Qwen model...")
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1, 
            n_ctx=1024, # Tăng context để chứa cả câu trả lời mẫu
            verbose=False,
            chat_format="qwen"
        )

    def _load_jsonl(self, path: str) -> List[Dict]:
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    def _format_document(self, chunk: Dict) -> str:
        full_name = chunk.get("full_name", "")
        signature = chunk.get("signature") or ""
        docstring = chunk.get("docstring", "")
        return f"{full_name}{signature}\n{docstring}"

    def generate_synthetic_data(self, chunk: Dict) -> Dict[str, str]:
        """Sử dụng LLM để sinh ra MỘT câu hỏi và MỘT câu trả lời mẫu (Ground Truth)"""
        doc_text = self._format_document(chunk)[:1000] # Lấy 1000 ký tự để prompt gọn
        
        prompt = (
            f"You are a Python expert. Read the documentation below:\n"
            f"---DOC---\n{doc_text}\n---END DOC---\n\n"
            f"Task 1: Write a user query asking how to solve a problem that this function solves. "
            f"Do NOT use the function name in the query.\n"
            f"Task 2: Write a clear, complete, and helpful answer to that query using ONLY the provided documentation.\n\n"
            f"Format your response exactly like this:\n"
            f"Query: <your query>\n"
            f"Answer: <your answer>"
        )

        try:
            response = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": "You are a helpful coding assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.3
            )
            output = response['choices'][0]['message']['content']
            
            # Tách Query và Answer từ output của LLM
            query = ""
            answer = ""
            for line in output.split('\n'):
                if line.lower().startswith('query:'):
                    query = line[6:].strip()
                elif line.lower().startswith('answer:'):
                    answer = line[7:].strip()
                    
            return {"query": query, "ground_truth": answer}
        except Exception as e:
            print(f"Error generation: {e}")
            return {"query": "", "ground_truth": ""}

    def get_hard_negatives(self, target_chunk: Dict, num_neg: int = 3) -> List[str]:
        lib = target_chunk.get("library_name", "unknown")
        pool = self.library_chunks.get(lib, self.chunks)
        valid_pool = [c for c in pool if c["chunk_id"] != target_chunk["chunk_id"]]
        
        if len(valid_pool) < num_neg:
            valid_pool = [c for c in self.chunks if c["chunk_id"] != target_chunk["chunk_id"]]
            
        sampled_negatives = random.sample(valid_pool, min(num_neg, len(valid_pool)))
        return [self._format_document(neg) for neg in sampled_negatives]

    def build_datasets(self, test_ratio: float = 0.1, benchmark_size: int = 50):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        random.shuffle(self.chunks)
        
        # CHIA TẬP DỮ LIỆU
        benchmark_chunks = self.chunks[:benchmark_size]
        retriever_chunks = self.chunks[benchmark_size:]
        
        # 1. TẠO TẬP RETRIEVER TRAIN/TEST (Cho BGE)
        print("1. Generating BGE format dataset...")
        bge_dataset = []
        for chunk in tqdm(retriever_chunks[:300]): # Giới hạn 300 để test nhanh
            synth_data = self.generate_synthetic_data(chunk)
            if not synth_data["query"]: continue
                
            bge_dataset.append({
                "query": synth_data["query"],
                "pos": [self._format_document(chunk)],
                "neg": self.get_hard_negatives(chunk, num_neg=3),
                "pos_scores": [],
                "neg_scores": [],
                "prompt": "Represent this sentence for searching relevant passages: ",
                "type": "normal"
            })
            
        split_idx = int(len(bge_dataset) * (1 - test_ratio))
        self._write_jsonl(bge_dataset[:split_idx], OUTPUT_DIR / "training" / "retriever_train.jsonl")
        self._write_jsonl(bge_dataset[split_idx:], OUTPUT_DIR / "training" / "retriever_test.jsonl")
        
        # 2. TẠO TẬP BENCHMARK END-TO-END
        print("2. Generating End-to-End Benchmark dataset...")
        benchmark_dataset = []
        for chunk in tqdm(benchmark_chunks):
            synth_data = self.generate_synthetic_data(chunk)
            if synth_data["query"] and synth_data["ground_truth"]:
                benchmark_dataset.append({
                    "query": synth_data["query"],
                    "reference_context": self._format_document(chunk),
                    "ground_truth": synth_data["ground_truth"],
                    "metadata": {
                        "library": chunk["library_name"],
                        "function": chunk["full_name"]
                    }
                })
        self._write_jsonl(benchmark_dataset, OUTPUT_DIR / "benchmark" / "benchmark.jsonl")
        print("Done!")

    def _write_jsonl(self, data: List[Dict], path: Path):
        with open(path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    generator = RAGDatasetGenerator(model_path=MODEL_PATH, chunks_path=CHUNKS_PATH)
    generator.build_datasets(test_ratio=0.1, benchmark_size=50)