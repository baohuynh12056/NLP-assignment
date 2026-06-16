# src/fine_tuning/evaluator.py
import re
import time
import json
import pandas as pd
from typing import List, Dict, Tuple
from tqdm import tqdm
from llama_cpp import Llama


class LLMJudge:
    def __init__(self, model_path: str, n_ctx: int = 2048):
        print(f"[LLM-Judge] Loading Qwen-Judge from: {model_path}")
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=n_ctx,
            verbose=False,
            chat_format="qwen"
        )

    def evaluate(self, query: str, generated_answer: str, ground_truth: str) -> int:
        """Use Qwen as an LLM Judge to score answers from 1 to 5."""
        
        prompt = (
            f"You are an AI evaluator (LLM-as-a-Judge).\n"
            f"Task: Evaluate the AI-generated answer based on the user question and the reference answer.\n"
            f"Scoring criteria:\n"
            f"- 1: Incorrect or irrelevant\n"
            f"- 3: Partially correct\n"
            f"- 5: Fully correct and comprehensive (including useful code examples when applicable)\n\n"
            f"[Question]: {query}\n"
            f"[Reference Answer]: {ground_truth}\n"
            f"[AI Answer]: {generated_answer}\n\n"
            f"Output only a single number from 1 to 5 enclosed in <score> tags.\n"
            f"Example: <score>5</score>"
        )

        try:
            response = self.llm.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an impartial and strict AI evaluator."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=20,
                temperature=0.1
            )

            output = response["choices"][0]["message"]["content"]

            # Extract score using regex
            match = re.search(r"<score>(\d+)</score>", output)
            if match:
                score = int(match.group(1))
                return min(max(score, 1), 5)

            # Fallback: extract first number found
            numbers = re.findall(r"\d+", output)
            if numbers:
                return min(max(int(numbers[0]), 1), 5)

            return 3

        except Exception as e:
            print(f"[LLM-Judge] Evaluation error: {e}")
            return 0


class E2EEvaluator:
    def __init__(self, judge_model_path: str):
        self.judge = LLMJudge(judge_model_path)

    def load_benchmark(self, path: str) -> List[Dict]:
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    def run_single_pipeline(self, orchestrator, query: str) -> Tuple[str, float]:
        """
        Execute the real End-to-End pipeline through the orchestrator.
        The orchestrator.run() method triggers:
        Parser -> Retriever (DB) -> Reranker -> Generator
        """
        start_time = time.time()

        answer = orchestrator.run(query)

        latency = time.time() - start_time
        return answer, latency

    def evaluate_pipeline(
        self,
        base_orchestrator,
        benchmark_path: str
    ) -> pd.DataFrame:

        benchmark_data = self.load_benchmark(benchmark_path)
        print(f"Loaded {len(benchmark_data)} benchmark samples.")

        results = []

        for row in tqdm(benchmark_data, desc="Running E2E Evaluation"):
            query = row["query"]
            ground_truth = row.get(
                "ground_truth",
                row.get("reference_context", "")
            )

            # Run Base Pipeline (without LoRA)
            base_ans, base_latency = self.run_single_pipeline(
                base_orchestrator,
                query
            )

            base_score = self.judge.evaluate(
                query,
                base_ans,
                ground_truth
            )

            results.append({
                "query": query,
                "base_score": base_score,
                "base_latency": base_latency,
            })

        return pd.DataFrame(results)