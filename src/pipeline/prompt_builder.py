from typing import List
import unicodedata

from core.schemas import Chunk


class PromptBuilder:
    """Builds compact RAG context for code-assistance answers."""

    MAX_CHUNK_CHARS = 1400

    @staticmethod
    def build_context(chunks: List[Chunk]) -> str:
        return "\n\n---\n\n".join(
            [
                "Source {index}\nFunction: {name}\nLibrary: {library}\nParameters: {params}\nDocumentation:\n{doc}".format(
                    index=index,
                    name=chunk.metadata.get("func_name", "Unknown"),
                    library=chunk.metadata.get("library_name", "Unknown"),
                    params=chunk.metadata.get("parameters", {}),
                    doc=PromptBuilder._trim_doc(chunk.content),
                )
                for index, chunk in enumerate(chunks, start=1)
            ]
        )

    @staticmethod
    def build_user_message(query: str, chunks: List[Chunk]) -> str:
        language = PromptBuilder._detect_language(query)
        return (
            "Retrieved documentation:\n"
            f"{PromptBuilder.build_context(chunks)}\n\n"
            f"User question: {query}\n\n"
            f"Detected user language: {language}\n\n"
            "Answer requirements:\n"
            "- Reply in the detected user language.\n"
            "- Start with the most likely function or API to use.\n"
            "- Include one short, runnable code example when useful.\n"
            "- Explain only the most important parameters for this question.\n"
            "- If the retrieved documentation is not enough, say what is missing instead of guessing."
        )

    @staticmethod
    def _trim_doc(doc: str) -> str:
        doc = (doc or "").strip()
        if len(doc) <= PromptBuilder.MAX_CHUNK_CHARS:
            return doc
        return doc[: PromptBuilder.MAX_CHUNK_CHARS].rsplit("\n", 1)[0].strip()

    @staticmethod
    def _detect_language(text: str) -> str:
        normalized = unicodedata.normalize("NFKD", text or "")
        normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch)).lower()
        vietnamese_terms = [
            "cach",
            "doc",
            "bang",
            "nhu the nao",
            "lam sao",
            "ket hop",
            "chia",
            "tach",
            "du lieu",
            "thu vien",
        ]
        if any(term in normalized for term in vietnamese_terms):
            return "Vietnamese"
        return "English"
