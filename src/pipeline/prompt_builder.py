from typing import List

from core.schemas import Chunk


class PromptBuilder:
    """Builds compact RAG context for code-assistance answers."""

    @staticmethod
    def build_context(chunks: List[Chunk]) -> str:
        return "\n\n---\n\n".join(
            [
                "Function: {name}\nParameters: {params}\nDocumentation:\n{doc}".format(
                    name=chunk.metadata.get("func_name", "Unknown"),
                    params=chunk.metadata.get("parameters", {}),
                    doc=chunk.content,
                )
                for chunk in chunks
            ]
        )

    @staticmethod
    def build_user_message(query: str, chunks: List[Chunk]) -> str:
        return (
            "Context Documentation:\n"
            f"{PromptBuilder.build_context(chunks)}\n\n"
            f"User Question: {query}"
        )
