"""GPT-4 reasoning agent for complex multi-hop queries, optionally with retrieved context."""

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from typing import List, Optional
from loguru import logger


class ReasoningAgent:
    """Chat-model reasoning agent for complex, multi-hop research questions.

    The agent prompts an OpenAI chat model to think step-by-step. When the
    caller provides retrieved context chunks (e.g. from a vector store), they
    are formatted into the user message so the model can ground its answer
    and cite sources.
    """

    def __init__(
        self,
        openai_api_key: str,
        openai_model: str = "gpt-4o",
        temperature: float = 0.2,
    ) -> None:
        """Initialize the reasoning agent.

        Args:
            openai_api_key: API key for OpenAI chat completions.
            openai_model: OpenAI chat model identifier (default: ``gpt-4o``).
            temperature: Sampling temperature for the chat model. Lower values
                produce more deterministic, focused reasoning.
        """
        self.llm = ChatOpenAI(
            model=openai_model,
            api_key=openai_api_key,
            temperature=temperature,
            max_tokens=1500,
        )
        self.system_prompt = (
            "You are OMNIMIND, an expert AI research assistant. Reason step-by-step. "
            "Break complex questions into parts. If unsure or if context is insufficient, "
            "say so clearly. Cite sources when provided."
        )
        logger.info(
            "ReasoningAgent initialized (model={}, temperature={}).",
            openai_model,
            temperature,
        )

    def run(
        self,
        query: str,
        context_chunks: Optional[List[dict]] = None,
    ) -> dict:
        """Answer ``query`` with step-by-step reasoning, optionally over context.

        Args:
            query: The user's natural-language question.
            context_chunks: Optional list of retrieved chunks. Each chunk should
                be a dict with at least a ``text`` field and a ``metadata`` dict
                that may contain a ``source`` key used for citation.

        Returns:
            A dict with keys:
                - ``answer`` (str): The model's reasoned answer.
                - ``sources`` (list[str]): Unique source identifiers extracted
                  from ``context_chunks`` metadata, in first-seen order.
                - ``route`` (str): Always ``"reasoning"`` for this agent.
        """
        messages = [SystemMessage(content=self.system_prompt)]
        sources: List[str] = []

        if context_chunks:
            context_text = "\n\n---\n\n".join(
                f"[{c['metadata'].get('source', 'unknown')}]\n{c['text']}"
                for c in context_chunks
            )

            seen: set[str] = set()
            for c in context_chunks:
                source = c.get("metadata", {}).get("source", "unknown")
                if source not in seen:
                    seen.add(source)
                    sources.append(source)

            user_content = (
                f"Context:\n{context_text}\n\n"
                f"Question: {query}\n\n"
                "Provide a step-by-step reasoned answer."
            )
        else:
            user_content = (
                f"Question: {query}\n\nProvide a step-by-step reasoned answer."
            )

        messages.append(HumanMessage(content=user_content))

        try:
            response = self.llm.invoke(messages)
            answer = response.content
        except Exception as e:
            logger.error("ReasoningAgent.run failed for query={!r}: {}", query, e)
            answer = f"Reasoning failed: {e}"

        return {"answer": answer, "sources": sources, "route": "reasoning"}
