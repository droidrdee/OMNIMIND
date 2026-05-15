"""LangChain ReAct agent with Tavily web search tool — for live/current information queries."""

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain_core.prompts import PromptTemplate
from typing import List
from loguru import logger
import os


_FALLBACK_REACT_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""


class WebSearchAgent:
    """ReAct agent that performs live web search using the Tavily search API.

    This agent is designed for queries requiring current/real-time information
    that may not be present in the local knowledge base. It uses an OpenAI chat
    model as the reasoning engine and Tavily as the search tool, wired together
    via LangChain's standard ReAct agent scaffolding.
    """

    def __init__(
        self,
        tavily_api_key: str,
        openai_api_key: str,
        openai_model: str = "gpt-4o",
        max_results: int = 5,
        max_iterations: int = 3,
    ) -> None:
        """Initialize the web search agent.

        Args:
            tavily_api_key: API key for the Tavily search service. Exported as
                the ``TAVILY_API_KEY`` env var so the underlying tool can read it.
            openai_api_key: API key for OpenAI chat completions.
            openai_model: OpenAI chat model identifier (default: ``gpt-4o``).
            max_results: Maximum number of search results returned per Tavily call.
            max_iterations: Maximum number of ReAct reasoning steps before the
                agent is forced to stop.
        """
        os.environ["TAVILY_API_KEY"] = tavily_api_key

        self.search_tool = TavilySearchResults(max_results=max_results)
        self.tools = [self.search_tool]

        self.llm = ChatOpenAI(
            model=openai_model,
            api_key=openai_api_key,
            temperature=0.0,
        )

        try:
            prompt = hub.pull("hwchase17/react")
            logger.debug("WebSearchAgent: pulled ReAct prompt from LangChain hub.")
        except Exception as e:
            logger.warning(
                "WebSearchAgent: hub.pull failed ({}). Using inline fallback ReAct prompt.",
                e,
            )
            prompt = PromptTemplate.from_template(_FALLBACK_REACT_PROMPT)

        agent = create_react_agent(llm=self.llm, tools=self.tools, prompt=prompt)

        self.executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=False,
            max_iterations=max_iterations,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
        )

        logger.info(
            "WebSearchAgent initialized (model={}, max_results={}, max_iterations={}).",
            openai_model,
            max_results,
            max_iterations,
        )

    def run(self, query: str) -> dict:
        """Execute the ReAct agent against ``query`` and return a structured result.

        Args:
            query: The user's natural-language question.

        Returns:
            A dict with keys:
                - ``answer`` (str): The final answer text from the agent.
                - ``sources`` (list[str]): URLs gathered from intermediate Tavily
                  search observations.
                - ``route`` (str): Always ``"web_search"`` for this agent.
        """
        try:
            result = self.executor.invoke({"input": query})
            answer = result.get("output", "")

            sources: List[str] = []
            seen: set[str] = set()
            for step in result.get("intermediate_steps", []) or []:
                try:
                    _action, observation = step
                except (TypeError, ValueError):
                    continue
                if isinstance(observation, list):
                    for item in observation:
                        if isinstance(item, dict):
                            url = item.get("url")
                            if isinstance(url, str) and url and url not in seen:
                                seen.add(url)
                                sources.append(url)

            return {"answer": answer, "sources": sources, "route": "web_search"}
        except Exception as e:
            logger.error("WebSearchAgent.run failed for query={!r}: {}", query, e)
            return {
                "answer": f"Web search failed: {e}",
                "sources": [],
                "route": "web_search",
            }
