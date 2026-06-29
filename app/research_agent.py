"""The deep research tool.

This wraps a LangChain *deep agent* (`create_deep_agent`) — an agent harness
with built-in planning, a virtual file system, and sub-agents — and exposes it
to the voice agent as a single ADK tool: `deep_research`.

When the voice agent decides it needs to research something, it calls this tool,
the deep agent goes off and runs multiple Tavily web searches while planning its
work, and a written report comes back. The voice model then narrates the answer.
"""

import logging

from deepagents import create_deep_agent
from google.adk.tools import LongRunningFunctionTool
from google.genai import types
from tavily import TavilyClient

from .config import RESEARCH_MODEL, require_env

logger = logging.getLogger("voice-agent.research")

# The instructions that shape how the deep agent researches.
RESEARCH_INSTRUCTIONS = """You are an expert research assistant.

Your job is to deeply research the user's question using the `internet_search`
tool, then write a clear, well-organized answer.

Workflow:
1. Break the question into sub-questions and plan your research with the todo list.
2. Run multiple targeted searches. Prefer recent, authoritative sources.
3. Cross-check important claims across more than one source.
4. Synthesize a concise report: lead with the direct answer, then supporting
   detail. Keep it tight — this will be read aloud by a voice assistant, so avoid
   markdown tables, long URLs, and bullet soup. Use short paragraphs.
"""


def internet_search(
    query: str,
    max_results: int = 5,
    topic: str = "general",
) -> dict:
    """Run a web search via Tavily and return the results.

    Args:
        query: The search query.
        max_results: Maximum number of results to return (default 5).
        topic: One of "general", "news", or "finance".
    """
    client = TavilyClient(api_key=require_env("TAVILY_API_KEY"))
    return client.search(
        query,
        max_results=max_results,
        topic=topic,
        include_raw_content=False,
    )


def deep_research(topic: str) -> dict:
    """Kick off in-depth research on a topic. Returns immediately.

    This is a NON-BLOCKING / long-running tool. It starts the research and
    returns right away so the conversation can continue — the written report is
    delivered to you a little later, as a follow-up result. Use it whenever the
    user asks something needing up-to-date facts, multiple sources, or careful
    investigation (news, comparisons, "what's the latest on...", deep-dives).

    Args:
        topic: The thing to research, phrased as a clear question or request.

    Returns:
        A status dict acknowledging that research has started.
    """
    return {
        "status": "researching",
        "topic": topic,
        "note": (
            "Research has started and will take up to a minute. Keep chatting "
            "with the user meanwhile; present the findings naturally when they "
            "arrive as a follow-up result."
        ),
    }


class _NonBlockingLongRunningTool(LongRunningFunctionTool):
    """LongRunningFunctionTool that also declares the tool NON_BLOCKING to Gemini.

    LongRunningFunctionTool handles the ADK-side orchestration (return an ack now,
    inject the real FunctionResponse later) but never sets the declaration's
    `behavior`. Without behavior=NON_BLOCKING, Gemini Live treats the tool as
    blocking and *ignores* the scheduling on our late response (defaulting to
    WHEN_IDLE — i.e. it won't speak the result until the user talks again).
    Marking it NON_BLOCKING lets the model keep conversing and honor
    scheduling=INTERRUPT, so it announces the findings as soon as they land.
    """

    def _get_declaration(self):
        declaration = super()._get_declaration()
        if declaration:
            declaration.behavior = types.Behavior.NON_BLOCKING
        return declaration


# Long-running (ADK injects the real result later) AND non-blocking (the model
# keeps talking while it runs). See app/main.py for the background delivery.
deep_research_tool = _NonBlockingLongRunningTool(func=deep_research)

# Built lazily on first use so the app can import without keys present.
_research_agent = None


def _get_research_agent():
    global _research_agent
    if _research_agent is None:
        # Fail early with a clear message if the key isn't set.
        require_env("GOOGLE_API_KEY")
        _research_agent = create_deep_agent(
            model=RESEARCH_MODEL,
            tools=[internet_search],
            system_prompt=RESEARCH_INSTRUCTIONS,
        )
    return _research_agent



async def run_research(topic: str) -> str:
    """Do the actual deep research and return the report as plain text.

    Called in the background by the server once the long-running tool fires;
    never raises — failures come back as a readable message the agent can relay.
    """
    try:
        result = await _get_research_agent().ainvoke(
            {"messages": [{"role": "user", "content": topic}]}
        )
    except Exception as exc:  # surface the failure instead of failing silently
        logger.exception("run_research failed")
        return f"The research didn't go through this time ({exc})."

    return _as_text(result["messages"][-1].content)


def _as_text(content) -> str:
    """Coerce a LangChain message's content to a plain string.

    Gemini (via LangChain) returns content as a list of structured blocks like
    [{"type": "text", "text": "...", "extras": {"signature": "..."}}], which
    carries a reasoning signature meant for *that* model. We must hand the Live
    model a clean string, not that list — otherwise the function-response is
    malformed and the Live model reports a failure.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)
