"""Central configuration, loaded from the environment (.env)."""

import os

from dotenv import load_dotenv

# Load .env once, at import time, before any client reads the environment.
load_dotenv()

# The Gemini Live model that powers the real-time voice conversation.
# "gemini-3.1-flash-live-preview" is the latest native-audio Live model.
LIVE_MODEL = os.getenv("LIVE_MODEL", "gemini-3.1-flash-live-preview")

# The model the deep research agent reasons with. This is a LangChain
# init_chat_model string ("<provider>:<model>"). We default to Gemini so the
# demo only needs one provider key, but you can swap in e.g.
# "anthropic:claude-sonnet-4-6" if you set ANTHROPIC_API_KEY.
RESEARCH_MODEL = os.getenv("RESEARCH_MODEL", "google_genai:gemini-2.5-flash")

# LangSmith tracing. LangChain/LangGraph (and therefore the deep research agent)
# auto-trace when LANGSMITH_TRACING is truthy and LANGSMITH_API_KEY is set — we
# don't wire anything up in code, we just surface the status for a startup log.
TRACING_ENABLED = os.getenv("LANGSMITH_TRACING", "").lower() in ("1", "true", "yes")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "default")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")

# How the model surfaces a finished background-research result. WHEN_IDLE waits
# for a natural pause before announcing; INTERRUPT cuts in as soon as results
# land; SILENT only folds it into context. (Honored because deep_research is
# declared NON_BLOCKING — see app/research_agent.py.)
RESEARCH_SCHEDULING = os.getenv("RESEARCH_SCHEDULING", "WHEN_IDLE").upper()

# Audio format constants for the Gemini Live API.
INPUT_SAMPLE_RATE = 16000   # mic -> model: 16-bit PCM, 16 kHz, mono
OUTPUT_SAMPLE_RATE = 24000  # model -> speakers: 16-bit PCM, 24 kHz, mono


def require_env(name: str) -> str:
    """Return an env var or raise a friendly error if it's missing."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            "Copy .env.example to .env and fill it in."
        )
    return value
