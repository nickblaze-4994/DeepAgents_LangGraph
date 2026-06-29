# 🎙️ Real-time Voice Research Agent — Google ADK × Gemini Live × deepagents

A small, hackable demo of a **real-time voice agent**:

- **[Google ADK](https://adk.dev)** runs a bidirectional **[Gemini Live API](https://ai.google.dev/gemini-api/docs/live-api)** session (`gemini-3.1-flash-live-preview`) — you talk, it talks back, with barge-in.
- When you ask something that needs real research, the voice agent calls a **[LangChain deepagents](https://github.com/langchain-ai/deepagents)** tool that plans, runs multiple **Tavily** web searches, and writes a report.
- The voice model then narrates the answer conversationally.
- Every research run is traced to **[LangSmith](https://smith.langchain.com)** so you can replay each plan step, search, and sub-agent call.

```
 ┌─────────┐  16kHz PCM   ┌──────────────┐  run_live()  ┌──────────────┐
 │ Browser │ ───────────▶ │  FastAPI WS  │ ───────────▶ │  Gemini Live │
 │ mic/spkr│ ◀─────────── │   (ADK)      │ ◀─────────── │  (ADK agent) │
 └─────────┘  24kHz PCM   └──────────────┘   events     └──────┬───────┘
                                                                │ tool call
                                                         ┌──────▼───────┐
                                                         │ deep_research│  ← deepagents
                                                         │  + Tavily    │
                                                         └──────────────┘
```

## Project layout

```
app/
  main.py            FastAPI WebSocket server; bridges browser audio <-> Gemini Live
  voice_agent.py     The ADK root agent (Gemini Live model + deep_research tool)
  research_agent.py  LangChain deep agent (planning + Tavily search), exposed as a tool
  config.py          Env/config and audio constants
static/
  index.html         Minimal UI with live captions
  app.js             Mic capture, WS transport, playback
  pcm-recorder-processor.js   AudioWorklet: mic -> Float32 frames
  pcm-player-processor.js     AudioWorklet: gapless PCM playback + barge-in flush
```

## Setup

You need two keys: a **Google AI Studio** key (Gemini Live + research LLM) and a **Tavily** key (web search).

```bash
# 1. Install deps (uses uv; see https://docs.astral.sh/uv/)
uv sync

# 2. Configure secrets
cp .env.example .env
# then edit .env and paste your GOOGLE_API_KEY and TAVILY_API_KEY
```

> Don't have `uv`? `pip install -e .` works too.

## Tracing with LangSmith

The whole agent traces to LangSmith as **one nested tree**. Just add to `.env`:

```bash
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=...                       # from https://smith.langchain.com
LANGSMITH_PROJECT=adk-realtime-deepagents-voice
```

Each conversation produces a single trace:

```
voice_session                       ← root span, one per connection
  ├─ user_speech / agent_speech     ← transcribed utterances
  └─ deep_research (tool)           ← the long-running tool call
       └─ deep agent (LangGraph)    ← nests automatically
            ├─ write_todos (plan)
            ├─ internet_search (Tavily)
            └─ … the final report
```

> How it works (`app/tracing.py`): we open a LangSmith `RunTree` per connection
> (the voice session), add child spans for utterances and the `deep_research`
> tool call, and run the deep agent inside `tracing_context(parent=tool_run)`.
> Because LangChain's own LangSmith tracer attaches to that parent, the deep
> agent's run lands **inside** the tool span — one unified tree instead of two.
> This follows the pattern in
> [langchain-ai/voice-demo](https://github.com/langchain-ai/voice-demo/tree/main/src/voice_demo/adk).

## Run

```bash
uv run uvicorn app.main:app --reload
```

Open **http://localhost:8000**, click **Start talking**, allow the mic, and speak.

Try: *“What’s the latest on the Gemini Live API?”* — the agent will say a quick
filler line, run deep research in the background, then narrate what it found.

## How it fits together

1. **`app/main.py`** accepts a WebSocket per browser tab and starts an ADK live
   session with `runner.run_live(...)` and `RunConfig(streaming_mode=BIDI,
   response_modalities=["AUDIO"])`. Mic audio comes in as base64 PCM and is fed to
   the model via `live_request_queue.send_realtime(types.Blob(...))`; model audio
   and transcripts stream back out over the same socket.
2. **`app/voice_agent.py`** is the ADK `Agent`. It's instructed to stay brief and
   to delegate anything research-y to its one tool.
3. **`app/research_agent.py`** builds a deep agent with `create_deep_agent(...)`.
   Deep agents come with planning (a todo list), a virtual filesystem, and
   sub-agents out of the box — so a single question can fan out into several
   searches and a synthesized report. We expose it to ADK as the async
   `deep_research(topic)` function tool.

## Customizing for the video

- **Swap the voice model** — set `LIVE_MODEL` in `.env` (e.g.
  `gemini-2.5-flash-native-audio` for the stable native-audio model).
- **Swap the research brain** — set `RESEARCH_MODEL`, a LangChain
  `init_chat_model` string. To use Claude: `RESEARCH_MODEL=anthropic:claude-sonnet-4-6`
  and add `ANTHROPIC_API_KEY` to `.env` (plus `langchain-anthropic` to deps).
- **Pick a voice** — add a `speech_config` to the `RunConfig` in `app/main.py`:
  ```python
  speech_config=types.SpeechConfig(
      voice_config=types.VoiceConfig(
          prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Puck")
      )
  )
  ```

## Notes

- This is a **local demo**: the WebSocket is unauthenticated and sessions live in
  memory. Add auth, TLS, and a session store before putting it anywhere public.
- Mic capture requires a secure context — `localhost` is fine; a remote host needs
  HTTPS.
- All secrets are read from environment variables; nothing is committed (`.env` is
  gitignored).
