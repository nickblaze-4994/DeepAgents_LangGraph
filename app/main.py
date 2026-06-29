"""FastAPI server that bridges a browser <-> the Gemini Live API via ADK.

Each browser connection opens a WebSocket. We run an ADK live session and pump
audio in both directions:

    browser mic  --16kHz PCM-->  LiveRequestQueue  -->  Gemini Live
    browser <--24kHz PCM--  run_live() events  <--  Gemini Live

The voice agent (app.voice_agent.root_agent) can call the deep_research tool
mid-conversation; ADK handles the tool call transparently inside run_live().
"""

import asyncio
import base64
import json
import logging
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from google.adk.agents.live_request_queue import LiveRequestQueue
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.runners import InMemoryRunner
from google.genai import types
from langsmith.run_helpers import tracing_context

from .config import (
    INPUT_SAMPLE_RATE,
    LANGSMITH_PROJECT,
    LIVE_MODEL,
    OUTPUT_SAMPLE_RATE,
    RESEARCH_SCHEDULING,
    TRACING_ENABLED,
)
from .research_agent import run_research
from .tracing import StereoRecorder, start_session
from .voice_agent import root_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice-agent")

# Map the configured scheduling string to the genai enum (default INTERRUPT, so
# the model volunteers research results as soon as they land).
RESEARCH_SCHEDULING_MODE = {
    "INTERRUPT": types.FunctionResponseScheduling.INTERRUPT,
    "WHEN_IDLE": types.FunctionResponseScheduling.WHEN_IDLE,
    "SILENT": types.FunctionResponseScheduling.SILENT,
}.get(RESEARCH_SCHEDULING, types.FunctionResponseScheduling.INTERRUPT)

if TRACING_ENABLED:
    logger.info("LangSmith tracing ON -> project '%s' (unified voice + research trace)", LANGSMITH_PROJECT)
else:
    logger.info("LangSmith tracing OFF (set LANGSMITH_TRACING=true + LANGSMITH_API_KEY to enable)")

APP_NAME = "adk-realtime-deepagents-voice"
STATIC_DIR = Path(__file__).parent.parent / "static"

app = FastAPI(title=APP_NAME)

# One runner for the whole process; it manages sessions internally.
runner = InMemoryRunner(app_name=APP_NAME, agent=root_agent)


async def start_agent_session(user_id: str):
    """Open a live ADK session and return (live_events, live_request_queue)."""
    session = await runner.session_service.create_session(
        app_name=APP_NAME,
        user_id=user_id,
    )

    run_config = RunConfig(
        streaming_mode=StreamingMode.BIDI,
        response_modalities=["AUDIO"],
        # Transcribe both sides — used to label utterances in the LangSmith trace.
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
    )

    live_request_queue = LiveRequestQueue()
    live_events = runner.run_live(
        user_id=user_id,
        session_id=session.id,
        live_request_queue=live_request_queue,
        run_config=run_config,
    )
    return live_events, live_request_queue


async def _deliver_research(
    live_request_queue: LiveRequestQueue,
    call_id: str,
    name: str,
    topic: str,
    tool_run=None,
):
    """Run the slow research, then hand the report back to the live model.

    The model called deep_research (a long-running tool) and already moved on.
    The research runs inside the deep_research tool span (tool_run) so the
    LangChain/LangGraph deep agent nests under it in the LangSmith trace. When
    the report is ready we send it as a FunctionResponse tagged with the original
    call id; the scheduling mode (RESEARCH_SCHEDULING, default INTERRUPT) tells
    the Live API when the model should speak the results.
    """
    if tool_run is not None:
        # parent=tool_run makes the deep agent's own LangSmith run a child span.
        with tracing_context(parent=tool_run):
            report = await run_research(topic)
        tool_run.end(outputs={"report": report})
        tool_run.patch()
    else:
        report = await run_research(topic)

    try:
        live_request_queue.send_content(
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        function_response=types.FunctionResponse(
                            id=call_id,
                            name=name,
                            response={"report": report},
                            scheduling=RESEARCH_SCHEDULING_MODE,
                        )
                    )
                ],
            )
        )
    except Exception:
        # The connection may have closed while research was running; that's fine.
        logger.info("Could not deliver research result for call %s", call_id)


async def agent_to_client(
    websocket: WebSocket,
    live_events,
    live_request_queue: LiveRequestQueue,
    background_tasks: set,
    session_trace=None,
    recorder: StereoRecorder | None = None,
):
    """Stream model audio to the browser; record the conversation to LangSmith."""
    async for event in live_events:
        # Barge-in interruption: tell the client to flush queued playback audio.
        if getattr(event, "interrupted", False):
            await websocket.send_text(json.dumps({"type": "interrupted"}))
            continue
        if getattr(event, "turn_complete", False):
            continue

        # Long-running (non-blocking) tool calls: the model gets the tool's
        # immediate ack and keeps talking; we run the real work in the background
        # and deliver the result later via _deliver_research.
        long_running_ids = set(getattr(event, "long_running_tool_ids", None) or [])
        for call in event.get_function_calls():
            if call.id in long_running_ids and call.name == "deep_research":
                topic = (call.args or {}).get("topic", "")
                # Open the deep_research tool span so the deep agent nests in it.
                tool_run = (
                    session_trace.begin_tool("deep_research", {"topic": topic})
                    if session_trace
                    else None
                )
                task = asyncio.create_task(
                    _deliver_research(
                        live_request_queue, call.id, call.name, topic, tool_run
                    )
                )
                background_tasks.add(task)
                task.add_done_callback(background_tasks.discard)

        # Record completed utterances as spans. ADK emits transcripts twice:
        # incremental partial=True deltas, then a consolidated partial=False event
        # with the full text. We use the consolidated one for a clean single span.
        if not getattr(event, "partial", False) and session_trace:
            input_tx = getattr(event, "input_transcription", None)
            if input_tx and input_tx.text:
                session_trace.event(
                    "user_speech", "chain", {}, {"transcript": input_tx.text}
                )
            output_tx = getattr(event, "output_transcription", None)
            if output_tx and output_tx.text:
                session_trace.event(
                    "agent_speech", "chain", {}, {"transcript": output_tx.text}
                )

        content = getattr(event, "content", None)
        if not content or not content.parts:
            continue

        # Stream the model's 24 kHz PCM audio to the browser (base64 for JSON).
        for part in content.parts:
            inline = getattr(part, "inline_data", None)
            if inline and inline.data and inline.mime_type.startswith("audio/"):
                if recorder is not None:
                    recorder.add_agent(inline.data)  # right channel of the recording
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "audio",
                            "data": base64.b64encode(inline.data).decode(),
                        }
                    )
                )


async def client_to_agent(
    websocket: WebSocket,
    live_request_queue: LiveRequestQueue,
    recorder: StereoRecorder | None = None,
):
    """Forward browser mic audio (and any typed text) into the live session."""
    while True:
        message = json.loads(await websocket.receive_text())
        msg_type = message.get("type")

        if msg_type == "audio":
            audio_bytes = base64.b64decode(message["data"])
            if recorder is not None:
                recorder.add_user(audio_bytes)  # left channel of the recording
            live_request_queue.send_realtime(
                types.Blob(
                    mime_type=f"audio/pcm;rate={INPUT_SAMPLE_RATE}",
                    data=audio_bytes,
                )
            )
        elif msg_type == "text":
            live_request_queue.send_content(
                types.Content(role="user", parts=[types.Part(text=message["data"])])
            )


@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await websocket.accept()
    logger.info("Client %s connected", user_id)

    live_events, live_request_queue = await start_agent_session(user_id)
    # Root LangSmith span for this whole conversation (None if tracing is off).
    session_trace = start_session(thread_id=user_id, metadata={"model": LIVE_MODEL})
    # Records both sides into one time-aligned stereo recording for the trace.
    recorder = StereoRecorder() if session_trace else None
    # Tracks in-flight background research tasks for this connection.
    background_tasks: set = set()
    try:
        # Run both directions concurrently until either side ends.
        await asyncio.gather(
            agent_to_client(
                websocket, live_events, live_request_queue, background_tasks,
                session_trace, recorder,
            ),
            client_to_agent(websocket, live_request_queue, recorder),
        )
    except WebSocketDisconnect:
        logger.info("Client %s disconnected", user_id)
    finally:
        for task in background_tasks:
            task.cancel()
        live_request_queue.close()
        if session_trace:
            # Attach the stereo conversation (user=left, agent=right) to the root.
            wav = recorder.build_wav(INPUT_SAMPLE_RATE, OUTPUT_SAMPLE_RATE)
            attachments = {"conversation": ("audio/wav", wav)} if wav else None
            session_trace.finalize(attachments=attachments)


class NoCacheStaticFiles(StaticFiles):
    """StaticFiles that tells the browser never to cache these assets.

    `no-store` (vs `no-cache`) guarantees the browser refetches index.html and
    app.js every load, so a stale/mismatched pair (new HTML + old JS) can't
    wedge the page while iterating on the demo.
    """

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-store"
        return response


# Serve the browser client. Mounted last so it doesn't shadow /ws.
app.mount("/", NoCacheStaticFiles(directory=STATIC_DIR, html=True), name="static")
