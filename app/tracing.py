"""Unified LangSmith tracing for the whole voice agent.

Ported from langchain-ai/voice-demo's sdk_tracing pattern: instead of two
disconnected traces (OTel for ADK + native LangChain for the research agent), we
build ONE tree with the LangSmith RunTree SDK.

    voice_session                     <- root, one per connection
      ├─ user_speech / agent_speech   <- transcript utterances
      └─ deep_research (tool)         <- the long-running tool call
           └─ deep agent (LangGraph)  <- nests automatically (see main.py)

The deep agent nests because run_research runs inside `tracing_context(parent=
tool_run)`, so LangChain's own LangSmith tracer attaches its run as a child of
the deep_research span.
"""

import io
import logging
import time
import wave

import numpy as np
from langsmith import RunTree

from .config import LANGSMITH_API_KEY, LANGSMITH_PROJECT, TRACING_ENABLED

logger = logging.getLogger("voice-agent.tracing")


class SessionTrace:
    """A single conversation's root span plus helpers to add children."""

    def __init__(self, run: RunTree):
        self.run = run

    def event(self, name: str, run_type: str, inputs: dict, outputs: dict) -> None:
        """Record a self-contained leaf span (e.g. a transcript utterance)."""
        child = self.run.create_child(name=name, run_type=run_type, inputs=inputs)
        child.post()
        child.end(outputs=outputs)
        child.patch()

    def begin_tool(self, name: str, inputs: dict) -> RunTree:
        """Open a tool span and return it so downstream work can nest under it."""
        child = self.run.create_child(name=name, run_type="tool", inputs=inputs)
        child.post()
        return child

    def finalize(self, outputs: dict | None = None, attachments: dict | None = None) -> None:
        # Attaching the conversation audio to the root run, with ls_modality=audio
        # already set in the metadata, makes LangSmith render an audio player.
        if attachments:
            self.run.attachments = attachments
        self.run.end(outputs=outputs or {})
        self.run.patch()


class StereoRecorder:
    """Records both sides of the conversation into one time-aligned stereo WAV.

    User audio goes to the left channel, agent audio to the right. Chunks are
    stamped with their arrival time so each sits at the right point on the
    timeline (with silence in the gaps) — otherwise both sides would start at
    t=0 and overlap. The user's 16 kHz audio is resampled to the agent's 24 kHz.
    """

    def __init__(self):
        self._t0 = time.monotonic()
        self._user: list[tuple[float, bytes]] = []   # (offset_seconds, pcm)
        self._agent: list[tuple[float, bytes]] = []

    def add_user(self, pcm: bytes) -> None:
        self._user.append((time.monotonic() - self._t0, pcm))

    def add_agent(self, pcm: bytes) -> None:
        self._agent.append((time.monotonic() - self._t0, pcm))

    def build_wav(
        self, user_rate: int, agent_rate: int, out_rate: int = 24000
    ) -> bytes | None:
        if not self._user and not self._agent:
            return None

        # Lay each side out into one contiguous track (at its native rate), then
        # resample the whole track once to the shared output rate.
        left = _resample(_layout(self._user, user_rate), user_rate, out_rate)
        right = _resample(_layout(self._agent, agent_rate), agent_rate, out_rate)

        total = max(left.size, right.size)
        if total == 0:
            return None
        stereo = np.zeros(total * 2, dtype=np.int16)
        stereo[0 : left.size * 2 : 2] = left   # user -> left channel
        stereo[1 : right.size * 2 : 2] = right  # agent -> right channel

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(2)
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(out_rate)
            wav.writeframes(stereo.tobytes())
        return buffer.getvalue()


def _layout(chunks: list[tuple[float, bytes]], rate: int) -> np.ndarray:
    """Place chunks on a timeline, keeping bursts contiguous (voice-demo style).

    Each chunk starts at the LATER of its arrival time and where the previous
    chunk ended. This avoids the two failure modes of naive arrival-time
    placement: bursty audio (the agent streams faster than real time)
    overlapping and compressing into fast/garbled playback, and jittery chunks
    overwriting each other. Genuine pauses still leave silence.
    """
    placements = []
    cursor = 0  # next free sample index
    for off, pcm in chunks:
        samples = np.frombuffer(pcm, dtype=np.int16)
        if samples.size == 0:
            continue
        start = max(int(off * rate), cursor)
        placements.append((start, samples))
        cursor = start + samples.size

    out = np.zeros(cursor, dtype=np.int16)
    for start, samples in placements:
        out[start : start + samples.size] = samples
    return out


def _resample(samples: np.ndarray, in_rate: int, out_rate: int) -> np.ndarray:
    """Resample a whole 16-bit mono track once (no per-chunk boundary clicks)."""
    if samples.size == 0 or in_rate == out_rate:
        return samples
    n_out = round(samples.size * out_rate / in_rate)
    x_old = np.arange(samples.size)
    x_new = np.linspace(0, samples.size - 1, n_out)
    return np.interp(x_new, x_old, samples).astype(np.int16)


def start_session(thread_id: str, metadata: dict) -> SessionTrace | None:
    """Create the conversation root span, or None when tracing is off."""
    if not (TRACING_ENABLED and LANGSMITH_API_KEY):
        return None

    run = RunTree(
        name="voice_session",
        run_type="chain",
        inputs={},
        project_name=LANGSMITH_PROJECT,
        tags=["voice-agent", "adk"],
        extra={"metadata": {"ls_modality": "audio", "thread_id": thread_id, **metadata}},
    )
    run.post()
    logger.info("Started LangSmith voice_session trace for %s", thread_id)
    return SessionTrace(run)
