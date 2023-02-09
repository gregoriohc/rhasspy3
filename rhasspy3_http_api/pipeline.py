import argparse
import asyncio
import logging
from collections import deque
from typing import Optional, Deque, Union

from quart import request, Quart, Response, jsonify, websocket

from rhasspy3.audio import (
    DEFAULT_IN_CHANNELS,
    DEFAULT_IN_RATE,
    DEFAULT_IN_WIDTH,
    DEFAULT_OUT_CHANNELS,
    DEFAULT_OUT_RATE,
    DEFAULT_OUT_WIDTH,
)
from rhasspy3.asr import DOMAIN as ASR_DOMAIN, Transcript, transcribe_stream
from rhasspy3.audio import AudioChunkConverter, AudioChunk, AudioStop
from rhasspy3.core import Rhasspy
from rhasspy3.config import PipelineConfig
from rhasspy3.event import Event, async_read_event, async_write_event
from rhasspy3.mic import DOMAIN as MIC_DOMAIN
from rhasspy3.handle import Handled, NotHandled, handle
from rhasspy3.program import create_process
from rhasspy3.intent import recognize, Intent
from rhasspy3.wake import detect
from rhasspy3.vad import segment, DOMAIN as VAD_DOMAIN, VoiceStopped, VoiceStarted
from rhasspy3.tts import synthesize_stream
from rhasspy3.pipeline import run as run_pipeline, StopAfterDomain

_LOGGER = logging.getLogger(__name__)


def add_pipeline(
    app: Quart, rhasspy: Rhasspy, pipeline: PipelineConfig, args: argparse.Namespace
) -> None:
    @app.route("/api/listen-for-command", methods=["GET", "POST"])
    async def api_listen_for_command() -> Response:
        mic_program = request.args.get("mic_program", pipeline.mic)
        wake_program = request.args.get("wake_program", pipeline.wake)
        vad_program = request.args.get("vad_program", pipeline.vad)
        asr_program = request.args.get("asr_program", pipeline.asr)
        intent_program = request.args.get("intent_program", pipeline.intent)
        handle_program = request.args.get("handle_program", pipeline.handle)
        tts_program = request.args.get("tts_program", pipeline.tts)
        snd_program = request.args.get("snd_program", pipeline.snd)
        #
        stop_after = request.args.get("stop_after")
        #
        samples_per_chunk = int(
            request.args.get("samples_per_chunk", args.samples_per_chunk)
        )
        asr_chunks_to_buffer = int(
            request.args.get("asr_chunks_to_buffer", args.asr_chunks_to_buffer)
        )

        _LOGGER.debug(
            "listen-for-command: "
            "mic=%s,"
            "wake=%s,"
            "vad=%s,"
            "asr=%s,"
            "intent=%s,"
            "handle=%s,"
            "tts=%s,"
            "snd=%s,"
            "stop_after=%s",
            mic_program,
            wake_program,
            vad_program,
            asr_program,
            intent_program,
            handle_program,
            tts_program,
            snd_program,
            stop_after,
        )

        pipeline_result = await run_pipeline(
            rhasspy,
            pipeline,
            samples_per_chunk,
            asr_chunks_to_buffer=asr_chunks_to_buffer,
            mic_program=mic_program,
            wake_program=wake_program,
            asr_program=asr_program,
            vad_program=vad_program,
            intent_program=intent_program,
            handle_program=handle_program,
            tts_program=tts_program,
            snd_program=snd_program,
            stop_after=StopAfterDomain(stop_after) if args.stop_after else None,
        )

        return jsonify(pipeline_result.to_dict())

    @app.websocket("/api/stream-to-stream")
    async def ws_api_stream_to_stream() -> None:
        used_pipeline = pipeline
        pipeline_name = websocket.args.get("pipeline")
        if pipeline_name:
            used_pipeline = rhasspy.config.pipelines[pipeline_name]

        asr_program = websocket.args.get("asr_program", used_pipeline.asr)
        assert asr_program, "Missing program for asr"

        vad_program = websocket.args.get("vad_program", used_pipeline.vad)
        assert vad_program, "Missing program for vad"

        handle_program = websocket.args.get("handle_program", used_pipeline.handle)
        assert handle_program, "Missing program for handle"

        tts_program = websocket.args.get("tts_program", used_pipeline.tts)
        assert tts_program, "Missing program for tts"

        in_rate = int(websocket.args.get("in_rate", DEFAULT_IN_RATE))
        in_width = int(websocket.args.get("in_width", DEFAULT_IN_WIDTH))
        in_channels = int(websocket.args.get("in_channels", DEFAULT_IN_CHANNELS))

        out_rate = int(websocket.args.get("out_rate", DEFAULT_OUT_RATE))
        out_width = int(websocket.args.get("out_width", DEFAULT_OUT_WIDTH))
        out_channels = int(websocket.args.get("out_channels", DEFAULT_OUT_CHANNELS))

        # asr + vad
        async with (
            await create_process(rhasspy, ASR_DOMAIN, asr_program)
        ) as asr_proc, (
            await create_process(rhasspy, VAD_DOMAIN, vad_program)
        ) as vad_proc:
            assert asr_proc.stdin is not None
            assert asr_proc.stdout is not None
            assert vad_proc.stdin is not None
            assert vad_proc.stdout is not None

            mic_task = asyncio.create_task(websocket.receive())
            vad_task = asyncio.create_task(async_read_event(vad_proc.stdout))
            pending = {mic_task, vad_task}

            while True:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )

                if mic_task in done:
                    audio_bytes = mic_task.result()
                    if isinstance(audio_bytes, bytes) and audio_bytes:
                        mic_chunk = AudioChunk(
                            in_rate, in_width, in_channels, audio_bytes
                        )
                        mic_chunk_event = mic_chunk.event()
                        await asyncio.gather(
                            async_write_event(mic_chunk_event, asr_proc.stdin),
                            async_write_event(mic_chunk_event, vad_proc.stdin),
                        )

                    mic_task = asyncio.create_task(websocket.receive())
                    pending.add(mic_task)

                if vad_task in done:
                    vad_event = vad_task.result()
                    if vad_event is None:
                        break

                    if VoiceStarted.is_type(vad_event.type):
                        _LOGGER.debug("stream-to-stream: voice started")
                    elif VoiceStopped.is_type(vad_event.type):
                        _LOGGER.debug("stream-to-stream: voice stopped")
                        break

                    vad_task = asyncio.create_task(async_read_event(vad_proc.stdout))
                    pending.add(vad_task)

            await websocket.send("stop")

            # Get transcript from asr
            await async_write_event(AudioStop().event(), asr_proc.stdin)
            transcript: Optional[Transcript] = None
            while True:
                asr_event = await async_read_event(asr_proc.stdout)
                if asr_event is None:
                    break

                if Transcript.is_type(asr_event.type):
                    transcript = Transcript.from_event(asr_event)
                    _LOGGER.debug("stream-to-stream: asr=%s", transcript)
                    break

            handle_result: Optional[Union[Handled, NotHandled]] = None
            if transcript is not None:
                handle_result = await handle(rhasspy, handle_program, transcript)
                _LOGGER.debug("stream-to-stream: handle=%s", handle_result)

            if (handle_result is not None) and handle_result.text:
                _LOGGER.debug("stream-to-stream: sending tts")
                converter = AudioChunkConverter(out_rate, out_width, out_channels)
                async for tts_chunk in synthesize_stream(
                    rhasspy, tts_program, handle_result.text
                ):
                    tts_chunk = converter.convert(tts_chunk)
                    await websocket.send(tts_chunk.audio)

                _LOGGER.debug("stream-to-stream: tts done")
