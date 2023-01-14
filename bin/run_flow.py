#!/usr/bin/env python3
import argparse
import asyncio
import logging
import time
from collections import deque
from enum import auto, Enum
from typing import Deque, Optional, Union

from rhasspy3.core import Rhasspy
from rhasspy3.event import async_read_event, async_write_event, Event
from rhasspy3.program import create_process
from rhasspy3.audio import AudioChunk, AudioStop, AudioStart
from rhasspy3.asr import DOMAIN as ASR_DOMAIN, Transcript
from rhasspy3.mic import DOMAIN as MIC_DOMAIN
from rhasspy3.vad import DOMAIN as VAD_DOMAIN, VoiceStarted, VoiceStopped
from rhasspy3.wake import DOMAIN as WAKE_DOMAIN, Detection
from rhasspy3.intent import DOMAIN as INTENT_DOMAIN, Recognize, NotRecognized, Intent
from rhasspy3.handle import DOMAIN as HANDLE_DOMAIN, Handled, NotHandled
from rhasspy3.tts import DOMAIN as TTS_DOMAIN, Synthesize
from rhasspy3.snd import DOMAIN as SND_DOMAIN

_LOGGER = logging.getLogger("run_flow")


class State(Enum):
    DETECT_WAKE = auto()
    BEFORE_COMMAND = auto()
    IN_COMMAND = auto()


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Configuration directory",
    )
    parser.add_argument("flow", help="Name of flow to run")
    parser.add_argument("--asr-buffer-chunks", type=int, default=0)
    parser.add_argument("--debug", action="store_true", help="Log DEBUG messages")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    rhasspy = Rhasspy.load(args.config)
    flow = rhasspy.config.flows[args.flow]
    procs = []

    try:
        mic_proc = await create_process(rhasspy, MIC_DOMAIN, flow.mic)
        assert mic_proc.stdout is not None
        procs.append(mic_proc)

        wake_proc = await create_process(rhasspy, WAKE_DOMAIN, flow.wake)
        assert wake_proc.stdin is not None
        assert wake_proc.stdout is not None
        procs.append(wake_proc)

        vad_proc = await create_process(rhasspy, VAD_DOMAIN, flow.vad)
        assert vad_proc.stdin is not None
        assert vad_proc.stdout is not None
        procs.append(vad_proc)

        asr_proc = await create_process(rhasspy, ASR_DOMAIN, flow.asr)
        assert asr_proc.stdin is not None
        assert asr_proc.stdout is not None
        procs.append(asr_proc)

        intent_proc = await create_process(rhasspy, INTENT_DOMAIN, flow.intent)
        assert intent_proc.stdin is not None
        assert intent_proc.stdout is not None
        procs.append(intent_proc)

        handle_proc = await create_process(rhasspy, HANDLE_DOMAIN, flow.handle)
        assert handle_proc.stdin is not None
        assert handle_proc.stdout is not None
        procs.append(handle_proc)

        tts_proc = await create_process(rhasspy, TTS_DOMAIN, flow.tts)
        assert tts_proc.stdin is not None
        assert tts_proc.stdout is not None
        procs.append(tts_proc)

        snd_proc = await create_process(rhasspy, SND_DOMAIN, flow.snd)
        assert snd_proc.stdin is not None
        procs.append(snd_proc)

        state = State.DETECT_WAKE
        stt_chunks: Deque[Event] = deque(maxlen=args.asr_buffer_chunks)
        timestamp = 0

        mic_task = asyncio.create_task(async_read_event(mic_proc.stdout))
        wake_task = asyncio.create_task(async_read_event(wake_proc.stdout))
        vad_task = asyncio.create_task(async_read_event(vad_proc.stdout))
        pending = {mic_task, wake_task, vad_task}

        # Silence detection + speech recognition
        while True:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            if mic_task in done:
                mic_event = mic_task.result()
                if mic_event is None:
                    break

                # Process chunk
                if AudioChunk.is_type(mic_event.type):
                    chunk = AudioChunk.from_event(mic_event)
                    timestamp = (
                        chunk.timestamp
                        if chunk.timestamp is not None
                        else time.monotonic_ns()
                    )

                    if state == State.DETECT_WAKE:
                        # Wake word detection
                        await async_write_event(mic_event, wake_proc.stdin)
                        stt_chunks.append(mic_event)
                    elif state == State.BEFORE_COMMAND:
                        # Voice detection
                        await asyncio.gather(
                            async_write_event(mic_event, asr_proc.stdin),
                            async_write_event(mic_event, vad_proc.stdin),
                        )
                    elif state == State.IN_COMMAND:
                        # Speech recognition and silence detection
                        await asyncio.gather(
                            async_write_event(mic_event, asr_proc.stdin),
                            async_write_event(mic_event, vad_proc.stdin),
                        )

                # Next chunk
                mic_task = asyncio.create_task(async_read_event(mic_proc.stdout))
                pending.add(mic_task)

            if wake_task in done:
                wake_event = wake_task.result()
                if wake_event is None:
                    break

                if Detection.is_type(wake_event.type):
                    if state == State.DETECT_WAKE:
                        detection = Detection.from_event(wake_event)
                        print(detection)
                        await async_write_event(
                            AudioStart(timestamp=timestamp).event(), asr_proc.stdin
                        )
                        # Drain chunk queue
                        write_coros = [
                            async_write_event(chunk_event, asr_proc.stdin)
                            for chunk_event in stt_chunks
                        ]
                        await asyncio.gather(*write_coros)
                        stt_chunks.clear()

                        state = State.BEFORE_COMMAND

                # Next wake event
                wake_task = asyncio.create_task(async_read_event(wake_proc.stdout))
                pending.add(wake_task)

            if vad_task in done:
                vad_event = vad_task.result()
                if vad_event is None:
                    break

                if VoiceStarted.is_type(vad_event.type):
                    if state == State.BEFORE_COMMAND:
                        _LOGGER.debug("Voice command started")
                        state = State.IN_COMMAND
                elif VoiceStopped.is_type(vad_event.type):
                    # End of voice command
                    _LOGGER.debug("Voice command stopped")
                    await async_write_event(
                        AudioStop(timestamp=timestamp).event(), asr_proc.stdin
                    )
                    break

                # Next VAD event
                vad_task = asyncio.create_task(async_read_event(vad_proc.stdout))
                pending.add(vad_task)

        # Wait for transcript
        transcript: Optional[Transcript] = None
        while True:
            asr_event = await async_read_event(asr_proc.stdout)
            if asr_event is None:
                break

            if Transcript.is_type(asr_event.type):
                transcript = Transcript.from_event(asr_event)
                break

        print(transcript)

        # Recognize intent
        intent_result: Optional[Union[Intent, NotRecognized]] = None
        if transcript is not None:
            await async_write_event(
                Recognize(text=transcript.text).event(), intent_proc.stdin
            )
            while True:
                intent_event = await async_read_event(intent_proc.stdout)
                if intent_event is None:
                    break

                if Intent.is_type(intent_event.type):
                    intent_result = Intent.from_event(intent_event)
                    break

                if NotRecognized.is_type(intent_event.type):
                    intent_result = NotRecognized.from_event(intent_event)
                    break

        print(intent_result)

        # Handle intent
        handle_result: Optional[Union[Handled, NotHandled]] = None
        if isinstance(intent_result, Intent):
            await async_write_event(intent_result.event(), handle_proc.stdin)
            while True:
                handle_event = await async_read_event(handle_proc.stdout)
                if handle_event is None:
                    break

                if Handled.is_type(handle_event.type):
                    handle_result = Handled.from_event(handle_event)
                    break

                if NotHandled.is_type(handle_event.type):
                    handle_result = NotHandled.from_event(handle_event)
                    break

        print(handle_result)

        # Text to speech
        if (handle_result is not None) and handle_result.text:
            await async_write_event(
                Synthesize(text=handle_result.text).event(), tts_proc.stdin
            )
            while True:
                tts_event = await async_read_event(tts_proc.stdout)
                if tts_event is None:
                    break

                if AudioChunk.is_type(tts_event.type):
                    await async_write_event(tts_event, snd_proc.stdin)
                elif AudioStop.is_type(tts_event.type):
                    break

    finally:
        for proc in procs:
            if proc.returncode is None:
                proc.terminate()


if __name__ == "__main__":
    asyncio.run(main())
