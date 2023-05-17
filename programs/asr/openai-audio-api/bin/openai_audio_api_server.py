#!/usr/bin/env python3
import argparse
import asyncio
import datetime
import io
import logging
import os
import wave
from functools import partial
from pathlib import Path
from typing import Optional

from openai_audio_api import AudioModel

from wyoming.asr import Transcript
from wyoming.audio import AudioChunk, AudioStop
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info
from wyoming.server import AsyncEventHandler, AsyncServer

_FILE = Path(__file__)
_DIR = _FILE.parent
_LOGGER = logging.getLogger(_FILE.stem)

_WHISPER_LANGUAGES = [
    "af",
    "am",
    "ar",
    "as",
    "az",
    "ba",
    "be",
    "bg",
    "bn",
    "bo",
    "br",
    "bs",
    "ca",
    "cs",
    "cy",
    "da",
    "de",
    "el",
    "en",
    "es",
    "et",
    "eu",
    "fa",
    "fi",
    "fo",
    "fr",
    "gl",
    "gu",
    "ha",
    "haw",
    "he",
    "hi",
    "hr",
    "ht",
    "hu",
    "hy",
    "id",
    "is",
    "it",
    "ja",
    "jw",
    "ka",
    "kk",
    "km",
    "kn",
    "ko",
    "la",
    "lb",
    "ln",
    "lo",
    "lt",
    "lv",
    "mg",
    "mi",
    "mk",
    "ml",
    "mn",
    "mr",
    "ms",
    "mt",
    "my",
    "ne",
    "nl",
    "nn",
    "no",
    "oc",
    "pa",
    "pl",
    "ps",
    "pt",
    "ro",
    "ru",
    "sa",
    "sd",
    "si",
    "sk",
    "sl",
    "sn",
    "so",
    "sq",
    "sr",
    "su",
    "sv",
    "sw",
    "ta",
    "te",
    "tg",
    "th",
    "tk",
    "tl",
    "tr",
    "tt",
    "uk",
    "ur",
    "uz",
    "vi",
    "yi",
    "yo",
    "zh",
]


class OpenAiAudioApiEventHandler(AsyncEventHandler):
    def __init__(
        self,
        wyoming_info: Info,
        cli_args: argparse.Namespace,
        model: AudioModel,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.cli_args = cli_args
        self.wyoming_info_event = wyoming_info.event()
        self.model = model

        self._wav_io = None
        self._wav_file: Optional[wave.Wave_write] = None

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent info")
            return True

        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)

            if self._wav_file is None:
                _LOGGER.debug("Receiving audio")
                timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                self._wav_io = f"var/run/audio_{timestamp}.wav"
                self._wav_file = wave.open(self._wav_io, "wb")
                self._wav_file.setframerate(chunk.rate)
                self._wav_file.setsampwidth(chunk.width)
                self._wav_file.setnchannels(chunk.channels)

            self._wav_file.writeframes(chunk.audio)
            return True

        if (
            AudioStop.is_type(event.type)
            and (self._wav_io is not None)
            and (self._wav_file is not None)
        ):
            _LOGGER.debug("Audio stopped")
            self._wav_file.close()
            segments, _info = self.model.transcribe(
                self._wav_io,
                language=self.cli_args.language,
            )

            text = " ".join(segment.text for segment in segments)
            _LOGGER.info(text)

            await self.write_event(Transcript(text=text).event())
            _LOGGER.debug("Completed request")

            os.remove(self._wav_io)
            self._wav_io = None
            self._wav_file = None

            return False

        return True


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("api_key", help="OpenAI API key to use")
    parser.add_argument("model", help="Model name")
    parser.add_argument("--uri", required=True, help="unix:// or tcp://")
    parser.add_argument(
        "--language",
        help="Language to set for transcription",
    )
    parser.add_argument("--debug", action="store_true", help="Log DEBUG messages")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    wyoming_info = Info(
        asr=[
            AsrProgram(
                name="openai-audio-api",
                attribution=Attribution(
                    name="OpenAI",
                    url="https://github.com/openai/openai-python",
                ),
                installed=True,
                models=[
                    AsrModel(
                        name=args.model,
                        attribution=Attribution(
                            name="OpenAI",
                            url="https://platform.openai.com/docs/models/whisper",
                        ),
                        installed=True,
                        languages=_WHISPER_LANGUAGES,
                    )
                ],
            )
        ],
    )

    # Load converted faster-whisper model
    model = AudioModel(args.model, api_key=args.api_key)

    server = AsyncServer.from_uri(args.uri)
    _LOGGER.info("Ready")
    await server.run(
        partial(
            OpenAiAudioApiEventHandler,
            wyoming_info,
            args,
            model,
        )
    )


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(main())
