#!/usr/bin/env python3
import argparse
import asyncio
import logging
from functools import partial

from wyoming.info import AsrModel, AsrProgram, Attribution, Info
from wyoming.server import AsyncServer

from .const import LANGUAGES
from .openai_audio_api import OpenAIAudioModel
from .openai_audio_api import AudioModel
from .handler import OpenAIAudioAPIEventHandler

_LOGGER = logging.getLogger(__name__)


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api-key",
        required=True,
        help="OpenAI API key to use",
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=list(OpenAIAudioModel),
        help="Name of OpenAI Audio API model to use",
    )
    parser.add_argument("--uri", required=True, help="unix:// or tcp://")
    parser.add_argument(
        "--language",
        help="Language to set for transcription",
    )
    parser.add_argument("--debug", action="store_true", help="Log DEBUG messages")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    # Look for model
    model = OpenAIAudioModel(args.model)

    if args.language and (args.language != "auto"):
        _LOGGER.debug("Language: %s", args.language)
        languages = [args.language]
    else:
        languages = LANGUAGES

        # OpenAI Audio API does not understand "auto"
        args.language = None

    wyoming_info = Info(
        asr=[
            AsrProgram(
                name="openai-audio-api",
                attribution=Attribution(
                    name="OpenAI",
                    url="https://github.com/openai/openai-python/tree/main",
                ),
                installed=True,
                models=[
                    AsrModel(
                        name=model.value,
                        attribution=Attribution(
                            name="OpenAI",
                            url="https://platform.openai.com/docs/models/whisper",
                        ),
                        installed=True,
                        languages=languages,
                    )
                ],
            )
        ],
    )

    # Load openai-audio-api model
    _LOGGER.debug("Loading %s", model.value)
    audio_model = AudioModel(
        model.value,
        api_key=args.api_key,
    )

    server = AsyncServer.from_uri(args.uri)
    _LOGGER.info("Ready")
    await server.run(
        partial(
            OpenAIAudioAPIEventHandler,
            wyoming_info,
            args,
            audio_model,
        )
    )


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(main())
