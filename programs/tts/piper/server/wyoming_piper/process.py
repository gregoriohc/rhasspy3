#!/usr/bin/env python3
import argparse
import asyncio
import json
import logging
import tempfile
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from wyoming.info import Attribution, Info, TtsProgram, TtsVoice, TtsVoiceSpeaker
from wyoming.server import AsyncServer
from wyoming.tts import Synthesize

from .download import ensure_voice_exists, find_voice, get_voices

_LOGGER = logging.getLogger(__name__)


@dataclass
class PiperProcess:
    name: str
    proc: "asyncio.subprocess.Process"
    config: Dict[str, Any]
    wav_dir: tempfile.TemporaryDirectory
    last_used: int = 0


class PiperProcessManager:
    def __init__(self, args: argparse.Namespace, voices_info: Dict[str, Any]):
        self.voices_info = voices_info
        self.args = args
        self.processes: Dict[str, PiperProcess] = {}
        self.processes_lock = asyncio.Lock()

    async def get_process(self, voice_name: Optional[str] = None) -> PiperProcess:
        if voice_name is None:
            # Default voice
            voice_name = self.args.voice

        assert voice_name is not None

        # Resolve alias
        voice_info = self.voices_info.get(voice_name, {})
        voice_name = voice_info.get("key", voice_name)

        piper_proc = self.processes.get(voice_name)
        if piper_proc is None:
            if self.args.max_piper_procs > 0:
                # Restrict number of parallel processes
                while len(self.processes) >= self.args.max_piper_procs:
                    # Stop least recently used process
                    lru_proc_name, lru_proc = sorted(
                        self.processes.items(), key=lambda kv: kv[1].last_used
                    )[0]
                    _LOGGER.debug("Stopping process for: %s", lru_proc_name)
                    self.processes.pop(lru_proc_name, None)
                    if lru_proc.proc.returncode is None:
                        try:
                            lru_proc.proc.terminate()
                            await lru_proc.proc.wait()
                        except Exception:
                            _LOGGER.exception("Unexpected error stopping piper process")

            _LOGGER.debug(
                "Starting process for: %s (%s/%s running)",
                voice_name,
                len(self.processes),
                self.args.max_piper_procs,
            )

            ensure_voice_exists(
                voice_name,
                self.args.data_dir,
                self.args.download_dir,
                self.voices_info,
            )

            onnx_path, config_path = find_voice(voice_name, self.args.data_dir)
            with open(config_path, "r", encoding="utf-8") as config_file:
                config = json.load(config_file)

            wav_dir = tempfile.TemporaryDirectory()
            piper_args = [
                "--model",
                str(onnx_path),
                "--config",
                str(config_path),
                "--output_dir",
                str(wav_dir.name),
                "--json-input",
            ]

            if self.args.noise_scale:
                piper_args.extend(["--noise-scale", str(self.args.noise_scale)])

            if self.args.length_scale:
                piper_args.extend(["--length-scale", str(self.args.length_scale)])

            if self.args.noise_w:
                piper_args.extend(["--noise-w", str(self.args.noise_w)])

            _LOGGER.debug(
                "Starting piper process: %s args=%s", self.args.piper, piper_args
            )
            piper_proc = PiperProcess(
                name=voice_name,
                proc=await asyncio.create_subprocess_exec(
                    self.args.piper,
                    *piper_args,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                ),
                config=config,
                wav_dir=wav_dir,
            )
            self.processes[voice_name] = piper_proc

        # Update used
        piper_proc.last_used = time.monotonic_ns()

        return piper_proc
