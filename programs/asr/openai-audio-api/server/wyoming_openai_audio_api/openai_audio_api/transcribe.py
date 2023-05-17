import collections
from enum import Enum
import openai


class Segment(collections.namedtuple("Segment", ("start", "end", "text"))):
    pass


class AudioInfo(
    collections.namedtuple("AudioInfo", ("language"))
):
    pass

class OpenAIAudioModel(str, Enum):
    """Available OpenAI Audio API models."""

    WHISPER_1 = "whisper-1"


class AudioModel:
    def __init__(
        self,
        model_name,
        api_key,
    ):
        """Initializes the OpenAI Audio API model.

        Args:
          model_name: OpenAI audio model name.
          api_key: OpenAI API Key.
        """
        self.model_name = model_name
        openai.api_key = api_key

    def transcribe(
        self,
        audio_bytes: bytes,
        language=None,
        temperature=0.0,
    ):
        """Transcribes an input file.

        Arguments:
          audio_bytes: Audio bytes.
          language: The language spoken in the audio. If not set, the language will be
            detected in the first 30 seconds of audio.
          temperature: Temperature for sampling. It can be a tuple of temperatures,
            which will be successively used upon failures according to either
            `compression_ratio_threshold` or `logprob_threshold`.

        Returns:
          A tuple with:

            - a generator over transcribed segments
            - an instance of AudioInfo
        """
        with open('myfile.wav', mode='bx') as f:
            f.write(audio_bytes)
        f = open('myfile.wav', "rb")

        params = {
            "temperature": temperature,
            "response_format": "verbose_json",
        }
        if language is not None:
            params["language"] = language
        
        transcript = openai.Audio.transcribe(self.model_name, f, **params)

        segments = self.generate_segments(transcript.segments)

        audio_info = AudioInfo(
            language=transcript.language,
        )

        return segments, audio_info

    def generate_segments(self, segments):
        for segment in segments:
            yield Segment(
                start=segment.start,
                end=segment.end,
                text=segment.text,
            )

