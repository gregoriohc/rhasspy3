import collections
import openai


class Segment(collections.namedtuple("Segment", ("start", "end", "text"))):
    pass


class AudioInfo(
    collections.namedtuple("AudioInfo", ("language"))
):
    pass


class AudioModel:
    def __init__(
        self,
        model_name,
        api_key,
    ):
        """Initializes the Audio model.

        Args:
          model_name: OpenAI audio model name.
          api_key: OpenAI API Key.
        """
        self.model_name = model_name
        openai.api_key = api_key

    def transcribe(
        self,
        input_file,
        language=None,
        temperature=0.0,
    ):
        """Transcribes an input file.

        Arguments:
          input_file: Path to the input file or a file-like object.
          language: The language of the input audio. Supplying the input language in
            ISO-639-1 format will improve accuracy and latency.
          temperature: The sampling temperature, between 0 and 1. Higher values like 0.8
            will make the output more random, while lower values like 0.2 will make it more
            focused and deterministic. If set to 0, the model will use log probability to
            automatically increase the temperature until certain thresholds are hit.

        Returns:
          A tuple with:

            - a generator over transcribed segments
            - an instance of AudioInfo
        """
        f = open(input_file, "rb")

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
