# OpenAI API audio transcription

This repository demonstrates how to implement the audio transcription using [OpenAI API](https://platform.openai.com/docs/api-reference/audio/create), which uses its audio models (currently only "whisper-1" is available).

## Installation

```bash
pip install -e .
```

## Usage

### Transcription

```python
from openai_audio_api import AudioModel

model = AudioModel("whisper-1")

segments, info = model.transcribe("audio.mp3")

print("Detected language '%s'" % (info.language))

for segment in segments:
    print("[%ds -> %ds] %s" % (segment.start, segment.end, segment.text))
```
