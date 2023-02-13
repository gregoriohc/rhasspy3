import argparse
import logging

from quart import Response, request, Quart, jsonify

from rhasspy3.core import Rhasspy
from rhasspy3.config import PipelineConfig
from rhasspy3.intent import recognize, Intent

_LOGGER = logging.getLogger(__name__)


def add_intent(
    app: Quart, rhasspy: Rhasspy, pipeline: PipelineConfig, args: argparse.Namespace
) -> None:
    @app.route("/intent/recognize", methods=["GET", "POST"])
    async def http_intent_recognize() -> Response:
        if request.method == "GET":
            text = request.args["text"]
        else:
            text = (await request.data).decode()

        intent_program = request.args.get("intent_program", pipeline.intent)
        assert intent_program, "Missing program for intent"
        _LOGGER.debug("recognize: intent=%s, text='%s'", intent_program, text)

        result = await recognize(rhasspy, intent_program, text)
        _LOGGER.debug("recognize: result=%s", result)

        return jsonify(result.event().to_dict() if result is not None else {})
