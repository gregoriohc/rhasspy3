#!/usr/bin/env python3
import logging
import sys
from pathlib import Path

from snowboy_shared import get_arg_parser, load_snowboy

_FILE = Path(__file__)
_DIR = _FILE.parent
_LOGGER = logging.getLogger(_FILE.stem)

# -----------------------------------------------------------------------------


def main() -> None:
    """Main method."""
    parser = get_arg_parser()
    args = parser.parse_args()

    # logging.basicConfig wouldn't work if a handler already existed.
    # snowboy must mess with logging, so this resets it.
    logging.getLogger().handlers = []
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    detectors = load_snowboy(args)
    bytes_per_chunk = args.samples_per_chunk * 2  # 16-bit samples

    # Read 16Khz, 16-bit mono PCM from stdin
    try:
        chunk = bytes()
        next_chunk = sys.stdin.buffer.read(bytes_per_chunk)
        while next_chunk:
            while len(chunk) >= bytes_per_chunk:
                for name, detector in detectors.items():
                    # Return is:
                    # -2 silence
                    # -1 error
                    #  0 voice
                    #  n index n-1
                    result_index = detector.RunDetection(chunk[:bytes_per_chunk])

                    if result_index > 0:
                        # Detection
                        print(name, flush=True)

                chunk = chunk[bytes_per_chunk:]

            next_chunk = sys.stdin.buffer.read(bytes_per_chunk)
            chunk += next_chunk
    except KeyboardInterrupt:
        pass


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
