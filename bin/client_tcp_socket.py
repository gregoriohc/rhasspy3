#!/usr/bin/env python3
"""Stream Wyoming events to and from a TCP socket server."""
import argparse
import asyncio
import logging

from rhasspy3.event import (
    async_get_stdin,
    async_read_event,
    async_write_event,
    write_event,
)

_LOGGER = logging.getLogger(__name__)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("address", help="ip:port of server")
    parser.add_argument("--debug", action="store_true", help="Log DEBUG messages")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    host, port_str = args.address.split(":", maxsplit=1)
    port = int(port_str)

    _LOGGER.debug("Connecting to %s", args.address)
    stdin_reader = await async_get_stdin()
    sock_reader, sock_writer = await asyncio.open_connection(host=host, port=port)
    _LOGGER.debug("Connected")

    # stdin -> socket
    # socket -> stdout
    stdin_task = asyncio.create_task(async_read_event(stdin_reader))
    sock_task = asyncio.create_task(async_read_event(sock_reader))
    pending = {stdin_task, sock_task}

    try:
        while True:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            if stdin_task in done:
                event = stdin_task.result()
                if event is None:
                    break

                # Forward to socket
                await async_write_event(event, sock_writer)

                # Next stdin event
                stdin_task = asyncio.create_task(async_read_event(stdin_reader))
                pending.add(stdin_task)

            if sock_task in done:
                event = sock_task.result()
                if event is None:
                    break

                # Forward to stdout (blocking)
                write_event(event)

                # Next socket event
                sock_task = asyncio.create_task(async_read_event(sock_reader))
                pending.add(sock_task)
    finally:
        for task in pending:
            task.cancel()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
