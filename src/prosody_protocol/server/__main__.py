"""Entry point for ``python -m prosody_protocol.server``."""

from __future__ import annotations

import argparse

from . import run


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m prosody_protocol.server",
        description="Run the Prosody Protocol REST API.",
    )
    parser.add_argument("--host", help="bind address (default: $PP_HOST or 127.0.0.1)")
    parser.add_argument("--port", type=int, help="port (default: $PP_PORT or 8000)")
    args = parser.parse_args(argv)
    run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
