"""Entry point for ``python -m prosody_protocol.server`` (the Docker image's command).

The same as ``prosody-protocol serve``: invalid options or ``PP_*``
settings, and a missing ``api`` extra, print a one-line ``error: ...`` and
exit with status 2.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Run the REST API; returns the exit status."""
    from ..cli import serve_main

    return serve_main(argv, prog="python -m prosody_protocol.server")


if __name__ == "__main__":
    sys.exit(main())
