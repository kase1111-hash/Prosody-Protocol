"""``python -m prosody_protocol``: the ``prosody-protocol`` command line.

Useful where the console script is not on ``PATH``. Only the CLI module is
imported here; each subcommand imports what it needs when it runs.
"""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
