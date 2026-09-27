"""Command-line interface: ``prosody-protocol``."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="prosody-protocol")
    parser.parse_args(argv)
    parser.print_help()
    return 0
