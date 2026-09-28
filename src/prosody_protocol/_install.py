"""Install instructions used in error messages and ``prosody-protocol doctor``.

The package is not on PyPI yet and nobody has registered the
``prosody-protocol`` name there, so hints must not say ``pip install
prosody-protocol[...]``: today that fails, and it would install whatever
someone else later publishes under the name. They point at the GitHub
repository instead. Change :func:`install_hint` (only) once a release is
published.
"""

from __future__ import annotations

REPOSITORY = "https://github.com/kase1111-hash/Prosody-Protocol"


def install_hint(extra: str | None = None) -> str:
    """Return the pip command that installs the package with *extra*."""
    spec = f"prosody-protocol[{extra}]" if extra else "prosody-protocol"
    return f'pip install "{spec} @ git+{REPOSITORY}"'
