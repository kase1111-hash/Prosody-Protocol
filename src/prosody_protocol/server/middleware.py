"""ASGI middleware: request body size limit and per-client rate limit.

Both are plain ASGI middleware (not ``BaseHTTPMiddleware``): the size limit
has to see the body as it arrives, and neither needs a ``Request`` object.
"""

from __future__ import annotations

import ipaddress
import math
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Collection, Sequence

from starlette.datastructures import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .config import IPNetwork
from .errors import error_response


class _BodyTooLarge(Exception):
    """Raised from ``receive`` once the body is over the limit."""


class UploadSizeLimitMiddleware:
    """Reject request bodies larger than ``max_bytes`` with 413.

    A ``Content-Length`` over the limit is rejected before the body is read.
    Otherwise the bytes actually received are counted, so chunked uploads
    (which have no ``Content-Length``) are limited too: once the count passes
    ``max_bytes`` the application gets an exception instead of more body, and
    whatever response it produces for that is replaced by the 413.
    """

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        declared = _content_length(scope)
        if declared is not None and declared > self.max_bytes:
            response = error_response(
                413,
                "payload_too_large",
                f"Request body ({declared} bytes) exceeds maximum allowed size "
                f"({self.max_bytes} bytes).",
            )
            await response(scope, receive, send)
            return

        received = 0
        exceeded = False
        response_started = False

        async def counting_receive() -> Message:
            nonlocal received, exceeded
            if exceeded:
                raise _BodyTooLarge
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    exceeded = True
                    raise _BodyTooLarge
            return message

        async def guarded_send(message: Message) -> None:
            nonlocal response_started
            if exceeded and not response_started:
                return  # The app's reaction to the cut-off body; a 413 is sent instead.
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, counting_receive, guarded_send)
        except Exception:
            # Frameworks may re-raise _BodyTooLarge as something else.
            if not exceeded or response_started:
                raise
        if exceeded and not response_started:
            response = error_response(
                413,
                "payload_too_large",
                f"Request body exceeds maximum allowed size ({self.max_bytes} bytes).",
            )
            await response(scope, receive, send)


def _content_length(scope: Scope) -> int | None:
    raw = Headers(scope=scope).get("content-length")
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None  # The server rejects it; the body is counted meanwhile.


class RateLimitMiddleware:
    """In-memory sliding-window rate limit: ``requests_per_minute`` per client.

    The client is the connecting address. ``X-Forwarded-For`` is used only
    when that address is one of ``trusted_proxies``: the header's addresses
    are read from the right, skipping trusted proxies, and the first other
    address is the client (the one the nearest proxy saw). Requests to
    ``exempt_paths`` (the health check) are neither limited nor counted.

    Rejected requests get 429 with a ``Retry-After`` header. At most
    ``max_clients`` clients are remembered; the least recently seen are
    forgotten first, so memory stays bounded under address floods.

    For production behind a reverse proxy, prefer the proxy's rate limiting
    (nginx, Traefik); this middleware is a safety net for direct exposure.
    """

    WINDOW_S = 60.0

    def __init__(
        self,
        app: ASGIApp,
        requests_per_minute: int,
        *,
        trusted_proxies: Sequence[IPNetwork] = (),
        exempt_paths: Collection[str] = ("/v1/health",),
        max_clients: int = 10_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.app = app
        self.rpm = requests_per_minute
        self.trusted_proxies = tuple(trusted_proxies)
        self.exempt_paths = frozenset(exempt_paths)
        self.max_clients = max_clients
        self._clock = clock
        # Least recently seen client first.
        self._hits: OrderedDict[str, deque[float]] = OrderedDict()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or self.rpm <= 0 or scope["path"] in self.exempt_paths:
            await self.app(scope, receive, send)
            return

        now = self._clock()
        cutoff = now - self.WINDOW_S
        self._forget_idle(cutoff)

        client = self.client_address(scope)
        hits = self._hits.pop(client, None)
        if hits is None:
            hits = deque()
        self._hits[client] = hits
        while len(self._hits) > self.max_clients:
            self._hits.popitem(last=False)
        while hits and hits[0] <= cutoff:
            hits.popleft()

        if len(hits) >= self.rpm:
            retry_after = max(1, math.ceil(hits[0] + self.WINDOW_S - now))
            response = error_response(
                429,
                "rate_limited",
                f"Rate limit exceeded ({self.rpm} requests/minute).",
                headers={"Retry-After": str(retry_after)},
            )
            await response(scope, receive, send)
            return

        hits.append(now)
        await self.app(scope, receive, send)

    def client_address(self, scope: Scope) -> str:
        """The address requests from *scope* are counted against."""
        client = scope.get("client")
        peer = str(client[0]) if client else "unknown"
        if not self.trusted_proxies or not self._is_trusted(peer):
            return peer
        forwarded = [
            address.strip()
            for header in Headers(scope=scope).getlist("x-forwarded-for")
            for address in header.split(",")
            if address.strip()
        ]
        for address in reversed(forwarded):
            if not self._is_trusted(address):
                return address
        return forwarded[0] if forwarded else peer

    def _is_trusted(self, address: str) -> bool:
        try:
            ip = ipaddress.ip_address(address)
        except ValueError:
            return False
        return any(ip.version == net.version and ip in net for net in self.trusted_proxies)

    def _forget_idle(self, cutoff: float) -> None:
        # Clients are ordered by last request, so idle ones are at the front.
        while self._hits:
            hits = next(iter(self._hits.values()))
            if hits and hits[-1] > cutoff:
                break
            self._hits.popitem(last=False)
