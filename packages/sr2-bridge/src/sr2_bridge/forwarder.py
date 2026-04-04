"""Async HTTP forwarder to upstream LLM API."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

import httpx

from sr2_bridge.config import BridgeForwardingConfig

logger = logging.getLogger(__name__)

# Hop-by-hop headers that must not be forwarded.
_HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "host",
        "content-length",
    }
)


def _filter_headers(headers: dict[str, str]) -> dict[str, str]:
    """Remove hop-by-hop and internal headers, keep auth and content-type."""
    return {k: v for k, v in headers.items() if k.lower() not in _HOP_BY_HOP}


class BridgeForwarder:
    """Forwards requests to the upstream LLM API using httpx."""

    def __init__(self, config: BridgeForwardingConfig):
        self._upstream = config.upstream_url.rstrip("/")
        self._timeout = config.timeout_seconds
        self._client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        """Create the httpx client."""
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self._timeout, connect=10.0),
            follow_redirects=True,
        )

    async def stop(self) -> None:
        """Close the httpx client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def forward(
        self,
        path: str,
        body: dict,
        headers: dict[str, str],
        query_params: str | None = None,
    ) -> httpx.Response:
        """Forward a non-streaming request to upstream.

        Returns the full httpx.Response (caller reads .json() / .content).
        """
        assert self._client is not None, "Forwarder not started"
        url = f"{self._upstream}{path}"
        if query_params:
            url = f"{url}?{query_params}"
        fwd_headers = _filter_headers(headers)

        logger.debug("Forwarding POST %s (non-streaming)", url)
        response = await self._client.post(url, json=body, headers=fwd_headers)
        return response

    async def forward_streaming(
        self,
        path: str,
        body: dict,
        headers: dict[str, str],
        query_params: str | None = None,
    ) -> AsyncIterator[bytes]:
        """Forward a streaming request, yielding raw SSE lines as bytes.

        Each yielded chunk is a single line from the SSE stream (including
        the trailing newline), suitable for direct passthrough to the caller.

        If the upstream returns a non-2xx status, yields the error body as a
        single SSE error event instead of raising, so the client sees the
        upstream error message.
        """
        assert self._client is not None, "Forwarder not started"
        url = f"{self._upstream}{path}"
        if query_params:
            url = f"{url}?{query_params}"
        fwd_headers = _filter_headers(headers)

        logger.debug("Forwarding POST %s (streaming)", url)
        async with self._client.stream("POST", url, json=body, headers=fwd_headers) as response:
            if response.status_code >= 400:
                error_body = await response.aread()
                logger.warning(
                    "Upstream returned %d: %s",
                    response.status_code,
                    error_body[:500].decode("utf-8", errors="replace"),
                )
                yield (b"event: error\ndata: " + error_body + b"\n\n")
                return
            async for line in response.aiter_lines():
                chunk = (line + "\n").encode("utf-8")
                yield chunk

    async def forward_passthrough(
        self,
        method: str,
        path: str,
        body: bytes | None,
        headers: dict[str, str],
    ) -> httpx.Response:
        """Forward an arbitrary request without body parsing (passthrough).

        Used for endpoints that don't need optimization (e.g. count_tokens).
        """
        assert self._client is not None, "Forwarder not started"
        url = f"{self._upstream}{path}"
        fwd_headers = _filter_headers(headers)

        logger.debug("Passthrough %s %s", method, url)
        response = await self._client.request(method, url, content=body, headers=fwd_headers)
        return response
