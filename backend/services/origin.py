"""WebSocket origin policy (framework-free so it can be unit tested)."""
from typing import Iterable, Optional
from urllib.parse import urlparse


def origin_allowed(origin: Optional[str], host: Optional[str], allowed: Iterable[str]) -> bool:
    """Allow: no Origin header (non-browser clients), an explicitly allowed origin, or the SAME origin the
    page was served from (``Origin`` host == ``Host`` header), which is the case when the backend serves the
    dashboard itself behind a proxy/tunnel."""
    if not origin:
        return True
    if origin in set(allowed):
        return True
    try:
        netloc = urlparse(origin).netloc.lower()
    except ValueError:
        return False
    return bool(host) and netloc == host.lower()
