"""Page discovery — find real, reachable TechAbout pages from the live nav."""
from __future__ import annotations

import re
from urllib.parse import urljoin

import requests

_ASSET_EXT = (".png", ".jpg", ".jpeg", ".svg", ".webp", ".gif", ".ico",
              ".css", ".js", ".xml", ".txt", ".pdf", ".woff", ".woff2")
_LINK_RE = re.compile(r'href="(/[^"#]*)"', re.IGNORECASE)
_ALLOWED = re.compile(r"^/[a-z0-9/-]+$", re.IGNORECASE)


def discover_pages(base_url: str, requested: list[str], timeout: float = 20.0) -> list[str]:
    """Return the requested pages that actually respond 200, in priority order.
    Falls back to ['/'] if nothing else responds."""
    ok: list[str] = []
    for page in ["/"] + [p for p in requested if p != "/"]:
        url = urljoin(base_url, page)
        try:
            r = requests.head(url, timeout=timeout, allow_redirects=True)
            if r.status_code >= 400:
                r = requests.get(url, timeout=timeout, allow_redirects=True,
                                 stream=True)
            if r.status_code < 400:
                if page not in ok:
                    ok.append(page)
        except requests.RequestException:
            continue
    return ok or ["/"]


def nav_links(base_url: str, timeout: float = 20.0) -> list[str]:
    """Internal links visible on the homepage (for context in reports)."""
    try:
        html = requests.get(base_url, timeout=timeout).text
    except requests.RequestException:
        return []
    pages = set()
    for href in _LINK_RE.findall(html):
        if href.endswith(_ASSET_EXT) or not _ALLOWED.match(href):
            continue
        pages.add(href.rstrip("/") or "/")
    return sorted(pages)
