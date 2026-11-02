import asyncio
import logging

from langchain.tools import tool

from deerflow.community.jina_ai.jina_client import JinaClient
from deerflow.config import get_app_config
from deerflow.utils.readability import ReadabilityExtractor

logger = logging.getLogger(__name__)

readability_extractor = ReadabilityExtractor()


def _coerce_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _coerce_timeout(value: object, default: int) -> int:
    """Coerce a configured request timeout into whole seconds.

    Positive integers and integer strings are kept. Anything else the operator
    actually wrote falls back to ``default`` with a warning, because the value is
    sent twice: as the ``X-Timeout`` header and as the per-request HTTPX timeout.
    ``0`` makes HTTPX give up before connecting, negatives are rejected outright,
    and an integer too large to become a float (``10**400``) raises
    ``OverflowError`` when the socket applies it. ``None`` is the "key absent"
    sentinel and stays silent — this runs on every fetch.
    """
    if value is None:
        return default
    timeout: int | None = None
    if not isinstance(value, bool) and isinstance(value, (int, str)):
        try:
            timeout = int(value)
            # float() here is the conversion socket.settimeout() applies later.
            float(timeout)
        except (ValueError, OverflowError):
            timeout = None
    if timeout is None or timeout <= 0:
        logger.warning("Jina web_fetch: unusable timeout %r in config; using %ss", value, default)
        return default
    return timeout


def _coerce_proxy(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    proxy = value.strip()
    return proxy or None


@tool("web_fetch", parse_docstring=True)
async def web_fetch_tool(url: str) -> str:
    """Fetch the contents of a web page at a given URL.
    Only fetch EXACT URLs that have been provided directly by the user or have been returned in results from the web_search and web_fetch tools.
    This tool can NOT access content that requires authentication, such as private Google Docs or pages behind login walls.
    Do NOT add www. to URLs that do NOT have them.
    URLs must include the schema: https://example.com is a valid URL while example.com is an invalid URL.

    Args:
        url: The URL to fetch the contents of.
    """
    jina_client = JinaClient()
    timeout = 10
    proxy = None
    trust_env = True
    request_options = {}
    # Config resolution stats and hashes the config files; keep it off the event loop.
    config = (await asyncio.to_thread(get_app_config)).get_tool_config("web_fetch")
    if config is not None:
        timeout = _coerce_timeout(config.model_extra.get("timeout"), timeout)
        proxy = _coerce_proxy(config.model_extra.get("proxy"))
        trust_env = _coerce_bool(config.model_extra.get("trust_env"), trust_env)
        request_options = {key: config.model_extra[key] for key in ("max_retries", "retry_budget_seconds", "max_response_bytes", "request_admission") if key in config.model_extra}
    html_content = await jina_client.crawl(url, return_format="html", timeout=timeout, proxy=proxy, trust_env=trust_env, **request_options)
    if isinstance(html_content, str) and html_content.startswith("Error:"):
        return html_content
    article = await asyncio.to_thread(readability_extractor.extract_article, html_content, url=url)
    return article.to_markdown()[:4096]
