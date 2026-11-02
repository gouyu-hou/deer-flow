"""A configured ``web_fetch`` timeout must not reach HTTPX in an unusable form.

``jina_ai`` sends its timeout twice: as the ``X-Timeout`` header Jina reads, and as
the per-request HTTPX timeout. An integer used to be forwarded verbatim, so
``timeout: 0`` asked HTTPX to give up before connecting and ``10**400`` raised
``OverflowError`` when the socket applied it.
"""

import logging
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest

from deerflow.community.jina_ai import tools

pytestmark = pytest.mark.anyio

URL = "https://example.com/article"
DEFAULT_TIMEOUT = 10
HTML = "<html><body><article><p>ok</p></article></body></html>"
LOGGER = "deerflow.community.jina_ai.tools"


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def wire(monkeypatch):
    """Run the real tool and client offline, capturing every outgoing request."""
    requests = []
    real_client = httpx.AsyncClient

    def handler(request):
        requests.append(request)
        return httpx.Response(200, text=HTML)

    def client(**kwargs):
        return real_client(transport=httpx.MockTransport(handler), trust_env=False)

    monkeypatch.setattr(httpx, "AsyncClient", client)
    monkeypatch.setenv("JINA_API_KEY", "dummy-test-key")
    # Article extraction is not under test; skip the readability pass.
    monkeypatch.setattr(tools, "readability_extractor", SimpleNamespace(extract_article=lambda html, url=None: SimpleNamespace(to_markdown=lambda: "ok")))
    return requests


async def fetch_with(requests, extras):
    """Invoke ``web_fetch_tool`` with ``extras`` as its tool config, return the request sent."""
    config = SimpleNamespace(model_extra=extras)
    app_config = SimpleNamespace(get_tool_config=lambda name: config)
    with patch.object(tools, "get_app_config", return_value=app_config):
        result = await tools.web_fetch_tool.ainvoke(URL)
    assert "Error:" not in result
    return requests[-1]


@pytest.mark.parametrize("configured", [0, -1, "0", "-1", 10**400], ids=["int-zero", "int-negative", "str-zero", "str-negative", "unrepresentable"])
async def test_unusable_timeout_falls_back_on_the_wire(wire, configured):
    """Zero, negative and oversized timeouts must not be forwarded to HTTPX or Jina."""
    request = await fetch_with(wire, {"timeout": configured})

    phases = request.extensions["timeout"]
    assert set(phases.values()) == {DEFAULT_TIMEOUT}
    # float() is what socket.settimeout() applies; 10**400 raises OverflowError there.
    assert float(phases["connect"]) == DEFAULT_TIMEOUT
    assert request.headers["x-timeout"] == str(DEFAULT_TIMEOUT)


@pytest.mark.parametrize(("configured", "expected"), [(45, 45), ("45", 45), (1, 1)], ids=["int", "str", "one-second"])
async def test_positive_timeout_is_forwarded_unchanged(wire, configured, expected):
    """The documented positive-integer and integer-string contract is preserved."""
    request = await fetch_with(wire, {"timeout": configured})

    assert set(request.extensions["timeout"].values()) == {expected}
    assert request.headers["x-timeout"] == str(expected)


@pytest.mark.parametrize("configured", [0, -1, "0", 10**400, True, False, 12.5, "not-a-number"], ids=["int-zero", "int-negative", "str-zero", "unrepresentable", "yaml-on", "yaml-off", "fractional", "non-numeric"])
async def test_unusable_timeout_is_reported(wire, caplog, configured):
    """A rejected value is logged once so the operator can find the typo."""
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        await fetch_with(wire, {"timeout": configured})

    rejected = [record for record in caplog.records if "unusable timeout" in record.getMessage()]
    assert len(rejected) == 1
    assert str(configured) in rejected[0].getMessage()


@pytest.mark.parametrize("extras", [{"timeout": 45}, {}], ids=["configured", "absent"])
async def test_usable_or_absent_timeout_is_not_reported(wire, caplog, extras):
    """A valid value and an unset key must stay silent — this runs on every fetch."""
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        await fetch_with(wire, extras)

    assert not [record for record in caplog.records if "unusable timeout" in record.getMessage()]


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        (0, DEFAULT_TIMEOUT),
        (-1, DEFAULT_TIMEOUT),
        ("0", DEFAULT_TIMEOUT),
        ("-1", DEFAULT_TIMEOUT),
        (10**400, DEFAULT_TIMEOUT),
        (45, 45),
        ("45", 45),
        # Long-standing fallbacks, kept so YAML booleans and typos stay harmless.
        (True, DEFAULT_TIMEOUT),
        (False, DEFAULT_TIMEOUT),
        (12.5, DEFAULT_TIMEOUT),
        ("not-a-number", DEFAULT_TIMEOUT),
        (None, DEFAULT_TIMEOUT),
    ],
)
def test_coerce_timeout_documents_the_policy(configured, expected):
    """Only positive integers and integer strings survive coercion."""
    assert tools._coerce_timeout(configured, DEFAULT_TIMEOUT) == expected
