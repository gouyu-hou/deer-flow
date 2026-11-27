"""Shared utility helpers for the Gateway layer."""

import hmac
import re

# C0 and C1 controls plus DEL, and the two Unicode line/paragraph separators
# that some log viewers render as a line break. A thread id, an OAuth error or
# a provider name reaches these helpers straight off the request, where a bare
# CR or a terminal escape in the middle of a log line can rewrite the line.
_CONTROL_CHARS_RE = re.compile("[\x00-\x1f\x7f-\x9f\u2028\u2029]")


def sanitize_log_param(value: str) -> str:
    """Strip control characters to prevent log injection."""
    return _CONTROL_CHARS_RE.sub("", value)


def constant_time_equals(a: str, b: str) -> bool:
    """Compare two secret strings in constant time, never raising on content.

    ``hmac.compare_digest`` raises ``TypeError`` for ``str`` operands holding
    non-ASCII characters, and headers, cookies and query values are client
    controlled, so a stray byte would turn a rejection into a 500. Comparing
    the UTF-8 encodings keeps the same result for ASCII input; ``surrogatepass``
    lets even a lone surrogate encode instead of raising.
    """
    return hmac.compare_digest(a.encode("utf-8", "surrogatepass"), b.encode("utf-8", "surrogatepass"))
