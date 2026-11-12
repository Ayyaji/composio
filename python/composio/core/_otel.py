"""Optional OpenTelemetry spans for tool execution.

``opentelemetry-api`` is not a runtime dependency. When it is missing, spans
are skipped and tool execution behaves as it does without tracing. Spans are
also inert when the API is installed but no SDK tracer provider is configured,
because the API's proxy tracer does not record.
"""

from __future__ import annotations

import contextvars
import typing as t
from contextlib import contextmanager

_tracer: t.Any = None
_Status: t.Any = None
_StatusCode: t.Any = None

try:
    from opentelemetry import trace as _trace
    from opentelemetry.trace import Status as _ImportedStatus
    from opentelemetry.trace import StatusCode as _ImportedStatusCode

    _Status = _ImportedStatus
    _StatusCode = _ImportedStatusCode
    # ProxyTracer follows a provider installed later, so importing this module
    # before the application calls set_tracer_provider is safe.
    _tracer = _trace.get_tracer("composio")
except ImportError:  # pragma: no cover
    pass

_active_span: contextvars.ContextVar[t.Any] = contextvars.ContextVar(
    "composio_execute_tool_span", default=None
)

_EXECUTE_TOOL = "execute_tool"
_ATTR_OPERATION = "gen_ai.operation.name"
_ATTR_TOOL = "gen_ai.tool.name"
_ATTR_TOOLKIT = "composio.toolkit.slug"
_ATTR_VERSION = "composio.tool.version"
_FAILED = "tool execution failed"


def _known(value: t.Optional[str]) -> t.Optional[str]:
    if isinstance(value, str) and value:
        return value
    return None


@contextmanager
def execute_tool_span(
    slug: str,
    *,
    toolkit_slug: t.Optional[str] = None,
    version: t.Optional[str] = None,
) -> t.Iterator[t.Any]:
    """Span covering one tool execution, or a no-op when tracing is absent.

    The span is current, so it nests under a caller that already has a trace.
    Only the slug, toolkit, and version are recorded.
    """
    tracer = _tracer
    if tracer is None:
        yield None
        return

    attributes: t.Dict[str, str] = {
        _ATTR_OPERATION: _EXECUTE_TOOL,
        _ATTR_TOOL: slug,
    }
    known_toolkit = _known(toolkit_slug)
    if known_toolkit is not None:
        attributes[_ATTR_TOOLKIT] = known_toolkit
    known_version = _known(version)
    if known_version is not None:
        attributes[_ATTR_VERSION] = known_version

    with tracer.start_as_current_span(
        f"{_EXECUTE_TOOL} {slug}",
        attributes=attributes,
    ) as span:
        token = _active_span.set(span)
        try:
            yield span
        finally:
            _active_span.reset(token)


def mark_failed(
    span: t.Any, error: t.Optional[t.Union[BaseException, str]] = None
) -> None:
    """Set ``span`` to ERROR for a result that returned instead of raising.

    Exceptions that leave :func:`execute_tool_span` are recorded by the
    OpenTelemetry API. This covers a normal return whose payload says the tool
    failed. ``error`` may be a short, non-sensitive description; exception
    objects are not stringified, because their messages can carry caller data.
    """
    if span is None or _Status is None or _StatusCode is None:
        return
    description = error if isinstance(error, str) and error else _FAILED
    span.set_status(_Status(_StatusCode.ERROR, description))


def note_toolkit_slug(span: t.Any, toolkit_slug: t.Optional[str]) -> None:
    """Record the toolkit once the tool schema (or custom tool) is known."""
    known = _known(toolkit_slug)
    if span is None or known is None:
        return
    span.set_attribute(_ATTR_TOOLKIT, known)


def note_tool_version(version: t.Optional[str]) -> None:
    """Record the version resolved for the active execute span, when there is one."""
    known = _known(version)
    span = _active_span.get()
    if span is None or known is None:
        return
    span.set_attribute(_ATTR_VERSION, known)
