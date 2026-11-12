"""Tests for OpenTelemetry spans around tool execution (issue #5267)."""

from unittest.mock import Mock, patch

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode

from composio.core import _otel
from composio.core.models.base import allow_tracking
from composio.core.models.tool_router_session import ToolRouterSession
from composio.core.models.tools import Tools, after_execute, before_execute
from tests.conftest import mock_http_client
from tests.test_tool_execution import TestToolExecution as _ToolFactory

_ALLOWED_ATTRIBUTES = {
    "gen_ai.operation.name",
    "gen_ai.tool.name",
    "composio.toolkit.slug",
    "composio.tool.version",
    "error.type",
}
_SUCCESS = {"data": {}, "error": None, "successful": True}


@pytest.fixture(autouse=True)
def disable_telemetry():
    token = allow_tracking.set(False)
    yield
    allow_tracking.reset(token)


@pytest.fixture
def exporter():
    memory = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(memory))
    with patch.object(_otel, "_tracer", provider.get_tracer("test")):
        yield memory
    provider.shutdown()


def make_tools(response):
    tools = Tools(client=mock_http_client(), provider=Mock())
    tool = _ToolFactory().create_mock_tool("GITHUB_CREATE_ISSUE", "github")
    tools._tool_schemas["GITHUB_CREATE_ISSUE"] = tool
    tools._execute_tool = Mock(return_value=response)
    return tools


def make_session():
    return ToolRouterSession(
        client=mock_http_client(),
        provider=Mock(),
        dangerously_allow_auto_upload_download_files=False,
        session_id="session_123",
        mcp=Mock(),
        experimental=Mock(),
        user_id="u1",
    )


def one_span(exporter):
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    return spans[0]


def span_attributes(span):
    attributes = span.attributes
    assert attributes is not None
    return attributes


def assert_private(span, *secrets: str) -> None:
    """Span attributes, status text, and events stay inside the allowlist."""
    attributes = span.attributes
    assert attributes is not None
    assert set(attributes).issubset(_ALLOWED_ATTRIBUTES)
    parts = [str(value) for value in attributes.values()]
    if span.status.description:
        parts.append(span.status.description)
    for event in span.events:
        parts.append(event.name)
        if event.attributes:
            parts.extend(str(value) for value in event.attributes.values())
    rendered = " ".join(parts)
    for secret in secrets:
        assert secret not in rendered


def test_execute_creates_span(exporter):
    tools = make_tools(_SUCCESS)
    tools.execute("GITHUB_CREATE_ISSUE", {"title": "secret"}, user_id="u1")

    span = one_span(exporter)
    attributes = span_attributes(span)
    assert span.name == "execute_tool GITHUB_CREATE_ISSUE"
    assert attributes["gen_ai.operation.name"] == "execute_tool"
    assert attributes["gen_ai.tool.name"] == "GITHUB_CREATE_ISSUE"
    assert attributes["composio.toolkit.slug"] == "github"
    assert span.status.status_code == StatusCode.UNSET
    assert "error.type" not in attributes
    assert_private(span, "secret", "u1")


def test_failed_result_marks_span_error(exporter):
    tools = make_tools({"data": {}, "error": "401", "successful": False})
    tools.execute("GITHUB_CREATE_ISSUE", {"title": "secret"}, user_id="u1")

    span = one_span(exporter)
    attributes = span_attributes(span)
    assert span.status.status_code == StatusCode.ERROR
    assert attributes["error.type"] == "tool_error"
    assert span.status.description == "tool execution failed"
    assert list(span.events) == []
    assert_private(span, "secret", "u1", "401")


def test_raised_error_marks_span_error(exporter):
    tools = make_tools(_SUCCESS)
    tools._execute_tool.side_effect = RuntimeError("bad title secret")

    with pytest.raises(RuntimeError, match="bad title secret"):
        tools.execute("GITHUB_CREATE_ISSUE", {"title": "secret"}, user_id="u1")

    span = one_span(exporter)
    attributes = span_attributes(span)
    assert span.status.status_code == StatusCode.ERROR
    assert attributes["error.type"] == "RuntimeError"
    assert span.status.description == "tool execution failed"
    assert list(span.events) == []
    assert_private(span, "secret", "u1", "bad title secret")


def test_no_arguments_or_user_on_span(exporter):
    tools = make_tools(_SUCCESS)
    tools.execute(
        "GITHUB_CREATE_ISSUE",
        {"title": "secret"},
        user_id="u1",
        connected_account_id="ca_secret",
    )

    span = one_span(exporter)
    assert_private(span, "secret", "u1", "ca_secret")


def test_execute_records_resolved_version(exporter):
    client = mock_http_client()
    tools = Tools(
        client=client,
        provider=Mock(),
        toolkit_versions={"github": "20251201_01"},
    )
    tool = _ToolFactory().create_mock_tool("GITHUB_CREATE_ISSUE", "github")
    tools._tool_schemas["GITHUB_CREATE_ISSUE"] = tool
    response = Mock()
    response.model_dump.return_value = {
        "data": {"title": "secret-result"},
        "error": None,
        "successful": True,
    }
    client.tools.execute.return_value = response

    result = tools.execute(
        "GITHUB_CREATE_ISSUE",
        {"title": "secret"},
        user_id="u1",
        connected_account_id="ca_secret",
    )

    assert result["successful"] is True
    span = one_span(exporter)
    attributes = span_attributes(span)
    assert attributes["composio.tool.version"] == "20251201_01"
    assert attributes["composio.toolkit.slug"] == "github"
    assert span.status.status_code == StatusCode.UNSET
    assert_private(span, "secret", "secret-result", "u1", "ca_secret")


def test_span_wraps_modifiers_and_execution(exporter):
    tools = make_tools(_SUCCESS)
    seen = []

    def executing(*args, **kwargs):
        current = trace.get_current_span()
        seen.append(("execute", current.is_recording(), current.name))
        return _SUCCESS

    tools._execute_tool.side_effect = executing

    @before_execute
    def before(tool, toolkit, params):
        current = trace.get_current_span()
        seen.append(("before", current.is_recording(), current.name))
        return params

    @after_execute
    def after(tool, toolkit, response):
        current = trace.get_current_span()
        seen.append(("after", current.is_recording(), current.name))
        return response

    tools.execute(
        "GITHUB_CREATE_ISSUE",
        {"title": "secret"},
        modifiers=[before, after],
    )

    assert seen == [
        ("before", True, "execute_tool GITHUB_CREATE_ISSUE"),
        ("execute", True, "execute_tool GITHUB_CREATE_ISSUE"),
        ("after", True, "execute_tool GITHUB_CREATE_ISSUE"),
    ]


def test_no_tracer_is_a_no_op():
    with patch.object(_otel, "_tracer", None):
        tools = make_tools(_SUCCESS)
        result = tools.execute(
            "GITHUB_CREATE_ISSUE",
            {"title": "secret"},
            user_id="u1",
        )
    assert result["successful"] is True


def test_session_execute_creates_span(exporter):
    session = make_session()
    response = Mock(result_type="completed", error=None)
    with patch(
        "composio.core.models.tool_router_session.require_execute_result",
        return_value=response,
    ):
        result = session.execute(
            "GITHUB_CREATE_ISSUE",
            arguments={"title": "secret"},
            account="ca_secret",
        )

    assert result is response
    span = one_span(exporter)
    attributes = span_attributes(span)
    assert span.name == "execute_tool GITHUB_CREATE_ISSUE"
    assert attributes["gen_ai.operation.name"] == "execute_tool"
    assert attributes["gen_ai.tool.name"] == "GITHUB_CREATE_ISSUE"
    assert "composio.toolkit.slug" not in attributes
    assert span.status.status_code == StatusCode.UNSET
    assert "error.type" not in attributes
    assert_private(span, "secret", "u1", "ca_secret")


@pytest.mark.parametrize("result_type", ["failed", "blocked"])
def test_session_failed_result_marks_span_error(exporter, result_type):
    session = make_session()
    response = Mock(result_type=result_type, error="401")
    with patch(
        "composio.core.models.tool_router_session.require_execute_result",
        return_value=response,
    ):
        session.execute("GITHUB_CREATE_ISSUE", arguments={"title": "secret"})

    span = one_span(exporter)
    attributes = span_attributes(span)
    assert span.status.status_code == StatusCode.ERROR
    assert attributes["error.type"] == "tool_error"
    assert span.status.description == "tool execution failed"
    assert list(span.events) == []
    assert_private(span, "secret", "u1", "401")


def test_session_no_tracer_is_a_no_op():
    with patch.object(_otel, "_tracer", None):
        session = make_session()
        response = Mock(result_type="completed", error=None)
        with patch(
            "composio.core.models.tool_router_session.require_execute_result",
            return_value=response,
        ):
            result = session.execute("GITHUB_CREATE_ISSUE", arguments={})
    assert result is response
