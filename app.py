"""Gradio entry point: serves the investment research tools over MCP.

Runs locally (`python app.py`) or as a Hugging Face Space. Every tool reads live
sources (SEC EDGAR, Yahoo Finance); a Space can additionally persist the screening
table to a private dataset (see agent/hf_runtime.py).
"""
from __future__ import annotations

import os
from typing import Any

# This must be set before importing agent.config through agent.mcp_server.
if os.getenv("SPACE_ID"):
    # A Space's app directory is not the place for caches; /tmp is disposable storage.
    os.environ.setdefault("INVESTMENT_AGENT_DATA_DIR", "/tmp/investment_agent/data")

import gradio as gr  # noqa: E402

from agent.mcp_server import dispatch_tool, tool_descriptors  # noqa: E402


def _type_expr(schema: dict[str, Any]) -> str:
    kind = schema.get("type")
    return {
        "string": "str",
        "integer": "int",
        "number": "float",
        "boolean": "bool",
        "array": "list",
        "object": "dict",
    }.get(kind, "object")


def _docstring(description: str, properties: dict[str, dict[str, Any]]) -> str:
    lines = [description.strip(), "", "Args:"]
    for name, schema in properties.items():
        detail = schema.get("description") or f"Value for {name}."
        if schema.get("enum"):
            detail += f" Allowed values: {schema['enum']}."
        lines.append(f"    {name}: {detail}")
    return "\n".join(lines)


async def _invoke(tool_name: str, payload: dict[str, Any]) -> str:
    arguments = {key: value for key, value in payload.items() if value is not None}
    blocks = await dispatch_tool(tool_name, arguments)
    text = [block.text for block in blocks if getattr(block, "type", None) == "text"]
    images = [block for block in blocks if getattr(block, "type", None) == "image"]
    if images:
        text.append(
            f"[{len(images)} chart image(s) were generated; the numeric evidence is "
            "included above.]"
        )
    return "\n\n".join(text) if text else "(no output)"


def _make_tool(descriptor):
    schema = descriptor.inputSchema or {"type": "object", "properties": {}}
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    ordered = [name for name in properties if name in required]
    ordered.extend(name for name in properties if name not in required)

    parameters = []
    for name in ordered:
        prop = properties[name]
        annotation = _type_expr(prop)
        if name in required:
            parameters.append(f"{name}: {annotation}")
        else:
            parameters.append(f"{name}: {annotation} = {prop.get('default', None)!r}")

    source = (
        f"async def {descriptor.name}({', '.join(parameters)}) -> str:\n"
        f"    return await _invoke({descriptor.name!r}, locals())\n"
    )
    namespace = {"_invoke": _invoke}
    exec(source, namespace)
    fn = namespace[descriptor.name]
    fn.__doc__ = _docstring(descriptor.description or descriptor.name, properties)
    return fn


_TOOLS = [_make_tool(descriptor) for descriptor in tool_descriptors()]


def _install_schema_passthrough() -> bool:
    """Restore `required` and `enum` to the MCP tool schemas.

    Gradio builds the MCP inputSchema from its own API info
    (gradio/mcp.py::GradioMCPServer.get_input_schema), which carries neither a
    `required` array nor enum constraints — so every tool arrived at the client
    looking optional and unconstrained, and an allowed-values list survived only
    as prose in the docstring. The authoritative schemas are the descriptors
    right here, so merge the dropped keys back in.

    Constraints are copied only onto a matching base type, and the whole thing is
    guarded: if a Gradio upgrade moves this method the app still starts, just
    with the plainer schema.
    """
    from gradio import mcp as gradio_mcp

    source = {d.name: (d.inputSchema or {}) for d in tool_descriptors()}
    original = gradio_mcp.GradioMCPServer.get_input_schema
    copy_onto = {"enum": {"string", "integer", "number"},
                 "items": {"array"},
                 "additionalProperties": {"object"}}

    def get_input_schema(self, tool_name, parameters=None):
        schema, filedata = original(self, tool_name, parameters)
        try:
            prefix = getattr(self, "tool_prefix", "") or ""
            key = (tool_name[len(prefix):]
                   if prefix and tool_name.startswith(prefix) else tool_name)
            spec = source.get(key) or {}
            properties = schema.get("properties") or {}
            for name, want in (spec.get("properties") or {}).items():
                have = properties.get(name)
                if not isinstance(have, dict):
                    continue
                # An optional parameter arrives as {"oneOf": [{"type": "null"},
                # {"type": "array", ...}]} with no type of its own, so the
                # constraints belong on the non-null branch, not the wrapper.
                targets = [have]
                for branch_key in ("oneOf", "anyOf"):
                    for branch in have.get(branch_key) or []:
                        if isinstance(branch, dict) and branch.get("type") != "null":
                            targets.append(branch)
                for target in targets:
                    for field, kinds in copy_onto.items():
                        # Gradio emits `items: {}` for a bare `list` annotation, so
                        # an empty value counts as absent — otherwise the element
                        # schema (and any enum inside it) never makes it through.
                        if (field in want and not target.get(field)
                                and target.get("type") in kinds):
                            target[field] = want[field]
            required = [n for n in (spec.get("required") or []) if n in properties]
            if required:
                schema["required"] = required
        except Exception as exc:  # never fail a tools/list over schema polish
            print(f"[mcp] schema merge skipped for {tool_name}: "
                  f"{type(exc).__name__}: {exc}", flush=True)
        return schema, filedata

    gradio_mcp.GradioMCPServer.get_input_schema = get_input_schema
    return True


try:
    _SCHEMA_PASSTHROUGH = _install_schema_passthrough()
except Exception as _exc:
    _SCHEMA_PASSTHROUGH = False
    print(f"[mcp] schema passthrough unavailable: {type(_exc).__name__}: {_exc}",
          flush=True)


def _endpoint() -> str:
    host = os.getenv("SPACE_HOST")
    base = f"https://{host}" if host else "http://localhost:7860"
    return f"{base}/gradio_api/mcp/"


with gr.Blocks(title="Investment Agent") as demo:
    gr.Markdown(
        "## Investment Agent\n"
        "US equity research tools served over MCP: SEC financial statements and "
        "filings, insider trades, executive pay, a filings-based stock screener, "
        "prices, technicals and portfolio risk.\n\n"
        "### Connect it\n"
        "MCP endpoint (keep the trailing slash — without it the server answers a "
        "307 and some clients drop the request body):\n\n"
        f"```\n{_endpoint()}\n```\n\n"
        "Claude web and Claude Desktop: Settings -> Connectors -> Add custom "
        "connector, paste that URL, leave OAuth empty. Claude Code: "
        "`claude mcp add --transport http investment-agent <url>`. Clients without "
        "remote HTTP MCP can proxy with `npx mcp-remote <url>`.\n\n"
        f"Tools available: **{len(_TOOLS)}** — all read-only except "
        "`refresh_screening_data`, which rebuilds the screening table from the SEC's "
        "public archive (rate-limited)."
    )
    for tool in _TOOLS:
        gr.api(
            tool,
            api_name=tool.__name__,
            api_description=tool.__doc__,
            queue=False,
        )


if __name__ == "__main__":
    demo.launch(mcp_server=True, show_error=True)
