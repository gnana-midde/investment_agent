"""Standalone stdio MCP server exposing the investment research tools.

This lets the tools run natively inside an MCP client (Claude Desktop, Claude Code,
or any other) using the client's own authentication — no API key needed here.
Register it in the client's MCP config; the client spawns this over stdio and gets
every tool in ALL_TOOLS (tools.py).

Run standalone (for a smoke test):  python -m agent.mcp_server  (then it waits on stdio)
"""
from __future__ import annotations

import asyncio

import mcp.types as mtypes
from mcp.server import Server
from mcp.server.stdio import stdio_server

from .tools import ALL_TOOLS

_BY_NAME = {t.name: t for t in ALL_TOOLS}

server = Server("investment-agent")


def tool_descriptors(*, oauth_scope: str | None = None) -> list[mtypes.Tool]:
    """Return deterministic MCP descriptors for local or authenticated transports."""
    security = [{"type": "oauth2", "scopes": [oauth_scope]}] if oauth_scope else None
    return [
        mtypes.Tool(
            name=t.name,
            description=t.description,
            inputSchema=t.input_schema,
            **({"securitySchemes": security} if security else {}),
        )
        for t in sorted(ALL_TOOLS, key=lambda item: item.name)
    ]


@server.list_tools()
async def list_tools() -> list[mtypes.Tool]:
    # Sorted, not registration order: MCP 2026-07-28 asks servers to return tools
    # deterministically so clients can cache the list and an unchanged tool list keeps
    # hitting the model's prompt cache. Cheap, and correct under any revision.
    return tool_descriptors()


def _blocks(result: dict) -> list[mtypes.TextContent | mtypes.ImageContent]:
    out: list[mtypes.TextContent | mtypes.ImageContent] = []
    for b in result.get("content", []):
        if b.get("type") == "text":
            out.append(mtypes.TextContent(type="text", text=b.get("text", "")))
        elif b.get("type") == "image":
            out.append(mtypes.ImageContent(type="image", data=b["data"],
                                           mimeType=b.get("mimeType", "image/png")))
    return out


async def dispatch_tool(name: str, arguments: dict) -> list[mtypes.TextContent | mtypes.ImageContent]:
    tool = _BY_NAME.get(name)
    if tool is None:
        raise ValueError(f"unknown tool '{name}' (have: {', '.join(sorted(_BY_NAME))})")
    try:
        # On a hosted server backed by a dataset, fetch what the tool reads before the
        # synchronous loaders inspect the data directory. Run this blocking
        # network/file work off the event loop.
        from . import hf_runtime
        if hf_runtime.enabled():
            await asyncio.to_thread(hf_runtime.prepare, name, arguments or {})
        result = await tool.handler(arguments or {})
    except Exception as e:
        # A raising handler would otherwise surface as a transport-level failure with
        # no indication of which tool broke. Re-raise so the model is told what failed
        # and can choose another route.
        raise RuntimeError(f"{name} failed: {type(e).__name__}: {e}") from e

    # The tool layer signals failure with is_error, but returning content blocks alone
    # DISCARDS that flag — an error then reaches the model formatted exactly like a
    # successful answer, which is how a failed lookup turns into a confident wrong
    # number. Raising makes the SDK mark the result isError.
    if result.get("is_error"):
        text = " ".join(b.get("text", "") for b in result.get("content", []))
        raise RuntimeError(text or f"{name} returned an error")

    return _blocks(result) or [mtypes.TextContent(type="text", text="(no output)")]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[mtypes.TextContent | mtypes.ImageContent]:
    return await dispatch_tool(name, arguments)


async def _main():
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def main():
    asyncio.run(_main())


if __name__ == "__main__":
    main()
