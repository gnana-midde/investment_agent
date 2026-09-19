"""Streamable-HTTP entrypoint for the MCP server, for container hosts such as Cloud Run.

The service uses stateless Streamable HTTP so any instance can handle any request.
Run it with an ASGI server, e.g. ``uvicorn agent.mcp_http:app --port 8080``.

Authentication is deliberately fail-closed: when ``MCP_AUTH_ISSUER`` is set,
every MCP request must carry a JWT for ``MCP_RESOURCE_URL`` with
``MCP_REQUIRED_SCOPE``. Without an issuer the endpoint itself is open, so the
platform in front of it (Cloud Run IAM, for example) must protect the service.
"""
from __future__ import annotations

import asyncio
import contextlib
import os
from typing import Any

import jwt
import mcp.types as mtypes
from mcp.server import Server
from mcp.server.auth.middleware.auth_context import AuthContextMiddleware
from mcp.server.auth.middleware.bearer_auth import BearerAuthBackend, RequireAuthMiddleware
from mcp.server.auth.provider import AccessToken
from mcp.server.auth.routes import build_resource_metadata_url, create_protected_resource_routes
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from pydantic import AnyHttpUrl
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.authentication import AuthenticationMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.types import Receive, Scope, Send

from .mcp_server import dispatch_tool, tool_descriptors


REQUIRED_SCOPE = os.getenv("MCP_REQUIRED_SCOPE", "investment:read")
AUTH_ISSUER = os.getenv("MCP_AUTH_ISSUER", "").rstrip("/")
RESOURCE_URL = os.getenv("MCP_RESOURCE_URL", "").rstrip("/")


cloud_server = Server("investment-agent-http")


@cloud_server.list_tools()
async def list_tools() -> list[mtypes.Tool]:
    return tool_descriptors(oauth_scope=REQUIRED_SCOPE if AUTH_ISSUER else None)


@cloud_server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[mtypes.TextContent | mtypes.ImageContent]:
    return await dispatch_tool(name, arguments)


class JWTVerifier:
    """Verify Auth0/OIDC RS256 access tokens against the issuer's JWKS."""

    def __init__(self, issuer: str, audience: str, required_scope: str):
        if not issuer or not audience:
            raise ValueError("MCP_AUTH_ISSUER and MCP_RESOURCE_URL must be set together")
        self.issuer = issuer + "/"
        self.audience = audience
        self.required_scope = required_scope
        self.jwks = jwt.PyJWKClient(f"{issuer}/.well-known/jwks.json", cache_keys=True)

    def _decode(self, token: str) -> dict[str, Any]:
        key = self.jwks.get_signing_key_from_jwt(token).key
        return jwt.decode(
            token,
            key,
            algorithms=["RS256"],
            audience=self.audience,
            issuer=self.issuer,
            options={"require": ["exp", "iat", "iss", "sub", "aud"]},
        )

    async def verify_token(self, token: str) -> AccessToken | None:
        try:
            claims = await asyncio.to_thread(self._decode, token)
            raw_scope = claims.get("scope", "")
            scopes = raw_scope.split() if isinstance(raw_scope, str) else list(raw_scope or [])
            permissions = claims.get("permissions")
            if isinstance(permissions, list):
                scopes = sorted(set(scopes).union(str(item) for item in permissions))
            if self.required_scope not in scopes:
                return None
            return AccessToken(
                token=token,
                client_id=str(claims.get("azp") or claims.get("client_id") or claims["sub"]),
                scopes=scopes,
                expires_at=int(claims["exp"]),
                resource=self.audience,
                subject=str(claims["sub"]),
                claims=claims,
            )
        except Exception:
            return None


class MCPASGI:
    def __init__(self, manager: StreamableHTTPSessionManager):
        self.manager = manager

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        await self.manager.handle_request(scope, receive, send)


async def health(_: Request) -> JSONResponse:
    return JSONResponse(
        {
            "status": "ok",
            "service": "investment-agent-http",
            "transport": "streamable-http",
            "auth": "oauth2" if AUTH_ISSUER else "platform",
            "data": "sec-edgar, yahoo-finance",
        }
    )


async def home(_: Request) -> JSONResponse:
    return JSONResponse(
        {
            "name": "Investment Agent",
            "health": "/healthz",
            "mcp": "/mcp",
            "authentication": "OAuth 2.1" if AUTH_ISSUER else "platform (none at this layer)",
        }
    )


def create_app() -> Starlette:
    manager = StreamableHTTPSessionManager(
        app=cloud_server,
        json_response=True,
        stateless=True,
        max_request_body_size=4 * 1024 * 1024,
    )

    @contextlib.asynccontextmanager
    async def lifespan(_: Starlette):
        async with manager.run():
            yield

    mcp_app: Any = MCPASGI(manager)
    middleware: list[Middleware] = []
    routes = [
        Route("/", home, methods=["GET"]),
        Route("/healthz", health, methods=["GET"]),
    ]

    if AUTH_ISSUER:
        if not RESOURCE_URL:
            raise ValueError("MCP_RESOURCE_URL is required when MCP_AUTH_ISSUER is set")
        issuer = AnyHttpUrl(AUTH_ISSUER)
        resource = AnyHttpUrl(RESOURCE_URL)
        verifier = JWTVerifier(AUTH_ISSUER, RESOURCE_URL, REQUIRED_SCOPE)
        metadata_url = build_resource_metadata_url(resource)
        middleware = [
            Middleware(AuthenticationMiddleware, backend=BearerAuthBackend(verifier)),
            Middleware(AuthContextMiddleware),
        ]
        mcp_app = RequireAuthMiddleware(mcp_app, [REQUIRED_SCOPE], metadata_url)
        routes.extend(
            create_protected_resource_routes(
                resource_url=resource,
                authorization_servers=[issuer],
                scopes_supported=[REQUIRED_SCOPE],
                resource_name="Investment Agent tools",
            )
        )

    routes.append(Route("/mcp", endpoint=mcp_app))
    return Starlette(routes=routes, middleware=middleware, lifespan=lifespan)


app = create_app()
