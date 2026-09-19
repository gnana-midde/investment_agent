"""Investment research agent — Claude Agent SDK entry point.

Usage (from the project root):
  python -m agent.chat                      # interactive chat
  python -m agent.chat "your question"      # one-shot query
  python -m agent.chat --login              # one-time subscription login

Env overrides:
  INVESTMENT_AGENT_MODEL  model id to use (default: the Claude Code CLI's default model)
  CLAUDE_CLI_PATH         full path to the Claude Code CLI when it is not on PATH
"""
from __future__ import annotations

import asyncio
import glob
import json
import os
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

from claude_agent_sdk import (
    AssistantMessage, ClaudeAgentOptions, ClaudeSDKClient, ResultMessage,
    TextBlock, ThinkingBlock, ToolUseBlock, create_sdk_mcp_server,
)
from dotenv import load_dotenv

from .config import ROOT
from .tools import ALL_TOOLS

load_dotenv(ROOT / ".env")  # picks up ANTHROPIC_API_KEY / INVESTMENT_AGENT_MODEL if present


# --------------------------------------------------------------- CLI + auth --
def find_claude_cli() -> str | None:
    """Locate the Claude Code CLI the SDK should spawn.

    The SDK's own discovery only checks PATH / npm-global / ~/.claude/local. On a
    Windows desktop-app install the CLI lives in a version-pinned folder under
    %APPDATA%\\Claude\\claude-code\\<ver>\\claude.exe that the SDK doesn't know
    about — so we find the newest one ourselves. Override with CLAUDE_CLI_PATH."""
    override = os.environ.get("CLAUDE_CLI_PATH")
    if override and Path(override).exists():
        return override
    onpath = shutil.which("claude")
    if onpath:
        return onpath
    pats = []
    for base in (os.environ.get("APPDATA"), os.environ.get("LOCALAPPDATA")):
        if not base:
            continue
        pats.append(os.path.join(base, "Claude", "claude-code", "*", "claude.exe"))
        pats.append(os.path.join(base, "Packages", "Claude_*", "LocalCache",
                                 "Roaming", "Claude", "claude-code", "*", "claude.exe"))
    found = []
    for pat in pats:
        found.extend(glob.glob(pat))

    def _ver(p: str):  # sort by the numeric version folder, newest last
        part = Path(p).parent.name
        try:
            return tuple(int(x) for x in part.split("."))
        except ValueError:
            return (0,)
    return max(found, key=_ver) if found else None


def run_login() -> int:
    """Interactive one-time subscription login (no API key): runs the CLI's
    `setup-token` flow, which stores a long-lived OAuth token the SDK then uses."""
    cli = find_claude_cli()
    if not cli:
        print("Could not find claude.exe. Install Claude Code, or set CLAUDE_CLI_PATH "
              "to its full path, then re-run: python -m agent.chat --login")
        return 1
    print(f"Using CLI: {cli}\nStarting subscription login (a browser will open)...\n")
    return subprocess.call([cli, "setup-token"])

SERVER_KEY = "investment_agent"


def known_exclusion_categories() -> list[str]:
    from . import ethics
    return ethics.known_categories()


def system_prompt() -> str:
    return f"""You are an investment research agent for US-listed equities. Today is {date.today().isoformat()}.

# Your data (via the investment_agent tools)
- Fundamentals from SEC EDGAR XBRL: income statement, balance sheet and cash flow (up to 12
  fiscal years) and quarterly results, in USD millions.
- SEC filings: 10-K / 10-Q / 8-K / proxy filings listed with URLs and readable as text;
  insider transactions from Forms 3/4/5; executive-pay disclosure in DEF 14A proxies.
- A screening table over every SEC filer's latest annual figures: size, margins, returns,
  leverage and an earnings multiple on public float. It has no price feed.
- Live market data from Yahoo Finance: prices, technicals, price analytics, valuation
  multiples and analyst consensus, US index levels, the 10-year Treasury yield, commodities.
- Portfolio risk from actual daily returns: correlation, volatility, drawdown,
  Sharpe/Sortino, historical VaR/CVaR, beta.

# Tool discipline
- User gives a company NAME -> call resolve_company first to get the ticker.
- Company question with NO period specified -> do not guess a "current quarter" from
  today's calendar date (filings lag by weeks). Use the latest column of
  financial_statements(quarterly_results) as the anchor period, and SAY which period you
  used (e.g. "using the quarter ended 2026-06-30, the latest filed").
- Quantitative questions -> financial_statements / valuation_summary / technicals_momentum.
- Qualitative questions (strategy, risk factors, MD&A, guidance, segment detail) ->
  company_documents to find the right 10-K / 10-Q / 8-K, then read_document on it.
- Multi-company questions -> ONE screen_stocks call covers the whole universe in a single
  pass; never loop over companies one tool call at a time.
- "Current/latest/today" questions -> the live market tools, and WebSearch for news and
  events after the latest filing. State clearly which numbers come from filings (with the
  period), which from live quotes (with the as-of date), and which from the web.
- Historical "as of <date>" questions -> price_history with a date range, plus the filing
  for that period.
- Screening requests -> screen_stocks with the objective criteria the user gave; if criteria
  are vague, propose concrete thresholds, state them, then screen. Report the match count.
  Remember size there is public float, not market cap.
- Insider activity -> insider_trades per quarter. Only open-market buys and sells (codes P/S)
  are decisions; grants, option exercises and tax withholding are compensation mechanics.
- Deep research -> plan briefly, then combine filings (several periods), financial trends,
  valuation, technicals and the web. Synthesize with sections.

# Hard rules
1. NEVER give buy/sell/hold recommendations, price targets of your own, or personalized
   investment advice. You may screen, rank by user-chosen metrics, and present valuation
   frameworks — always as information, never as advice. If asked "should I buy X", explain
   you provide analysis only, then offer the relevant analysis.
2. Cite your sources inline: form type and period for filings (e.g. "FY2025 10-K",
   "10-Q for the quarter ended 2026-03-31"); as-of dates for market data.
3. Numbers you compute must come from tool outputs — never invent figures.
4. If data is missing for a company, say so and fall back to WebSearch when appropriate.
5. Analyst targets and recommendations from valuation_summary are third-party consensus —
   present them as such, never as your own view.

# Screen -> shortlist workflow
When the user wants to screen the universe down to a shortlist (e.g. "give me the top 20
to study further"):
1. Apply exclusions only for categories the user has explicitly confirmed, via
   screen_stocks(exclude_categories=[...]) (currently defined: {", ".join(known_exclusion_categories()) or "none"}).
   Never assume a category the user hasn't named, and relay the exclusion counts the tool
   reports so it's transparent which companies were dropped and why.
2. Apply the user's quantitative criteria in the SAME screen_stocks call — filters and
   exclusions compose in one pass.
3. Report the match count at each stage (universe -> post-exclusion -> post-filter ->
   top-N), then sort to the requested top-N (default 20).
4. For each shortlisted company, on request, build a compact profile card: business
   description (company_overview), financial trend (financial_statements), valuation
   (valuation_summary) and momentum (technicals_momentum) — enough for the user to decide
   where to study deeper. This is information for their own research, never a
   recommendation to buy.

# Style
- Lead with the answer, then supporting detail. Tables for screens/comparisons.
- Be precise with periods (fiscal year vs calendar year vs quarter).
- End substantive analyses with a one-line data-freshness note.
"""


def build_options() -> ClaudeAgentOptions:
    server = create_sdk_mcp_server(name=SERVER_KEY, version="1.0.0", tools=ALL_TOOLS)
    tool_names = [f"mcp__{SERVER_KEY}__{t.name}" for t in ALL_TOOLS]
    return ClaudeAgentOptions(
        mcp_servers={SERVER_KEY: server},
        allowed_tools=tool_names + ["WebSearch", "WebFetch"],
        system_prompt=system_prompt(),
        permission_mode="bypassPermissions",
        model=os.environ.get("INVESTMENT_AGENT_MODEL") or None,
        max_turns=60,
        cli_path=find_claude_cli(),          # desktop-app CLI (PATH-independent)
    )


# ------------------------------------------------------------------ output --
def _print_message(msg):
    if isinstance(msg, AssistantMessage):
        for block in msg.content:
            if isinstance(block, TextBlock):
                print(block.text, flush=True)
            elif isinstance(block, ToolUseBlock):
                arg = json.dumps(block.input, ensure_ascii=False, default=str)
                name = block.name.replace(f"mcp__{SERVER_KEY}__", "")
                print(f"  [tool] {name} {arg[:160]}", flush=True)
            elif isinstance(block, ThinkingBlock):
                pass
    elif isinstance(msg, ResultMessage):
        cost = f" | ${msg.total_cost_usd:.4f}" if msg.total_cost_usd else ""
        print(f"\n-- done in {msg.duration_ms/1000:.1f}s | {msg.num_turns} turns{cost} --",
              flush=True)


async def run_once(prompt: str):
    async with ClaudeSDKClient(options=build_options()) as client:
        await client.query(prompt)
        async for msg in client.receive_response():
            _print_message(msg)


async def repl():
    print("Investment Agent - US equities research")
    print("Data: SEC filings | financials | insider trades | screening | prices | portfolio risk")
    print("Type your question ('exit' to quit).\n")
    async with ClaudeSDKClient(options=build_options()) as client:
        while True:
            try:
                q = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                continue
            if q.lower() in ("exit", "quit", "/exit", "/quit"):
                break
            await client.query(q)
            async for msg in client.receive_response():
                _print_message(msg)
            print()
    print("bye.")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = sys.argv[1:]
    if args and args[0] in ("--login", "login"):
        sys.exit(run_login())
    if args:
        asyncio.run(run_once(" ".join(args)))
    else:
        asyncio.run(repl())


if __name__ == "__main__":
    main()
