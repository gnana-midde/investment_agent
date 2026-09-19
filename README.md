---
title: Investment Agent
emoji: 📊
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.20.0
app_file: app.py
pinned: false
---

# Investment Agent

Research tools for US-listed equities, served over the Model Context Protocol (MCP)
so Claude or any other MCP client can call them. It also includes a command-line
research agent built on the Claude Agent SDK.

The tools use public sources only. Fundamentals, filings, insider trades and
executive pay come from **SEC EDGAR**. Prices, quote-derived ratios and index levels
come from **Yahoo Finance** (through `yfinance`). No API keys are needed.

## Tools

| Tool | What it returns | Source |
| --- | --- | --- |
| `resolve_company` | Ticker and CIK for a company name | SEC |
| `company_overview` | Business description, sector, size, SEC registration details | Yahoo, SEC |
| `financial_statements` | Income statement, balance sheet, cash flow (up to 12 years) and quarterly results | SEC XBRL |
| `valuation_summary` | Multiples, margins, leverage, 52-week range, analyst consensus | Yahoo |
| `technicals_momentum` | Moving averages, RSI, MACD, trailing returns | Yahoo |
| `price_history` | Daily, weekly or monthly OHLCV between dates | Yahoo |
| `price_analytics` | Drawdowns, volatility, volume trend, golden/death cross, relative strength | Yahoo |
| `portfolio_risk` | Correlation, volatility, Sharpe/Sortino, historical VaR/CVaR and beta for a set of holdings | Yahoo |
| `index_data` | S&P 500, Nasdaq and Dow levels | Yahoo |
| `macro_data` | 10-year Treasury yield, crude, gold, silver | Yahoo |
| `screen_stocks` | Screen of every SEC filer's latest annual figures, with a funnel and near misses | SEC bulk data |
| `refresh_screening_data` | Rebuilds the screening table from the SEC's nightly archive | SEC bulk data |
| `insider_trades` | Forms 3/4/5 transactions for a quarter; open-market trades kept separate from grants and exercises | SEC |
| `executive_pay` | The DEF 14A proxy filings that hold the compensation tables | SEC |
| `company_documents` | Recent 10-K, 10-Q, 8-K and proxy filings with URLs | SEC |
| `read_document` | Text of an SEC filing, or of a PDF | SEC, any PDF URL |

Everything is read-only except `refresh_screening_data`, which writes the screening
table to the local data directory.

## Quick start

Requires Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The SEC rejects requests that don't include a contact address, so set one before
using the SEC tools:

```bash
export SEC_USER_AGENT="InvestmentAgent/1.0 (you@example.com)"   # Windows: set SEC_USER_AGENT=...
```

Then run the tools in one of three ways.

**As a local MCP server (stdio)** for Claude Desktop, Claude Code or another client.
Add this to the client's MCP configuration, with your own paths filled in:

```json
{
  "mcpServers": {
    "investment-agent": {
      "command": "/path/to/investment_agent/.venv/bin/python",
      "args": ["-m", "agent.mcp_server"],
      "env": {
        "PYTHONPATH": "/path/to/investment_agent",
        "SEC_USER_AGENT": "InvestmentAgent/1.0 (you@example.com)"
      }
    }
  }
}
```

**As a web app with an HTTP MCP endpoint:**

```bash
python app.py
```

The MCP endpoint is `http://localhost:7860/gradio_api/mcp/`. Keep the trailing
slash, because some clients drop the request body on the redirect you get without it.

**As a command-line research agent** (needs the Claude Code CLI installed and signed in):

```bash
python -m agent.chat                       # interactive chat
python -m agent.chat "Compare AAPL and MSFT free cash flow"
```

## The screening table

`screen_stocks` reads a table built from the SEC's bulk Company Facts archive. The
archive is about 1.4 GB and parsing it takes a few minutes, so build the table once
before you screen:

```bash
python -m agent.sec_refresh              # downloads only if the SEC archive changed
python -m agent.sec_refresh --status     # when it was built and whether it's current
```

You can also have your MCP client call `refresh_screening_data(action="start")` and
check back with `action="status"`.

The table comes from filings and has no price feed. Company size is **public float**
(the market value of non-affiliate shares, reported on the 10-K cover), not market
cap. `pe_on_float` is public float divided by net income, so it understates the real
P/E of companies with large insider holdings.

## Configuration

| Variable | Required | Purpose |
| --- | --- | --- |
| `SEC_USER_AGENT` | for the SEC tools | A User-Agent that includes a contact address, e.g. `InvestmentAgent/1.0 (you@example.com)`. Without it sec.gov returns 403, and every SEC tool reports that it can't reach the SEC. |
| `INVESTMENT_AGENT_DATA_DIR` | no | Where caches and the screening table go. Defaults to `./data`. |
| `HF_DATA_REPO` | no | A private Hugging Face dataset (`your-username/your-dataset`) for keeping the screening table across restarts of a hosted server. Setting it turns the dataset backing on. |
| `HF_TOKEN` | with `HF_DATA_REPO` | Read access downloads the table; write access also uploads each refresh. |
| `HF_WRITE_TOKEN` | no | A separate write token, if you'd rather not give `HF_TOKEN` write access. |
| `HF_DATA_REVISION` | no | Pins the dataset revision. Defaults to `main`. |
| `REFRESH_TOKEN` | recommended on a public server | A passphrase that `refresh_screening_data(action="start")` requires, so strangers can't trigger the 1.4 GB download. |
| `REFRESH_MIN_INTERVAL_HOURS` | no | Minimum time between refreshes. Defaults to 20 hours; the SEC regenerates the archive nightly. |
| `INVESTMENT_AGENT_MODEL` | no | The model the command-line agent uses. |
| `CLAUDE_CLI_PATH` | no | Path to the Claude Code CLI, if it isn't on `PATH`. |

## Deployment

**Hugging Face Spaces.** The front matter at the top of this README configures a
Gradio Space, and `app.py` is its entry point. Set `SEC_USER_AGENT` as a Space
secret. To keep the screening table across restarts, also set `HF_DATA_REPO` and
`HF_TOKEN`, and set `REFRESH_TOKEN` because the endpoint is public. A free Space
sleeps when idle, so the first request after a pause can take about a minute.

**Any container host.** `agent/mcp_http.py` is a Streamable-HTTP MCP server with
optional OAuth 2.1 bearer-token checks:

```bash
uvicorn agent.mcp_http:app --host 0.0.0.0 --port 8080
```

Set `MCP_AUTH_ISSUER` and `MCP_RESOURCE_URL` (plus `MCP_REQUIRED_SCOPE`, which
defaults to `investment:read`) to require tokens. Without them the endpoint has no
authentication, so put it behind your platform's access control.

## Project layout

```
app.py                  Gradio app and HTTP MCP endpoint
agent/
  tools.py              MCP tool definitions
  mcp_server.py         stdio MCP server
  mcp_http.py           Streamable-HTTP MCP server
  chat.py               command-line research agent (Claude Agent SDK)
  sec.py                SEC EDGAR: tickers, XBRL statements, filings
  sec_refresh.py        builds the screening table from the SEC bulk archive
  screener.py           screening engine: filters, funnel, near misses
  insider.py            Forms 3/4/5 transactions and proxy lookup
  live.py               Yahoo Finance prices, quotes and PDF extraction
  data_access.py        loaders and price analytics the tools share
  portfolio_risk.py     portfolio risk statistics
  ethics.py             optional exclusion categories for screening
  hf_runtime.py         optional Hugging Face dataset persistence
  config.py             paths
```

## Disclaimer

This is a research tool. Its output is not investment advice, and figures from
third-party sources can be late or wrong. Check anything important against the
original filing.
