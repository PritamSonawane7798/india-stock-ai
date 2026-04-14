# 📈 IndiaStockAI — Deep Agentic Workflow for Indian Stock Market Analysis

**Multi-agent AI system** that orchestrates 8 specialized research agents to deliver institutional-quality equity research on NSE/BSE listed Indian stocks.

Powered by **Claude Sonnet** + **LangGraph** + **LangChain** | Live NSE/BSE data via yfinance

> **Portfolio Project** — Demonstrates applied [Claude Certified Architect](https://claudecertifications.com/claude-certified-architect/domains) concepts across all 5 exam domains.

---

## 🎯 What It Does

Ask questions in plain English. The orchestrator routes to the right agents automatically:

```
"Full analysis of RELIANCE"
→ routes to: DCF + Risk + Earnings + Technical + Competitive agents

"Screen for IT stocks with P/E below 25 and ROE above 15%"  
→ routes to: Screener agent

"Build a ₹5 lakh diversified portfolio"
→ routes to: Screener + Portfolio agents
```

## 🤖 The 8 Agents

| Agent | What it does |
|-------|-------------|
| 🔍 **Stock Screener** | Filters 30+ NSE large-caps by P/E, ROE, debt, dividend yield, sector |
| 💰 **DCF Valuation** | Intrinsic value with base/bull/bear scenarios, margin of safety |
| ⚠️ **Risk Analysis** | Beta, Sharpe, VaR (95%), CVaR, max drawdown vs Nifty 50 |
| 📊 **Earnings Breakdown** | Quarterly EPS actuals vs estimates, margin trends, beat/miss history |
| 🗂️ **Portfolio Builder** | Markowitz optimization, optimal weights, expected risk-return |
| 📉 **Technical Analysis** | RSI, MACD, Bollinger Bands, EMA crossovers, support/resistance |
| 💸 **Dividend Strategy** | Yield, FCF coverage, dividend growth CAGR, consistency score |
| 🏆 **Competitive Advantage** | Economic moat, Porter's Five Forces, peer comparison |

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────┐
│  Supervisor Node    │  ─── LangGraph StateGraph (Domain 1)
│  (keyword routing)  │
└──────────┬──────────┘
           │  parallel dispatch
    ┌──────┴──────────────────────────────────┐
    ▼      ▼      ▼       ▼     ▼      ▼      ▼      ▼
 [Screen][DCF][Risk][Earnings][Portf][Tech][Div][Comp]
                          │
                          ▼
               ┌──────────────────┐
               │  Synthesis Node  │  ─── Progressive summarization (Domain 5)
               │  (Claude Sonnet) │
               └────────┬─────────┘
                        ▼
               Final Investment Report
```

## 🎓 Claude Certified Architect Concepts Applied

| Domain | Concept | Implementation |
|--------|---------|----------------|
| **Domain 1** | Multi-agent orchestration | LangGraph `StateGraph` with supervisor routing node |
| **Domain 1** | Agentic loops & task decomposition | Query → route → parallel agents → synthesis |
| **Domain 2** | Tool descriptions | Detailed docstrings guide Claude's tool selection |
| **Domain 2** | Structured tool returns | All tools return JSON — no free-text |
| **Domain 3** | CLAUDE.md hierarchy | Root `CLAUDE.md` with dev conventions |
| **Domain 4** | Explicit system prompts | Per-agent prompts with JSON schema requirements |
| **Domain 4** | Validation-retry loops | `run_agent_with_retry()` with error context injection |
| **Domain 5** | Progressive summarization | Synthesis node aggregates + summarizes agent outputs |
| **Domain 5** | Error propagation | Status fields (`success/error/skipped`) — no crashes |
| **Domain 5** | Context positioning | Critical findings (valuation verdict) placed first |

## 🚀 Setup

### Prerequisites
- Python 3.11+
- Anthropic API key ([console.anthropic.com](https://console.anthropic.com))

### Installation

```bash
git clone https://github.com/pritamsonawane/india-stock-ai.git
cd india-stock-ai

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
```

Edit `.env`:
```
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### Run

**Streamlit web app:**
```bash
streamlit run src/app.py
```
Opens at `http://localhost:8501`

**CLI demo:**
```bash
python examples/demo_analysis.py
```

## 📁 Project Structure

```
stock-analysis/
├── src/
│   ├── agents/
│   │   ├── base.py              # Shared agent setup, retry logic
│   │   ├── orchestrator.py      # LangGraph StateGraph orchestrator
│   │   ├── screener_agent.py
│   │   ├── dcf_agent.py
│   │   ├── risk_agent.py
│   │   ├── earnings_agent.py
│   │   ├── portfolio_agent.py
│   │   ├── technical_agent.py
│   │   ├── dividend_agent.py
│   │   └── competitive_agent.py
│   ├── tools/
│   │   ├── market_data.py       # yfinance wrappers: info, history, financials
│   │   ├── financial_calc.py    # DCF engine, risk metrics, portfolio optimizer
│   │   └── technical_indicators.py  # RSI, MACD, Bollinger Bands, etc.
│   ├── prompts/
│   │   └── templates.py         # Per-agent system prompts
│   └── app.py                   # Streamlit UI
├── docs/
│   └── index.html               # GitHub Pages landing page
├── examples/
│   └── demo_analysis.py         # CLI demo
├── CLAUDE.md                    # Claude Code configuration
├── .env.example
├── requirements.txt
└── README.md
```

## 🔑 Example Queries

```
# Full analysis
"Give me a comprehensive analysis of RELIANCE — valuation, risk, and technicals"

# Screening
"Screen for banking stocks with P/E below 20, ROE above 12%, and dividend yield above 1%"

# Valuation
"DCF valuation for TCS: assume 15% FCF growth for 3 years, then 8% for the next 7, 5% terminal"

# Risk
"What is the risk profile of HDFCBANK — beta, VaR, and Sharpe ratio over 2 years?"

# Portfolio
"Build an optimized portfolio of 5 stocks with ₹2 lakh investment — maximize Sharpe ratio"

# Dividends
"Find the best dividend stocks in NSE — consistent payers with sustainable yield"

# Technical
"Is INFY technically a buy right now? Check RSI, MACD, and EMA crossovers"

# Moat
"Compare HINDUNILVR's competitive moat vs ITC and Nestle India"
```

## 📊 Data Sources

- **Yahoo Finance** (yfinance) — NSE (`.NS`) / BSE (`.BO`) stocks
- **Nifty 50** (`^NSEI`) — Benchmark index for risk calculations
- All data is fetched live at query time — no stale cached data

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. It is not SEBI-registered investment advice. Always consult a qualified financial advisor before making investment decisions.

---

*Built by [Pritam Sonawane](https://github.com/pritamsonawane) | Applying Claude Certified Architect concepts to real-world agentic AI*
