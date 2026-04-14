"""
Streamlit UI for the Indian Stock Market Analysis Agent.

Run with: streamlit run src/app.py
"""

import json
import os
import sys
import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()

st.set_page_config(
    page_title="IndiaStockAI — Deep Market Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

AGENT_META = {
    "screener":    {"icon": "🔍", "label": "Stock Screener"},
    "dcf":         {"icon": "💰", "label": "DCF Valuation"},
    "risk":        {"icon": "⚠️", "label": "Risk Analysis"},
    "earnings":    {"icon": "📊", "label": "Earnings Breakdown"},
    "portfolio":   {"icon": "🗂️", "label": "Portfolio Builder"},
    "technical":   {"icon": "📉", "label": "Technical Analysis"},
    "dividend":    {"icon": "💸", "label": "Dividend Strategy"},
    "competitive": {"icon": "🏆", "label": "Competitive Advantage"},
}

EXAMPLE_QUERIES = [
    "Full analysis of RELIANCE — valuation, risk, and technicals",
    "Screen for IT stocks with P/E below 25 and ROE above 15%",
    "DCF valuation for TCS assuming 12% growth for 3 years",
    "Risk analysis for HDFCBANK — beta, VaR and Sharpe ratio",
    "Build a diversified portfolio with ₹5 lakh across 5 stocks",
    "Dividend strategy: find high-yield consistent dividend payers",
    "Technical analysis for INFY — buy or sell signal?",
    "Competitive moat analysis for HINDUUNILVR vs peers",
    "Earnings breakdown for ICICIBANK last 4 quarters",
]


def _nse(t: str) -> str:
    t = t.upper().strip()
    return t if t.endswith((".NS", ".BO")) else t + ".NS"


@st.cache_data(ttl=300)
def fetch_quick_chart(ticker: str, period: str = "6mo"):
    try:
        df = yf.Ticker(_nse(ticker)).history(period=period)
        if df.empty:
            return None
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        return None


def render_price_chart(ticker: str):
    df = fetch_quick_chart(ticker)
    if df is None:
        st.warning("Could not load price data.")
        return

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name=ticker
    ))

    sma20 = df["Close"].rolling(20).mean()
    sma50 = df["Close"].rolling(50).mean()
    fig.add_trace(go.Scatter(x=df.index, y=sma20, name="SMA 20", line=dict(color="#f59e0b", width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=sma50, name="SMA 50", line=dict(color="#6366f1", width=1.5)))

    fig.update_layout(
        title=f"{ticker} — 6 Month Price Chart",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=400,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_agent_result(agent: str, result: dict):
    meta = AGENT_META.get(agent, {"icon": "🤖", "label": agent.title()})
    with st.expander(f"{meta['icon']} {meta['label']}", expanded=True):
        if not isinstance(result, dict):
            st.markdown(str(result))
            return
        # Model returned rich markdown prose — render it directly
        if "markdown" in result:
            st.markdown(result["markdown"])
            return
        if "error" in result and not any(k for k in result if k != "error"):
            st.error(f"Agent error: {result['error']}")
            return

        if agent == "risk":
            cols = st.columns(4)
            metrics = [
                ("Beta", result.get("beta")),
                ("Sharpe Ratio", result.get("sharpe_ratio")),
                ("Ann. Volatility", f"{result.get('annualized_volatility_pct', 'N/A')}%"),
                ("Max Drawdown", f"{result.get('max_drawdown_pct', 'N/A')}%"),
            ]
            for col, (label, val) in zip(cols, metrics):
                col.metric(label, val if val is not None else "N/A")

            risk_label = result.get("risk_label", "")
            risk_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(risk_label, "⚪")
            st.info(f"**Risk Level:** {risk_color} {result.get('risk_score', 'N/A')} — {risk_label}")
            if result.get("var_interpretation"):
                st.write(f"📋 {result['var_interpretation']}")

        elif agent == "dcf":
            col1, col2, col3 = st.columns(3)
            col1.metric("Intrinsic Value", f"₹{result.get('intrinsic_value_per_share', 'N/A')}")
            col2.metric("Market Price", f"₹{result.get('current_market_price', 'N/A')}")
            mos = result.get("margin_of_safety_pct")
            col3.metric("Margin of Safety", f"{mos}%" if mos is not None else "N/A",
                       delta="Undervalued" if (mos or 0) > 0 else "Overvalued",
                       delta_color="normal" if (mos or 0) > 0 else "inverse")
            st.success(f"**Verdict:** {result.get('verdict', 'N/A')}")

            if result.get("projected_fcf"):
                fcf_data = result["projected_fcf"]
                df_fcf = pd.DataFrame(fcf_data)
                fig = px.bar(df_fcf, x="year", y=["fcf_cr", "pv_cr"],
                            barmode="group", title="Projected FCF vs Present Value (₹ Cr)",
                            template="plotly_dark", color_discrete_map={"fcf_cr": "#22c55e", "pv_cr": "#3b82f6"})
                fig.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig, use_container_width=True)

        elif agent == "technical":
            indicators = result.get("indicators", {})
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Current Price", f"₹{result.get('current_price', 'N/A')}")
            col2.metric("RSI (14)", round(indicators.get("rsi_14") or 0, 1))
            col3.metric("Support", f"₹{result.get('support_level', 'N/A')}")
            col4.metric("Resistance", f"₹{result.get('resistance_level', 'N/A')}")

            overall = result.get("overall_signal", "NEUTRAL")
            signal_color = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "🟡"}.get(overall, "⚪")
            st.info(f"**Overall Signal:** {signal_color} {overall} "
                   f"({result.get('bullish_signals', 0):.1f} bullish vs {result.get('bearish_signals', 0):.1f} bearish signals)")

            if result.get("signals"):
                sig_df = pd.DataFrame(result["signals"])
                sig_df["color"] = sig_df["signal"].map({"BUY": "🟢", "SELL": "🔴", "NEUTRAL": "🟡"})
                for _, row in sig_df.iterrows():
                    st.write(f"{row['color']} **{row['indicator']}**: {row['reason']}")

        elif agent == "dividend":
            col1, col2, col3 = st.columns(3)
            col1.metric("Trailing Yield", f"{result.get('trailing_dividend_yield_pct', 0):.2f}%")
            col2.metric("Years Paying", result.get("years_of_consistent_dividend", "N/A"))
            col3.metric("Consistency", result.get("consistency_score", "N/A"))

            if result.get("annual_dividends"):
                ann = result["annual_dividends"]
                fig = px.bar(x=list(ann.keys()), y=list(ann.values()),
                            title="Annual Dividend (₹)", template="plotly_dark",
                            labels={"x": "Year", "y": "Dividend (₹)"}, color_discrete_sequence=["#22c55e"])
                fig.update_layout(height=280, margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig, use_container_width=True)

        elif agent == "portfolio":
            if result.get("optimal_weights"):
                weights = result["optimal_weights"]
                col1, col2, col3 = st.columns(3)
                col1.metric("Expected Return", f"{result.get('expected_annual_return_pct', 0):.1f}%")
                col2.metric("Expected Volatility", f"{result.get('expected_annual_volatility_pct', 0):.1f}%")
                col3.metric("Sharpe Ratio", round(result.get("sharpe_ratio", 0), 2))

                df_w = pd.DataFrame(weights)
                if not df_w.empty:
                    fig = px.pie(df_w, names="ticker", values="weight_pct",
                                title="Optimal Portfolio Allocation",
                                template="plotly_dark", hole=0.4)
                    fig.update_layout(height=320, margin=dict(l=0, r=0, t=40, b=0))
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(df_w[["ticker", "weight_pct", "allocation_inr", "shares", "price"]].rename(columns={
                        "ticker": "Ticker", "weight_pct": "Weight %",
                        "allocation_inr": "Allocation (₹)", "shares": "Shares", "price": "Price (₹)"
                    }), hide_index=True)

        elif agent == "screener":
            results = result.get("results", [])
            if results:
                st.success(f"**{len(results)} stocks** passed screening out of {result.get('stocks_screened', '?')}")
                df_screen = pd.DataFrame(results)
                display_cols = [c for c in ["ticker", "name", "sector", "pe", "roe_pct", "market_cap_cr", "dividend_yield_pct"] if c in df_screen.columns]
                st.dataframe(df_screen[display_cols].rename(columns={
                    "ticker": "Ticker", "name": "Company", "sector": "Sector",
                    "pe": "P/E", "roe_pct": "ROE %", "market_cap_cr": "Mkt Cap (₹Cr)",
                    "dividend_yield_pct": "Div Yield %"
                }), hide_index=True)
            else:
                st.warning("No stocks passed the screening criteria.")

        elif agent == "competitive":
            peers = result.get("peers", [])
            if peers:
                df_peers = pd.DataFrame(peers)
                cols_to_show = [c for c in ["ticker", "name", "pe", "pb", "roe_pct", "market_cap_cr", "profit_margin_pct"] if c in df_peers.columns]
                st.dataframe(df_peers[cols_to_show].rename(columns={
                    "ticker": "Ticker", "name": "Company", "pe": "P/E", "pb": "P/B",
                    "roe_pct": "ROE %", "market_cap_cr": "Mkt Cap (₹Cr)", "profit_margin_pct": "Net Margin %"
                }), hide_index=True)

        else:
            st.json(result)


def render_final_report(report: dict):
    if not report or "error" in report:
        return

    st.markdown("---")
    st.subheader("📋 Investment Research Report")

    rec = report.get("investment_recommendation", "N/A")
    conf = report.get("confidence_level", "")
    rec_color = {"BUY": "🟢", "HOLD": "🟡", "SELL": "🔴", "AVOID": "🔴"}.get(rec, "⚪")

    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown(f"### {rec_color} {rec}")
        st.caption(f"Confidence: {conf}")

    with col2:
        if report.get("executive_summary"):
            st.write(report["executive_summary"])

    col_pos, col_neg = st.columns(2)
    with col_pos:
        st.markdown("**✅ Bull Case**")
        for p in report.get("key_positives", []):
            st.write(f"• {p}")
    with col_neg:
        st.markdown("**⚠️ Bear Case**")
        for r in report.get("key_risks", []):
            st.write(f"• {r}")

    targets = report.get("price_targets", {})
    if any(targets.values()):
        st.markdown("**🎯 Price Targets**")
        t_cols = st.columns(3)
        t_cols[0].metric("Bear Case", f"₹{targets.get('bear_case') or 'N/A'}")
        t_cols[1].metric("Base Case", f"₹{targets.get('base_case') or 'N/A'}")
        t_cols[2].metric("Bull Case", f"₹{targets.get('bull_case') or 'N/A'}")

    if report.get("contradictions"):
        st.warning(f"**⚡ Analyst Note:** {report['contradictions']}")

    if report.get("action_items"):
        st.markdown("**📌 Action Items**")
        for item in report["action_items"]:
            st.write(f"→ {item}")

    st.caption(f"⚠️ {report.get('disclaimer', 'Not investment advice.')}")


def main():
    with st.sidebar:
        provider = os.environ.get("LLM_PROVIDER", "anthropic").title()
        model_labels = {
            "Groq": os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
            "Ollama": os.environ.get("OLLAMA_MODEL", "llama3.1"),
            "Anthropic": os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
        }
        active_model = model_labels.get(provider, provider)
        st.info(f"**Provider:** {provider}  \n**Model:** `{active_model}`")
        st.markdown("## 🇮🇳 IndiaStockAI")
        st.caption("Deep Agentic Workflow for NSE/BSE Analysis")

        st.markdown("---")
        st.markdown("### 🤖 Active Agents")
        for agent, meta in AGENT_META.items():
            st.write(f"{meta['icon']} {meta['label']}")

        st.markdown("---")
        st.markdown("### ⚙️ Architecture")
        st.markdown("""
**LangGraph** multi-agent orchestration
**LangChain** tool-calling agents
**Claude Sonnet** reasoning backbone
**yfinance** live NSE/BSE data
**Markowitz** portfolio optimization
""")
        st.markdown("---")
        st.caption("Data from Yahoo Finance (NSE: .NS). For educational purposes only.")

    st.title("📈 Indian Stock Market — Deep Analysis Agent")
    st.caption("Multi-agent AI system powered by Claude + LangGraph | 8 Specialized Agents | Live NSE/BSE Data")

    tab_analyze, tab_chart, tab_about = st.tabs(["🔬 Analyze", "📊 Quick Chart", "ℹ️ About"])

    with tab_analyze:
        st.markdown("### Ask anything about Indian stocks")

        col_input, col_btn = st.columns([4, 1])
        with col_input:
            query = st.text_input(
                "Query",
                placeholder="e.g. Full analysis of RELIANCE — valuation, risk, and technicals",
                label_visibility="collapsed",
            )
        with col_btn:
            run_clicked = st.button("🚀 Analyze", type="primary", use_container_width=True)

        st.markdown("**Try these examples:**")
        cols = st.columns(3)
        for i, example in enumerate(EXAMPLE_QUERIES):
            if cols[i % 3].button(example[:55] + "…" if len(example) > 55 else example, key=f"ex_{i}"):
                query = example
                run_clicked = True

        if run_clicked and query:
            provider = os.environ.get("LLM_PROVIDER", "anthropic").lower()
            key_map = {"groq": "GROQ_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
            required_key = key_map.get(provider)
            if required_key and not os.environ.get(required_key):
                st.error(f"🔑 **{required_key} not found.** Please add it to your `.env` file.")
                st.code(f"LLM_PROVIDER={provider}\n{required_key}=your_key_here", language="bash")
                st.stop()

            from src.agents.orchestrator import run_analysis, route_query, extract_ticker

            ticker = extract_ticker(query)
            agents_to_run = route_query(query)

            st.markdown("---")
            if ticker:
                st.markdown(f"**Analysing:** `{ticker}` | **Agents:** {' · '.join([AGENT_META[a]['icon'] + ' ' + AGENT_META[a]['label'] for a in agents_to_run if a in AGENT_META])}")
            else:
                st.markdown(f"**Running:** {' · '.join([AGENT_META[a]['icon'] + ' ' + AGENT_META[a]['label'] for a in agents_to_run if a in AGENT_META])}")

            progress_bar = st.progress(0, text="Initializing agents...")
            status_area = st.empty()

            with st.spinner("Agents working..."):
                try:
                    start = time.time()
                    status_area.info("🔄 Orchestrator routing query to specialized agents...")
                    progress_bar.progress(10, text="Routing to agents...")

                    result = run_analysis(query)

                    progress_bar.progress(90, text="Synthesizing findings...")
                    status_area.info("🧠 Synthesizing agent findings into investment report...")

                    elapsed = round(time.time() - start, 1)
                    progress_bar.progress(100, text=f"Complete in {elapsed}s")
                    status_area.success(f"✅ Analysis complete in {elapsed}s — {len(result.get('agent_results', []))} agent results processed")

                    render_final_report(result.get("final_report", {}))

                    st.markdown("---")
                    st.subheader("🤖 Individual Agent Results")
                    agent_results = result.get("agent_results", [])
                    for msg in agent_results:
                        agent_name = msg["agent"]
                        if msg["status"] == "skipped":
                            continue
                        render_agent_result(agent_name, msg["result"])

                except EnvironmentError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    st.exception(e)

    with tab_chart:
        st.markdown("### Quick Price Chart")
        col_t, col_p = st.columns([2, 1])
        chart_ticker = col_t.text_input("NSE Ticker", value="RELIANCE", placeholder="e.g. TCS, INFY, HDFCBANK")
        chart_period = col_p.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)

        if chart_ticker:
            render_price_chart(chart_ticker.upper())

            info_data = None
            try:
                info_data = yf.Ticker(_nse(chart_ticker)).info
            except Exception:
                pass

            if info_data:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Current Price", f"₹{info_data.get('currentPrice', 'N/A')}")
                c2.metric("P/E Ratio", round(info_data.get('trailingPE') or 0, 1))
                c3.metric("52W High", f"₹{info_data.get('fiftyTwoWeekHigh', 'N/A')}")
                c4.metric("52W Low", f"₹{info_data.get('fiftyTwoWeekLow', 'N/A')}")

    with tab_about:
        st.markdown("""
## About IndiaStockAI

IndiaStockAI is a **deep agentic AI system** that orchestrates 8 specialized research agents
to deliver institutional-quality equity research on Indian stocks.

### Architecture

```
User Query
    │
    ▼
┌─────────────────────┐
│   LangGraph         │
│   Orchestrator      │  ← Supervisor routing (Domain 1)
│   (StateGraph)      │
└─────────┬───────────┘
          │  Routes to relevant agents
    ┌─────┴──────────────────────────────┐
    │                                    │
    ▼         ▼         ▼         ▼      ▼
┌───────┐ ┌─────┐ ┌──────┐ ┌───────┐ ┌──────┐
│Screener│ │ DCF │ │ Risk │ │Earnings│ │Portf.│
└───────┘ └─────┘ └──────┘ └───────┘ └──────┘
    ▼         ▼         ▼         ▼      ▼
┌───────┐ ┌──────────┐ ┌──────────┐
│ Tech. │ │ Dividend │ │Competitive│
└───────┘ └──────────┘ └──────────┘
          │
          ▼
    ┌──────────┐
    │ Synthesis│  ← Progressive summarization (Domain 5)
    │  Node    │
    └──────────┘
          │
          ▼
    Final Report
```

### Claude Certified Architect Concepts Applied

| Domain | Concept | Where Used |
|--------|---------|------------|
| Domain 1 | Multi-agent orchestration | LangGraph StateGraph with supervisor |
| Domain 1 | Task decomposition | Query → agent routing → synthesis |
| Domain 2 | Tool descriptions | All 10+ custom yfinance/calc tools |
| Domain 2 | Structured tool returns | JSON-only tool outputs |
| Domain 3 | CLAUDE.md hierarchy | Root CLAUDE.md with dev conventions |
| Domain 4 | Prompt engineering | Per-agent system prompts with explicit criteria |
| Domain 4 | Structured output | JSON schema enforcement in all prompts |
| Domain 4 | Validation-retry loops | `run_agent_with_retry()` in base.py |
| Domain 5 | Progressive summarization | Synthesis node aggregates agent outputs |
| Domain 5 | Error propagation | Status fields prevent graph crashes |
| Domain 5 | Context management | Each agent gets focused, minimal context |

### Data Sources
- **Yahoo Finance** (via yfinance) — NSE/BSE OHLCV, fundamentals, dividends
- **Nifty 50** (^NSEI) — Benchmark for beta/alpha calculations

### Disclaimer
This tool is for **educational and research purposes only**.
Not SEBI-registered investment advice.
""")


if __name__ == "__main__":
    main()
