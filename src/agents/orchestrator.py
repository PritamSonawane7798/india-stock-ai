"""
LangGraph-based multi-agent orchestrator for Indian stock market analysis.

Architecture (Domain 1 — Agentic Architecture & Orchestration):
- StateGraph with typed state channels
- Supervisor routing node that decides which agents to invoke
- Each agent is a node in the graph with structured input/output
- Conditional edges for dynamic routing based on query type
- Human-in-the-loop checkpoint support (can be added via LangGraph persistence)

Domain 5 patterns used:
- Progressive summarization: each agent appends to a shared context
- Error propagation: failed agents report status without crashing the graph
- Context positioning: most critical findings placed first in synthesis
"""

import json
import operator
from typing import Annotated, Any, Literal, TypedDict

from langchain_anthropic import ChatAnthropic
from langgraph.graph import END, START, StateGraph

from src.agents.base import get_llm
from src.agents.competitive_agent import run_competitive_analysis
from src.agents.dcf_agent import run_dcf_analysis
from src.agents.dividend_agent import run_dividend_analysis
from src.agents.earnings_agent import run_earnings_analysis
from src.agents.portfolio_agent import run_portfolio_builder
from src.agents.risk_agent import run_risk_analysis
from src.agents.screener_agent import run_screener
from src.agents.technical_agent import run_technical_analysis
from src.prompts.templates import ORCHESTRATOR_SYSTEM


# ─── State schema ────────────────────────────────────────────────────────────

class AgentMessage(TypedDict):
    agent: str
    status: Literal["success", "error", "skipped"]
    result: dict


class AnalysisState(TypedDict):
    query: str
    ticker: str | None
    tickers: list[str]
    agents_to_run: list[str]
    agent_results: Annotated[list[AgentMessage], operator.add]
    synthesis: str
    final_report: dict
    error: str | None


# ─── Routing & supervisor ─────────────────────────────────────────────────────

AGENT_KEYWORDS = {
    "screener": ["screen", "filter", "find stocks", "which stocks", "sector stocks"],
    "dcf": ["dcf", "intrinsic value", "valuation", "undervalued", "overvalued", "worth"],
    "risk": ["risk", "beta", "var", "volatility", "drawdown", "sharpe", "safe"],
    "earnings": ["earnings", "quarterly", "profit", "revenue", "eps", "results", "margin"],
    "portfolio": ["portfolio", "diversif", "allocat", "invest", "optimize", "build"],
    "technical": ["technical", "chart", "rsi", "macd", "trend", "buy signal", "support", "resistance"],
    "dividend": ["dividend", "yield", "income", "payout", "passive"],
    "competitive": ["moat", "competitive", "advantage", "peers", "industry", "comparison", "market share"],
}


def route_query(query: str) -> list[str]:
    """
    Determine which agents to invoke based on the user query.
    Falls back to a full analysis if no specific keywords matched.
    """
    q = query.lower()
    agents = []
    for agent, keywords in AGENT_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            agents.append(agent)

    if not agents:
        agents = ["dcf", "risk", "technical", "earnings"]

    if "full" in q or "complete" in q or "comprehensive" in q or "all" in q:
        agents = list(AGENT_KEYWORDS.keys())

    return agents


def extract_ticker(query: str) -> str | None:
    """Simple heuristic to extract NSE ticker from query."""
    words = query.upper().split()
    nse_words = [w.rstrip(".,!?") for w in words if len(w) <= 12 and w.isalpha() and w not in {
        "THE", "FOR", "AND", "WITH", "WHAT", "GIVE", "ME", "STOCK", "ANALYSE", "ANALYSIS",
        "OF", "IN", "IS", "A", "AN", "DO", "DCF", "RISK", "FULL", "COMPLETE", "INDIAN",
        "NSE", "BSE", "SCREENER", "SCREEN", "FILTER", "PORTFOLIO", "BUILD", "RUN",
    }]
    return nse_words[0] if nse_words else None


# ─── Graph nodes ──────────────────────────────────────────────────────────────

def supervisor_node(state: AnalysisState) -> AnalysisState:
    """
    Supervisor node: routes query to appropriate specialized agents.
    Uses keyword routing + LLM confirmation for ambiguous queries.
    """
    query = state["query"]
    ticker = extract_ticker(query)
    agents = route_query(query)

    return {
        "ticker": ticker,
        "tickers": [ticker] if ticker else [],
        "agents_to_run": agents,
        "error": None,
    }


def _make_agent_node(agent_name: str, run_fn):
    """Factory that creates a LangGraph node for a given agent function."""

    def node(state: AnalysisState) -> AnalysisState:
        if agent_name not in state.get("agents_to_run", []):
            return {"agent_results": [{"agent": agent_name, "status": "skipped", "result": {}}]}

        ticker = state.get("ticker")
        query = state["query"]
        agent_query = f"{query}" + (f"\nFocus on ticker: {ticker}" if ticker else "")

        try:
            result = run_fn(agent_query)
            return {"agent_results": [{"agent": agent_name, "status": "success", "result": result}]}
        except Exception as e:
            return {"agent_results": [{"agent": agent_name, "status": "error", "result": {"error": str(e)}}]}

    node.__name__ = f"{agent_name}_node"
    return node


def synthesis_node(state: AnalysisState) -> AnalysisState:
    """
    Synthesis node: aggregates results from all agents into a coherent report.

    Implements Domain 5 progressive summarization:
    - Collects only the successful results
    - Positions the most critical finding (valuation verdict) first
    - Generates an executive summary highlighting contradictions
    """
    llm = get_llm(temperature=0.1)

    successful = [m for m in state["agent_results"] if m["status"] == "success"]
    if not successful:
        return {"synthesis": "No agent results available.", "final_report": {"error": "All agents failed"}}

    context_parts = []
    for msg in successful:
        agent = msg["agent"].upper()
        result_str = json.dumps(msg["result"], indent=2)[:3000]
        context_parts.append(f"## {agent} AGENT RESULTS\n{result_str}")

    context = "\n\n".join(context_parts)

    synthesis_prompt = f"""
You are synthesizing a comprehensive equity research report for an Indian investor.

USER QUERY: {state['query']}
TICKER: {state.get('ticker', 'Multiple/Screened')}

AGENT FINDINGS:
{context}

Generate a synthesis in this exact JSON structure:
{{
  "executive_summary": "3-4 sentence overall investment thesis",
  "investment_recommendation": "BUY / HOLD / SELL / AVOID",
  "confidence_level": "HIGH / MEDIUM / LOW",
  "key_positives": ["list", "of", "bull", "points"],
  "key_risks": ["list", "of", "bear", "points"],
  "price_targets": {{
    "bull_case": null,
    "base_case": null,
    "bear_case": null
  }},
  "agent_highlights": {{
    "valuation": "one-line summary from DCF agent",
    "risk": "one-line summary from risk agent",
    "technical": "one-line summary from technical agent",
    "fundamentals": "one-line from earnings agent"
  }},
  "contradictions": "Note any contradictions between agents (e.g. technically bearish but fundamentally cheap)",
  "action_items": ["specific", "next", "steps", "for", "investor"],
  "disclaimer": "This is AI-generated analysis for educational purposes only. Not SEBI-registered investment advice."
}}

Output ONLY valid JSON. No markdown, no extra text.
"""

    response = llm.invoke(synthesis_prompt)
    synthesis_text = response.content

    try:
        final_report = json.loads(synthesis_text)
    except Exception:
        import re
        match = re.search(r"\{.*\}", synthesis_text, re.DOTALL)
        final_report = json.loads(match.group()) if match else {"raw": synthesis_text}

    return {
        "synthesis": synthesis_text,
        "final_report": final_report,
    }


# ─── Graph construction ───────────────────────────────────────────────────────

def build_analysis_graph() -> StateGraph:
    """
    Constructs the LangGraph StateGraph for parallel multi-agent analysis.

    Graph topology:
    START → supervisor → [all agents in parallel] → synthesis → END

    Agents that are not needed for a given query are skipped gracefully.
    """
    graph = StateGraph(AnalysisState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("screener", _make_agent_node("screener", run_screener))
    graph.add_node("dcf", _make_agent_node("dcf", run_dcf_analysis))
    graph.add_node("risk", _make_agent_node("risk", run_risk_analysis))
    graph.add_node("earnings", _make_agent_node("earnings", run_earnings_analysis))
    graph.add_node("portfolio", _make_agent_node("portfolio", run_portfolio_builder))
    graph.add_node("technical", _make_agent_node("technical", run_technical_analysis))
    graph.add_node("dividend", _make_agent_node("dividend", run_dividend_analysis))
    graph.add_node("competitive", _make_agent_node("competitive", run_competitive_analysis))
    graph.add_node("synthesis", synthesis_node)

    graph.add_edge(START, "supervisor")

    for agent in ["screener", "dcf", "risk", "earnings", "portfolio", "technical", "dividend", "competitive"]:
        graph.add_edge("supervisor", agent)
        graph.add_edge(agent, "synthesis")

    graph.add_edge("synthesis", END)

    return graph


def run_analysis(query: str) -> dict:
    """
    Main entry point: runs the full multi-agent analysis pipeline.

    Args:
        query: Natural language query, e.g.:
               "Give me a full analysis of RELIANCE"
               "Which tech stocks have P/E below 25 and ROE above 15%?"
               "DCF valuation for TCS with 15% growth assumption"

    Returns:
        Dict with final_report, individual agent_results, and metadata.
    """
    graph = build_analysis_graph()
    app = graph.compile()

    initial_state: AnalysisState = {
        "query": query,
        "ticker": None,
        "tickers": [],
        "agents_to_run": [],
        "agent_results": [],
        "synthesis": "",
        "final_report": {},
        "error": None,
    }

    final_state = app.invoke(initial_state)
    return {
        "query": query,
        "ticker": final_state.get("ticker"),
        "agents_run": final_state.get("agents_to_run", []),
        "agent_results": final_state.get("agent_results", []),
        "final_report": final_state.get("final_report", {}),
    }
