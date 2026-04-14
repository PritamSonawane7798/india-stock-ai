from src.agents.base import build_agent_executor, run_agent_with_retry
from src.prompts.templates import COMPETITIVE_SYSTEM
from src.tools.market_data import get_peers, get_stock_info, get_financials


def run_competitive_analysis(query: str) -> dict:
    """
    Competitive advantage agent: economic moat analysis using Porter's Five Forces,
    peer comparison, and moat durability rating.

    Highlights Domain 4: few-shot prompting with explicit moat classification schema.
    """
    executor = build_agent_executor(COMPETITIVE_SYSTEM, [get_peers, get_stock_info, get_financials])
    return run_agent_with_retry(executor, query)
