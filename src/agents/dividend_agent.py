from src.agents.base import build_agent_executor, run_agent_with_retry
from src.prompts.templates import DIVIDEND_SYSTEM
from src.tools.market_data import get_dividend_history, get_financials, get_stock_info


def run_dividend_analysis(query: str) -> dict:
    """
    Dividend strategy agent: yield analysis, sustainability, growth trajectory,
    and comparison with fixed income alternatives.
    """
    executor = build_agent_executor(DIVIDEND_SYSTEM, [get_dividend_history, get_financials, get_stock_info])
    return run_agent_with_retry(executor, query)
