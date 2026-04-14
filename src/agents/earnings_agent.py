from src.agents.base import build_agent_executor, run_agent_with_retry
from src.prompts.templates import EARNINGS_SYSTEM
from src.tools.market_data import get_quarterly_earnings, get_financials, get_stock_info


def run_earnings_analysis(query: str) -> dict:
    """
    Earnings breakdown agent: quarterly performance, margin trends, guidance analysis.
    """
    executor = build_agent_executor(EARNINGS_SYSTEM, [get_quarterly_earnings, get_financials, get_stock_info])
    return run_agent_with_retry(executor, query)
