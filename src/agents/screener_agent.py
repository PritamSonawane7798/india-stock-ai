from src.agents.base import build_agent_executor, run_agent_with_retry
from src.prompts.templates import SCREENER_SYSTEM
from src.tools.market_data import screen_stocks, get_stock_info


def run_screener(query: str) -> dict:
    """
    Stock screener agent: filters Indian large-cap stocks against criteria.

    Demonstrates Domain 2 concepts: tool descriptions guide LLM tool selection.
    """
    executor = build_agent_executor(SCREENER_SYSTEM, [screen_stocks, get_stock_info])
    return run_agent_with_retry(executor, query)
