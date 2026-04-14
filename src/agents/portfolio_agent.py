from src.agents.base import build_agent_executor, run_agent_with_retry
from src.prompts.templates import PORTFOLIO_SYSTEM
from src.tools.financial_calc import optimize_portfolio
from src.tools.market_data import get_stock_info, screen_stocks


def run_portfolio_builder(query: str) -> dict:
    """
    Portfolio builder agent: Markowitz mean-variance optimization with
    sector diversification for Indian retail investors.

    Demonstrates Domain 1: task decomposition — screen → select → optimize.
    """
    executor = build_agent_executor(PORTFOLIO_SYSTEM, [optimize_portfolio, get_stock_info, screen_stocks])
    return run_agent_with_retry(executor, query)
