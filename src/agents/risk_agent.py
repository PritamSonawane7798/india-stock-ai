from src.agents.base import build_agent_executor, run_agent_with_retry
from src.prompts.templates import RISK_SYSTEM
from src.tools.financial_calc import calculate_risk_metrics
from src.tools.market_data import get_stock_info, get_price_history


def run_risk_analysis(query: str) -> dict:
    """
    Risk analysis agent: calculates beta, VaR, CVaR, Sharpe ratio, max drawdown.

    Highlights Domain 5: error propagation with structured status fields.
    """
    executor = build_agent_executor(RISK_SYSTEM, [calculate_risk_metrics, get_stock_info, get_price_history])
    return run_agent_with_retry(executor, query)
