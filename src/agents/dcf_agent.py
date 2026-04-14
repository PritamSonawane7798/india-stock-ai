from src.agents.base import build_agent_executor, run_agent_with_retry
from src.prompts.templates import DCF_SYSTEM
from src.tools.financial_calc import run_dcf_valuation
from src.tools.market_data import get_financials, get_stock_info


def run_dcf_analysis(query: str) -> dict:
    """
    DCF valuation agent: runs intrinsic value analysis with scenario modelling.

    Highlights Domain 4: structured output with explicit JSON schema requirements.
    """
    executor = build_agent_executor(DCF_SYSTEM, [run_dcf_valuation, get_financials, get_stock_info])
    return run_agent_with_retry(executor, query)
