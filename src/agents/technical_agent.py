from src.agents.base import build_agent_executor, run_agent_with_retry
from src.prompts.templates import TECHNICAL_SYSTEM
from src.tools.technical_indicators import get_technical_analysis
from src.tools.market_data import get_price_history, get_stock_info


def run_technical_analysis(query: str) -> dict:
    """
    Technical analysis agent: RSI, MACD, Bollinger Bands, EMA crossovers,
    support/resistance levels, and buy/sell signals.
    """
    executor = build_agent_executor(TECHNICAL_SYSTEM, [get_technical_analysis, get_price_history, get_stock_info])
    return run_agent_with_retry(executor, query)
