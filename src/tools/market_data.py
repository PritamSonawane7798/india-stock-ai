"""
Custom tools for fetching Indian stock market data via yfinance.

NSE-listed stocks use the `.NS` suffix, BSE stocks use `.BO`.
All tools return structured dicts so downstream agents can parse reliably.
"""

import json
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import yfinance as yf
from langchain.tools import tool


def _nse(ticker: str) -> str:
    t = ticker.upper().strip()
    if not t.endswith((".NS", ".BO")):
        return t + ".NS"
    return t


@tool
def get_stock_info(ticker: str) -> str:
    """
    Fetch fundamental company information for an Indian stock.

    Args:
        ticker: NSE ticker symbol (e.g., RELIANCE, TCS, INFY).
                The .NS suffix is added automatically for NSE stocks.

    Returns:
        JSON string with company name, sector, market cap, P/E, P/B, EPS,
        52-week high/low, dividend yield, and key financial ratios.
    """
    try:
        stock = yf.Ticker(_nse(ticker))
        info = stock.info

        result = {
            "ticker": ticker.upper(),
            "name": info.get("longName", "N/A"),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap_cr": round((info.get("marketCap", 0) or 0) / 1e7, 2),
            "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "pb_ratio": info.get("priceToBook"),
            "eps_ttm": info.get("trailingEps"),
            "roe": round((info.get("returnOnEquity") or 0) * 100, 2),
            "roce": None,
            "debt_to_equity": info.get("debtToEquity"),
            "current_ratio": info.get("currentRatio"),
            "dividend_yield_pct": round((info.get("dividendYield") or 0) * 100, 2),
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
            "beta": info.get("beta"),
            "book_value": info.get("bookValue"),
            "revenue_cr": round((info.get("totalRevenue") or 0) / 1e7, 2),
            "profit_margin_pct": round((info.get("profitMargins") or 0) * 100, 2),
            "operating_margin_pct": round((info.get("operatingMargins") or 0) * 100, 2),
            "free_cash_flow_cr": round((info.get("freeCashflow") or 0) / 1e7, 2),
            "analyst_target_price": info.get("targetMeanPrice"),
            "recommendation": info.get("recommendationKey", "N/A"),
            "currency": info.get("currency", "INR"),
        }
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e), "ticker": ticker})


@tool
def get_price_history(ticker: str, period: str = "1y") -> str:
    """
    Fetch historical OHLCV price data for an Indian stock.

    Args:
        ticker: NSE ticker symbol (e.g., RELIANCE, TCS).
        period: Time period — one of 1mo, 3mo, 6mo, 1y, 2y, 5y.

    Returns:
        JSON string with date-indexed OHLCV data and basic return stats.
    """
    try:
        stock = yf.Ticker(_nse(ticker))
        df = stock.history(period=period)
        if df.empty:
            return json.dumps({"error": "No price data found", "ticker": ticker})

        df.index = df.index.strftime("%Y-%m-%d")
        records = df[["Open", "High", "Low", "Close", "Volume"]].round(2).to_dict(orient="index")

        closes = df["Close"]
        total_return = round(((closes.iloc[-1] / closes.iloc[0]) - 1) * 100, 2)
        volatility = round(closes.pct_change().std() * (252 ** 0.5) * 100, 2)

        return json.dumps({
            "ticker": ticker.upper(),
            "period": period,
            "total_return_pct": total_return,
            "annualized_volatility_pct": volatility,
            "current_price": round(float(closes.iloc[-1]), 2),
            "start_price": round(float(closes.iloc[0]), 2),
            "data_points": len(df),
            "prices": records,
        })
    except Exception as e:
        return json.dumps({"error": str(e), "ticker": ticker})


@tool
def get_financials(ticker: str) -> str:
    """
    Fetch annual income statement, balance sheet, and cash flow data.

    Args:
        ticker: NSE ticker symbol (e.g., HDFCBANK, WIPRO).

    Returns:
        JSON string with revenue, EBITDA, net income, total debt,
        total equity, operating cash flow for the last 4 fiscal years.
    """
    try:
        stock = yf.Ticker(_nse(ticker))

        def df_to_crores(df: pd.DataFrame) -> dict:
            if df is None or df.empty:
                return {}
            df = df / 1e7
            df.columns = [str(c)[:10] for c in df.columns]
            return df.round(2).to_dict()

        income = stock.financials
        balance = stock.balance_sheet
        cashflow = stock.cashflow

        def safe_row(df, *keys):
            for k in keys:
                if df is not None and k in df.index:
                    row = df.loc[k]
                    return {str(c)[:10]: round(float(v) / 1e7, 2) for c, v in row.items() if pd.notna(v)}
            return {}

        return json.dumps({
            "ticker": ticker.upper(),
            "revenue_cr": safe_row(income, "Total Revenue"),
            "gross_profit_cr": safe_row(income, "Gross Profit"),
            "ebitda_cr": safe_row(income, "EBITDA"),
            "net_income_cr": safe_row(income, "Net Income"),
            "total_debt_cr": safe_row(balance, "Total Debt", "Long Term Debt"),
            "total_equity_cr": safe_row(balance, "Stockholders Equity", "Total Equity Gross Minority Interest"),
            "total_assets_cr": safe_row(balance, "Total Assets"),
            "operating_cashflow_cr": safe_row(cashflow, "Operating Cash Flow"),
            "capex_cr": safe_row(cashflow, "Capital Expenditure"),
            "free_cashflow_cr": safe_row(cashflow, "Free Cash Flow"),
        })
    except Exception as e:
        return json.dumps({"error": str(e), "ticker": ticker})


@tool
def get_quarterly_earnings(ticker: str) -> str:
    """
    Fetch quarterly earnings history with EPS estimates vs actuals.

    Args:
        ticker: NSE ticker symbol.

    Returns:
        JSON string with quarterly EPS actual, estimated, and surprise percentage.
    """
    try:
        stock = yf.Ticker(_nse(ticker))
        earnings = stock.quarterly_earnings

        if earnings is None or earnings.empty:
            return json.dumps({"error": "No quarterly earnings data", "ticker": ticker})

        earnings.index = earnings.index.astype(str)
        result = []
        for quarter, row in earnings.iterrows():
            result.append({
                "quarter": quarter,
                "eps_actual": round(float(row.get("Earnings", 0) or 0), 2),
                "eps_estimate": round(float(row.get("Estimate", 0) or 0), 2),
                "surprise_pct": round(float(row.get("Surprise(%)", 0) or 0), 2),
            })

        return json.dumps({"ticker": ticker.upper(), "quarterly_earnings": result})
    except Exception as e:
        return json.dumps({"error": str(e), "ticker": ticker})


@tool
def get_dividend_history(ticker: str) -> str:
    """
    Fetch complete dividend payment history for an Indian stock.

    Args:
        ticker: NSE ticker symbol.

    Returns:
        JSON with all dividend payments, yield calculations, and payout consistency score.
    """
    try:
        stock = yf.Ticker(_nse(ticker))
        divs = stock.dividends
        info = stock.info

        if divs.empty:
            return json.dumps({"ticker": ticker.upper(), "dividends": [], "message": "No dividend history"})

        divs.index = divs.index.strftime("%Y-%m-%d")
        div_list = [{"date": d, "amount": round(float(v), 2)} for d, v in divs.items()]

        annual = {}
        for item in div_list:
            year = item["date"][:4]
            annual[year] = round(annual.get(year, 0) + item["amount"], 2)

        current_price = info.get("currentPrice") or info.get("regularMarketPrice") or 1
        latest_annual = list(annual.values())[-1] if annual else 0
        trailing_yield = round((latest_annual / current_price) * 100, 2) if current_price else 0

        years_paying = len(annual)
        consistency_score = min(10, years_paying)

        return json.dumps({
            "ticker": ticker.upper(),
            "dividends": div_list[-20:],
            "annual_dividends": annual,
            "trailing_dividend_yield_pct": trailing_yield,
            "years_of_consistent_dividend": years_paying,
            "consistency_score": f"{consistency_score}/10",
            "current_price": current_price,
        })
    except Exception as e:
        return json.dumps({"error": str(e), "ticker": ticker})


@tool
def screen_stocks(criteria: str) -> str:
    """
    Screen a predefined universe of large-cap Indian stocks against given criteria.

    Args:
        criteria: JSON string with screening criteria, e.g.:
                  '{"max_pe": 30, "min_roe": 15, "min_market_cap_cr": 10000}'
                  Supported fields: max_pe, min_roe, min_market_cap_cr,
                  max_debt_to_equity, min_dividend_yield_pct, sector.

    Returns:
        JSON string with list of stocks that pass the screening criteria.
    """
    universe = [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
        "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
        "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "SUNPHARMA",
        "TITAN", "BAJFINANCE", "WIPRO", "NESTLEIND", "ULTRACEMCO",
        "TECHM", "POWERGRID", "NTPC", "ONGC", "COALINDIA",
        "DIVISLAB", "DRREDDY", "CIPLA", "TATASTEEL", "JSWSTEEL",
    ]

    try:
        filters = json.loads(criteria) if isinstance(criteria, str) else criteria
    except Exception:
        filters = {}

    passed = []
    for symbol in universe:
        try:
            stock = yf.Ticker(f"{symbol}.NS")
            info = stock.info
            if not info.get("currentPrice"):
                continue

            pe = info.get("trailingPE") or 999
            roe = (info.get("returnOnEquity") or 0) * 100
            mcap = (info.get("marketCap") or 0) / 1e7
            dte = info.get("debtToEquity") or 0
            div_yield = (info.get("dividendYield") or 0) * 100
            sector = info.get("sector", "")

            if filters.get("max_pe") and pe > filters["max_pe"]:
                continue
            if filters.get("min_roe") and roe < filters["min_roe"]:
                continue
            if filters.get("min_market_cap_cr") and mcap < filters["min_market_cap_cr"]:
                continue
            if filters.get("max_debt_to_equity") and dte > filters["max_debt_to_equity"]:
                continue
            if filters.get("min_dividend_yield_pct") and div_yield < filters["min_dividend_yield_pct"]:
                continue
            if filters.get("sector") and filters["sector"].lower() not in sector.lower():
                continue

            passed.append({
                "ticker": symbol,
                "name": info.get("longName", symbol),
                "sector": sector,
                "pe": round(pe, 1) if pe < 999 else None,
                "roe_pct": round(roe, 1),
                "market_cap_cr": round(mcap, 0),
                "debt_to_equity": round(dte, 2),
                "dividend_yield_pct": round(div_yield, 2),
                "current_price": info.get("currentPrice"),
            })
        except Exception:
            continue

    return json.dumps({
        "criteria_applied": filters,
        "stocks_screened": len(universe),
        "stocks_passed": len(passed),
        "results": passed,
    })


@tool
def get_peers(ticker: str) -> str:
    """
    Fetch comparable peer companies in the same sector for competitive analysis.

    Args:
        ticker: NSE ticker symbol.

    Returns:
        JSON with peer companies and their key valuation multiples for comparison.
    """
    sector_peers = {
        "Information Technology": ["TCS", "INFY", "WIPRO", "TECHM", "HCLTECH"],
        "Financial Services": ["HDFCBANK", "ICICIBANK", "KOTAKBANK", "AXISBANK", "SBIN"],
        "Consumer Defensive": ["HINDUNILVR", "ITC", "NESTLEIND", "DABUR", "MARICO"],
        "Energy": ["RELIANCE", "ONGC", "BPCL", "IOC", "GAIL"],
        "Basic Materials": ["TATASTEEL", "JSWSTEEL", "HINDALCO", "ULTRACEMCO", "AMBUJACEM"],
        "Healthcare": ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "APOLLOHOSP"],
        "Consumer Cyclical": ["MARUTI", "TITAN", "BAJAJ-AUTO", "EICHERMOT", "HEROMOTOCO"],
        "Industrials": ["LT", "POWERGRID", "NTPC", "BHEL", "SIEMENS"],
    }

    try:
        stock = yf.Ticker(_nse(ticker))
        sector = stock.info.get("sector", "")
        peers = []
        for s, tickers in sector_peers.items():
            if s.lower() in sector.lower():
                peers = [t for t in tickers if t != ticker.upper()][:4]
                break

        if not peers:
            peers = ["TCS", "RELIANCE", "HDFCBANK", "INFY"]

        results = []
        for p in peers:
            try:
                pi = yf.Ticker(f"{p}.NS").info
                results.append({
                    "ticker": p,
                    "name": pi.get("longName", p),
                    "pe": round(pi.get("trailingPE") or 0, 1),
                    "pb": round(pi.get("priceToBook") or 0, 2),
                    "roe_pct": round((pi.get("returnOnEquity") or 0) * 100, 1),
                    "market_cap_cr": round((pi.get("marketCap") or 0) / 1e7, 0),
                    "profit_margin_pct": round((pi.get("profitMargins") or 0) * 100, 1),
                    "revenue_growth_pct": round((pi.get("revenueGrowth") or 0) * 100, 1),
                })
            except Exception:
                continue

        return json.dumps({
            "ticker": ticker.upper(),
            "sector": sector,
            "peers": results,
        })
    except Exception as e:
        return json.dumps({"error": str(e), "ticker": ticker})
