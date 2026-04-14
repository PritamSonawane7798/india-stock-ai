"""
Financial calculation tools: DCF valuation, risk metrics, portfolio optimization.

These tools encapsulate pure financial logic so agents can call them
without needing to implement the math themselves.
"""

import json
import math
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from langchain.tools import tool
from scipy.optimize import minimize


def _nse(t: str) -> str:
    t = t.upper().strip()
    return t if t.endswith((".NS", ".BO")) else t + ".NS"


@tool
def run_dcf_valuation(params: str) -> str:
    """
    Run a Discounted Cash Flow (DCF) valuation for an Indian stock.

    Args:
        params: JSON string with required fields:
                - ticker: NSE stock symbol
                - growth_rate_yr1_3: expected FCF growth rate for years 1-3 (e.g., 0.15 for 15%)
                - growth_rate_yr4_10: expected FCF growth rate for years 4-10 (e.g., 0.10)
                - terminal_growth_rate: long-term growth rate (e.g., 0.05)
                - discount_rate: WACC / required rate of return (e.g., 0.12)

    Returns:
        JSON with intrinsic value per share, margin of safety, and valuation verdict.
    """
    try:
        p = json.loads(params) if isinstance(params, str) else params
        ticker = p["ticker"]
        g1 = float(p.get("growth_rate_yr1_3", 0.15))
        g2 = float(p.get("growth_rate_yr4_10", 0.10))
        tg = float(p.get("terminal_growth_rate", 0.05))
        dr = float(p.get("discount_rate", 0.12))

        stock = yf.Ticker(_nse(ticker))
        info = stock.info
        cf = stock.cashflow

        base_fcf = None
        if cf is not None and not cf.empty:
            for key in ["Free Cash Flow", "Operating Cash Flow"]:
                if key in cf.index:
                    vals = cf.loc[key].dropna()
                    if not vals.empty:
                        base_fcf = float(vals.iloc[0]) / 1e7
                        break

        if base_fcf is None or base_fcf <= 0:
            eps = info.get("trailingEps") or 0
            shares = (info.get("sharesOutstanding") or 1e8)
            base_fcf = (eps * shares) / 1e7 if eps > 0 else 100

        projected_fcf = []
        for yr in range(1, 11):
            g = g1 if yr <= 3 else g2
            fcf = base_fcf * ((1 + g) ** yr)
            pv = fcf / ((1 + dr) ** yr)
            projected_fcf.append({"year": yr, "fcf_cr": round(fcf, 2), "pv_cr": round(pv, 2)})

        terminal_value = (projected_fcf[-1]["fcf_cr"] * (1 + tg)) / (dr - tg)
        pv_terminal = terminal_value / ((1 + dr) ** 10)

        total_pv = sum(x["pv_cr"] for x in projected_fcf) + pv_terminal
        net_debt = ((info.get("totalDebt") or 0) - (info.get("totalCash") or 0)) / 1e7
        equity_value = max(total_pv - net_debt, 0)

        shares = (info.get("sharesOutstanding") or 1e8) / 1e7
        intrinsic_value = equity_value / shares if shares > 0 else 0

        current_price = info.get("currentPrice") or info.get("regularMarketPrice") or 1
        margin_of_safety = round(((intrinsic_value - current_price) / intrinsic_value) * 100, 1) if intrinsic_value > 0 else 0

        if margin_of_safety >= 30:
            verdict = "UNDERVALUED — Strong Buy Signal"
        elif margin_of_safety >= 10:
            verdict = "SLIGHTLY UNDERVALUED — Consider Buying"
        elif margin_of_safety >= -10:
            verdict = "FAIRLY VALUED"
        elif margin_of_safety >= -30:
            verdict = "SLIGHTLY OVERVALUED — Caution"
        else:
            verdict = "OVERVALUED — Avoid or Wait"

        return json.dumps({
            "ticker": ticker.upper(),
            "base_fcf_cr": round(base_fcf, 2),
            "assumptions": {"g1_3": g1, "g4_10": g2, "terminal_growth": tg, "discount_rate": dr},
            "projected_fcf": projected_fcf,
            "terminal_value_cr": round(terminal_value, 2),
            "pv_terminal_cr": round(pv_terminal, 2),
            "total_enterprise_value_cr": round(total_pv, 2),
            "net_debt_cr": round(net_debt, 2),
            "equity_value_cr": round(equity_value, 2),
            "intrinsic_value_per_share": round(intrinsic_value, 2),
            "current_market_price": round(current_price, 2),
            "margin_of_safety_pct": margin_of_safety,
            "verdict": verdict,
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def calculate_risk_metrics(params: str) -> str:
    """
    Calculate comprehensive risk metrics for an Indian stock.

    Args:
        params: JSON string with:
                - ticker: NSE stock symbol
                - period: historical period for calculation (default: '2y')
                - confidence_level: VaR confidence level (default: 0.95)

    Returns:
        JSON with beta, alpha, Sharpe ratio, VaR, CVaR, max drawdown,
        volatility, and an overall risk score (1-10).
    """
    try:
        p = json.loads(params) if isinstance(params, str) else params
        ticker = p["ticker"]
        period = p.get("period", "2y")
        conf = float(p.get("confidence_level", 0.95))

        stock_data = yf.Ticker(_nse(ticker)).history(period=period)["Close"]
        nifty_data = yf.Ticker("^NSEI").history(period=period)["Close"]

        stock_ret = stock_data.pct_change().dropna()
        nifty_ret = nifty_data.pct_change().dropna()

        aligned = pd.concat([stock_ret, nifty_ret], axis=1).dropna()
        aligned.columns = ["stock", "nifty"]

        cov = np.cov(aligned["stock"], aligned["nifty"])
        beta = cov[0][1] / cov[1][1] if cov[1][1] != 0 else 1.0

        rf_daily = 0.065 / 252
        excess = aligned["stock"] - rf_daily
        sharpe = (excess.mean() / excess.std()) * (252 ** 0.5) if excess.std() != 0 else 0

        alpha = (aligned["stock"].mean() - rf_daily - beta * (aligned["nifty"].mean() - rf_daily)) * 252

        var_daily = float(np.percentile(aligned["stock"], (1 - conf) * 100))
        cvar_daily = float(aligned["stock"][aligned["stock"] <= var_daily].mean())

        cumulative = (1 + aligned["stock"]).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = float(drawdown.min()) * 100

        annualized_vol = float(aligned["stock"].std()) * (252 ** 0.5) * 100

        risk_score = 5
        if abs(beta) > 1.5:
            risk_score += 2
        elif abs(beta) > 1.2:
            risk_score += 1
        if annualized_vol > 40:
            risk_score += 2
        elif annualized_vol > 25:
            risk_score += 1
        if max_drawdown < -50:
            risk_score += 1
        risk_score = min(10, max(1, risk_score))

        return json.dumps({
            "ticker": ticker.upper(),
            "period": period,
            "beta": round(beta, 3),
            "alpha_annualized": round(alpha * 100, 2),
            "sharpe_ratio": round(sharpe, 3),
            "annualized_volatility_pct": round(annualized_vol, 2),
            "var_daily_pct": round(var_daily * 100, 3),
            "cvar_daily_pct": round(cvar_daily * 100, 3),
            "max_drawdown_pct": round(max_drawdown, 2),
            "risk_score": f"{risk_score}/10",
            "risk_label": "High" if risk_score >= 7 else "Medium" if risk_score >= 4 else "Low",
            "var_interpretation": f"At {int(conf*100)}% confidence, daily loss will not exceed {abs(round(var_daily*100,2))}%",
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def optimize_portfolio(params: str) -> str:
    """
    Run Mean-Variance (Markowitz) portfolio optimization for a set of Indian stocks.

    Args:
        params: JSON string with:
                - tickers: list of NSE ticker symbols (e.g., ["RELIANCE", "TCS", "HDFCBANK"])
                - period: historical period for returns (default: '2y')
                - objective: 'max_sharpe' or 'min_volatility' (default: 'max_sharpe')
                - investment_amount: total investment in INR (optional, default: 100000)

    Returns:
        JSON with optimal weights, expected annual return, volatility, Sharpe ratio,
        and suggested allocation in INR.
    """
    try:
        p = json.loads(params) if isinstance(params, str) else params
        tickers = p["tickers"]
        period = p.get("period", "2y")
        objective = p.get("objective", "max_sharpe")
        amount = float(p.get("investment_amount", 100000))

        prices = pd.DataFrame()
        for t in tickers:
            data = yf.Ticker(_nse(t)).history(period=period)["Close"]
            if not data.empty:
                prices[t] = data

        prices = prices.dropna()
        returns = prices.pct_change().dropna()
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        rf = 0.065
        n = len(tickers)

        def neg_sharpe(weights):
            port_ret = np.dot(weights, mean_returns)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return -(port_ret - rf) / port_vol if port_vol > 0 else 0

        def port_vol(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(0.05, 0.40)] * n
        x0 = np.array([1 / n] * n)

        fn = neg_sharpe if objective == "max_sharpe" else port_vol
        result = minimize(fn, x0, method="SLSQP", bounds=bounds, constraints=constraints)
        weights = result.x

        exp_ret = np.dot(weights, mean_returns)
        exp_vol = port_vol(weights)
        sharpe = (exp_ret - rf) / exp_vol if exp_vol > 0 else 0

        allocation = []
        for i, t in enumerate(tickers):
            w = round(float(weights[i]), 4)
            price = float(prices[t].iloc[-1])
            alloc_inr = round(amount * w, 0)
            shares = int(alloc_inr / price)
            allocation.append({
                "ticker": t,
                "weight_pct": round(w * 100, 2),
                "allocation_inr": alloc_inr,
                "shares": shares,
                "price": round(price, 2),
            })

        return json.dumps({
            "tickers": tickers,
            "objective": objective,
            "optimal_weights": allocation,
            "expected_annual_return_pct": round(exp_ret * 100, 2),
            "expected_annual_volatility_pct": round(exp_vol * 100, 2),
            "sharpe_ratio": round(sharpe, 3),
            "investment_amount_inr": amount,
            "correlation_matrix": returns.corr().round(3).to_dict(),
        })
    except Exception as e:
        return json.dumps({"error": str(e)})
