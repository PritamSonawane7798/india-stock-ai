"""
Technical analysis tools using the `ta` library.

Computes momentum, trend, volatility, and volume indicators
from historical OHLCV data and returns a structured analysis.
"""

import json

import pandas as pd
import yfinance as yf
from langchain.tools import tool

try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False


def _nse(t: str) -> str:
    t = t.upper().strip()
    return t if t.endswith((".NS", ".BO")) else t + ".NS"


def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if not TA_AVAILABLE:
        return df

    df["rsi"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()

    macd = ta.trend.MACD(df["Close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    bb = ta.volatility.BollingerBands(df["Close"], window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_lower"] = bb.bollinger_lband()

    df["ema_20"] = ta.trend.EMAIndicator(df["Close"], window=20).ema_indicator()
    df["ema_50"] = ta.trend.EMAIndicator(df["Close"], window=50).ema_indicator()
    df["ema_200"] = ta.trend.EMAIndicator(df["Close"], window=200).ema_indicator()

    df["sma_20"] = ta.trend.SMAIndicator(df["Close"], window=20).sma_indicator()
    df["sma_50"] = ta.trend.SMAIndicator(df["Close"], window=50).sma_indicator()

    adx = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"], window=14)
    df["adx"] = adx.adx()

    stoch = ta.momentum.StochasticOscillator(df["High"], df["Low"], df["Close"])
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    df["atr"] = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"]).average_true_range()

    df["obv"] = ta.volume.OnBalanceVolumeIndicator(df["Close"], df["Volume"]).on_balance_volume()

    return df


@tool
def get_technical_analysis(ticker: str, period: str = "1y") -> str:
    """
    Perform comprehensive technical analysis on an Indian stock.

    Computes RSI, MACD, Bollinger Bands, EMA/SMA crossovers, ADX,
    Stochastic Oscillator, ATR, and OBV. Generates buy/sell signals.

    Args:
        ticker: NSE ticker symbol (e.g., RELIANCE, TCS).
        period: historical period — 6mo, 1y, 2y (default: 1y).

    Returns:
        JSON with all indicators, current values, signals, support/resistance,
        trend direction, and an overall technical score (Bullish/Bearish/Neutral).
    """
    try:
        stock = yf.Ticker(_nse(ticker))
        df = stock.history(period=period)

        if df.empty or len(df) < 50:
            return json.dumps({"error": "Insufficient price data", "ticker": ticker})

        df = _add_indicators(df)
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        price = float(latest["Close"])

        signals = []
        bullish_count = 0
        bearish_count = 0

        if TA_AVAILABLE:
            rsi = float(latest.get("rsi", 50) or 50)
            if rsi < 30:
                signals.append({"indicator": "RSI", "signal": "BUY", "reason": f"RSI={rsi:.1f} — oversold territory"})
                bullish_count += 1
            elif rsi > 70:
                signals.append({"indicator": "RSI", "signal": "SELL", "reason": f"RSI={rsi:.1f} — overbought territory"})
                bearish_count += 1
            else:
                signals.append({"indicator": "RSI", "signal": "NEUTRAL", "reason": f"RSI={rsi:.1f} — neutral zone"})

            macd = float(latest.get("macd", 0) or 0)
            macd_sig = float(latest.get("macd_signal", 0) or 0)
            prev_macd = float(prev.get("macd", 0) or 0)
            prev_sig = float(prev.get("macd_signal", 0) or 0)
            if macd > macd_sig and prev_macd <= prev_sig:
                signals.append({"indicator": "MACD", "signal": "BUY", "reason": "MACD bullish crossover"})
                bullish_count += 1
            elif macd < macd_sig and prev_macd >= prev_sig:
                signals.append({"indicator": "MACD", "signal": "SELL", "reason": "MACD bearish crossover"})
                bearish_count += 1
            else:
                sig = "BUY" if macd > macd_sig else "SELL"
                signals.append({"indicator": "MACD", "signal": sig, "reason": f"MACD {'above' if macd > macd_sig else 'below'} signal line"})
                if sig == "BUY":
                    bullish_count += 0.5
                else:
                    bearish_count += 0.5

            ema20 = float(latest.get("ema_20") or price)
            ema50 = float(latest.get("ema_50") or price)
            ema200 = float(latest.get("ema_200") or price)
            if price > ema20 > ema50 > ema200:
                signals.append({"indicator": "EMA", "signal": "BUY", "reason": "Price above all EMAs — strong uptrend"})
                bullish_count += 1
            elif price < ema20 < ema50 < ema200:
                signals.append({"indicator": "EMA", "signal": "SELL", "reason": "Price below all EMAs — strong downtrend"})
                bearish_count += 1
            else:
                signals.append({"indicator": "EMA", "signal": "NEUTRAL", "reason": "Mixed EMA alignment"})

            bb_upper = float(latest.get("bb_upper") or price * 1.05)
            bb_lower = float(latest.get("bb_lower") or price * 0.95)
            if price <= bb_lower:
                signals.append({"indicator": "Bollinger Bands", "signal": "BUY", "reason": "Price at lower band — potential reversal"})
                bullish_count += 0.5
            elif price >= bb_upper:
                signals.append({"indicator": "Bollinger Bands", "signal": "SELL", "reason": "Price at upper band — potential pullback"})
                bearish_count += 0.5

            adx_val = float(latest.get("adx", 20) or 20)
            stoch_k = float(latest.get("stoch_k", 50) or 50)
            if stoch_k < 20:
                signals.append({"indicator": "Stochastic", "signal": "BUY", "reason": f"Stoch K={stoch_k:.1f} — oversold"})
                bullish_count += 0.5
            elif stoch_k > 80:
                signals.append({"indicator": "Stochastic", "signal": "SELL", "reason": f"Stoch K={stoch_k:.1f} — overbought"})
                bearish_count += 0.5

        recent_high = float(df["High"].tail(20).max())
        recent_low = float(df["Low"].tail(20).min())
        support = round(recent_low, 2)
        resistance = round(recent_high, 2)

        if bullish_count > bearish_count * 1.5:
            overall = "BULLISH"
        elif bearish_count > bullish_count * 1.5:
            overall = "BEARISH"
        else:
            overall = "NEUTRAL"

        recent_closes = df["Close"].tail(5).round(2).tolist()
        recent_volumes = df["Volume"].tail(5).tolist()
        avg_volume = int(df["Volume"].tail(20).mean())
        latest_volume = int(latest["Volume"])
        volume_signal = "HIGH" if latest_volume > avg_volume * 1.5 else "NORMAL" if latest_volume > avg_volume * 0.5 else "LOW"

        return json.dumps({
            "ticker": ticker.upper(),
            "current_price": round(price, 2),
            "period": period,
            "indicators": {
                "rsi_14": round(float(latest.get("rsi", 50) or 50), 2) if TA_AVAILABLE else None,
                "macd": round(float(latest.get("macd", 0) or 0), 4) if TA_AVAILABLE else None,
                "macd_signal": round(float(latest.get("macd_signal", 0) or 0), 4) if TA_AVAILABLE else None,
                "ema_20": round(float(latest.get("ema_20") or price), 2) if TA_AVAILABLE else None,
                "ema_50": round(float(latest.get("ema_50") or price), 2) if TA_AVAILABLE else None,
                "ema_200": round(float(latest.get("ema_200") or price), 2) if TA_AVAILABLE else None,
                "bb_upper": round(float(latest.get("bb_upper") or price), 2) if TA_AVAILABLE else None,
                "bb_lower": round(float(latest.get("bb_lower") or price), 2) if TA_AVAILABLE else None,
                "adx_14": round(float(latest.get("adx", 0) or 0), 2) if TA_AVAILABLE else None,
                "stoch_k": round(float(latest.get("stoch_k", 50) or 50), 2) if TA_AVAILABLE else None,
                "atr": round(float(latest.get("atr", 0) or 0), 2) if TA_AVAILABLE else None,
            },
            "signals": signals,
            "overall_signal": overall,
            "bullish_signals": bullish_count,
            "bearish_signals": bearish_count,
            "support_level": support,
            "resistance_level": resistance,
            "volume_analysis": {
                "latest_volume": latest_volume,
                "avg_20d_volume": avg_volume,
                "volume_signal": volume_signal,
            },
            "recent_closes": recent_closes,
        })
    except Exception as e:
        return json.dumps({"error": str(e), "ticker": ticker})
