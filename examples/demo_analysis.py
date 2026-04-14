"""
Demo script: runs the analysis pipeline without the Streamlit UI.
Useful for testing your setup before launching the web app.

Usage:
    python examples/demo_analysis.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()

from src.agents.orchestrator import run_analysis


def print_section(title: str, content):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("=" * 60)
    if isinstance(content, dict):
        print(json.dumps(content, indent=2))
    else:
        print(content)


def main():
    print("\n🇮🇳  IndiaStockAI — Demo Analysis")
    print("Powered by Claude Sonnet + LangGraph\n")

    query = "Give me a DCF valuation and risk analysis for TCS"
    print(f"Query: {query}\n")

    print("Running agents (this may take 30-60 seconds)...")
    result = run_analysis(query)

    print_section("TICKER IDENTIFIED", result.get("ticker", "N/A"))
    print_section("AGENTS INVOKED", result.get("agents_run", []))

    for msg in result.get("agent_results", []):
        if msg["status"] != "skipped":
            print_section(f"{msg['agent'].upper()} AGENT [{msg['status'].upper()}]", msg["result"])

    print_section("FINAL INVESTMENT REPORT", result.get("final_report", {}))


if __name__ == "__main__":
    main()
