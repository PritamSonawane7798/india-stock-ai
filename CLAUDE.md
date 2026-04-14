# Indian Stock Market Analysis Agent — Claude Code Config

## Project Overview
Deep agentic AI system for Indian stock market analysis using LangGraph multi-agent orchestration,
custom tools, and Claude as the reasoning backbone.

## Architecture
- **Orchestrator**: LangGraph StateGraph with supervisor routing
- **Agents**: 8 specialized agents (screener, DCF, risk, earnings, portfolio, technical, dividend, competitive)
- **Tools**: Custom yfinance wrappers, financial calculators, technical indicators
- **UI**: Streamlit with Plotly charts

## Dev Commands
```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run src/app.py

# Run a demo analysis
python examples/demo_analysis.py
```

## File Conventions
- All agents live in `src/agents/`
- Custom tools in `src/tools/`
- Prompt templates in `src/prompts/templates.py`
- NSE stocks use `.NS` suffix in yfinance (e.g., `RELIANCE.NS`)

## Environment
Requires `.env` with `ANTHROPIC_API_KEY`. Copy `.env.example` to `.env`.

## Key Design Decisions
- Agents communicate via LangGraph state (typed TypedDict)
- All tools return structured JSON for reliable parsing
- Context is progressively summarized across agent handoffs
- Error propagation uses explicit status fields, not exceptions
