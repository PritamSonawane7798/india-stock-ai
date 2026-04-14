"""
Base agent setup with multi-provider LLM support.

Supports three providers, selected via LLM_PROVIDER in .env:
  - anthropic  (Claude Sonnet — default)
  - groq       (Llama 3.3 70B — free tier, fastest)
  - ollama     (fully local, no API key needed)

Implements Domain 5 (Context Management & Reliability) patterns:
- Error propagation via status fields
- Validation-retry loops for structured output
- Progressive context summarization
"""

import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models import BaseChatModel

load_dotenv()


def get_llm(temperature: float = 0.1) -> BaseChatModel:
    """
    Returns a configured LLM based on LLM_PROVIDER env var.

    Provider options (set in .env):
        LLM_PROVIDER=groq       → Groq Llama 3.3 70B (free, recommended)
        LLM_PROVIDER=ollama     → Ollama local model (zero cost)
        LLM_PROVIDER=anthropic  → Claude Sonnet (default)
    """
    provider = os.environ.get("LLM_PROVIDER", "anthropic").lower()

    if provider == "groq":
        from langchain_groq import ChatGroq
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY not set.\n"
                "Get a free key at https://console.groq.com → API Keys.\n"
                "Add GROQ_API_KEY=your_key to .env"
            )
        model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
        return ChatGroq(api_key=api_key, model=model, temperature=temperature)

    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        model = os.environ.get("OLLAMA_MODEL", "llama3.1")
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(model=model, base_url=base_url, temperature=temperature)

    else:
        from langchain_anthropic import ChatAnthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY not set. Copy .env.example to .env and add your key."
            )
        model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        return ChatAnthropic(
            model=model,
            anthropic_api_key=api_key,
            temperature=temperature,
            max_tokens=4096,
        )


def build_agent_executor(system_prompt: str, tools: list, verbose: bool = False) -> AgentExecutor:
    """
    Builds a LangChain tool-calling agent with the given system prompt and tools.

    Uses create_tool_calling_agent which works with any provider that supports
    the tool-calling interface — aligning with Domain 2 (Tool Design & MCP Integration).
    """
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=verbose, handle_parsing_errors=True, max_iterations=6)


def safe_parse_json(text: str) -> dict:
    """Extract and parse the first JSON block from agent output. Fallback to raw text."""
    text = text.strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {"data": parsed}
    except Exception:
        pass

    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(1))
            return parsed if isinstance(parsed, dict) else {"data": parsed}
        except Exception:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        try:
            parsed = json.loads(text[start:end + 1])
            return parsed if isinstance(parsed, dict) else {"data": parsed}
        except Exception:
            pass

    return {"raw_output": text, "parse_error": "Could not extract structured JSON"}


def _extract_text(output) -> str:
    """
    Normalize agent output to a plain string.

    Some providers return a list of content blocks, e.g.:
        [{"type": "text", "text": "...", "index": 0}]
    Others return a plain string. Handle both.
    """
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        parts = []
        for block in output:
            if isinstance(block, dict):
                parts.append(block.get("text") or block.get("content") or "")
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(p for p in parts if p).strip()
    return str(output)


def run_agent_with_retry(executor: AgentExecutor, query: str, retries: int = 2) -> dict:
    """
    Run an agent executor with automatic retry on failure.

    Implements the validation-retry loop pattern from Domain 4.
    Each retry injects the previous error as context for self-correction.
    """
    last_error = None
    for attempt in range(retries + 1):
        try:
            if attempt > 0:
                query = (
                    f"{query}\n\n[Retry {attempt}: Previous attempt failed with: {last_error}. "
                    "Please ensure your response is valid JSON.]"
                )
            result = executor.invoke({"input": query})
            output = _extract_text(result.get("output", ""))
            parsed = safe_parse_json(output)
            # If the model returned markdown prose (no JSON found), store it
            # as structured markdown so the UI can render it nicely.
            if "parse_error" in parsed and "raw_output" in parsed:
                return {"markdown": parsed["raw_output"]}
            return parsed
        except Exception as e:
            last_error = str(e)

    return {"error": f"Agent failed after {retries + 1} attempts: {last_error}"}
