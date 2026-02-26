"""
Igris AI Agent — Web Search Skill  (Issue #1 — OpenClaw skill)

Justification:
A terminal AI agent needs access to real-time information beyond its training data.
This skill uses DuckDuckGo search (no API key required) to fetch web results.
"""

from langchain.tools import tool

try:
    from duckduckgo_search import DDGS
    _DDGS_AVAILABLE = True
except ImportError:
    _DDGS_AVAILABLE = False


@tool
def web_search(query: str) -> str:
    """Search the web using DuckDuckGo and return top results.
    Use this when the user asks about current events, facts, or anything
    that requires up-to-date information."""
    if not _DDGS_AVAILABLE:
        return ("Web search unavailable. Install it with: pip install duckduckgo-search")

    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=5):
                title = r.get("title", "No title")
                body = r.get("body", "No snippet")
                href = r.get("href", "")
                results.append(f"**{title}**\n{body}\nURL: {href}")

        if not results:
            return f"No results found for: {query}"

        return "\n\n---\n\n".join(results)
    except Exception as e:
        return f"Web search failed: {e}"


WEB_SEARCH_TOOLS = [web_search]
