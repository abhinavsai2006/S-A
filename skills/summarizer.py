"""
Igris AI Agent — Text Summarizer Skill  (Issue #1 — OpenClaw skill)

Justification:
Summarization is one of the most common tasks users need. Instead of sending
the full text to the main LLM (which may hit token limits), this skill provides
a dedicated summarization tool that handles chunking for long texts.
"""

from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from config import GROQ_API_KEY, FALLBACK_MODEL_NAME


def _get_summarizer_llm():
    """Get a lighter LLM instance specifically for summarization."""
    return ChatGroq(
        model=FALLBACK_MODEL_NAME,  # use lighter model for summarization efficiency
        api_key=GROQ_API_KEY,
        temperature=0.3,
        max_tokens=1024,
    )


@tool
def summarize_text(text: str) -> str:
    """Summarize a given text into key points.
    Use this when the user asks to summarize a long piece of text, article, or document content."""
    if not text or len(text.strip()) < 50:
        return "Text is too short to summarize meaningfully."

    try:
        llm = _get_summarizer_llm()

        # If text is very long, chunk it
        max_chunk = 6000  # characters per chunk
        if len(text) > max_chunk:
            chunks = [text[i:i + max_chunk] for i in range(0, len(text), max_chunk)]
            summaries = []

            for i, chunk in enumerate(chunks[:5]):  # limit to 5 chunks
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a precise summarizer. Summarize the following text into key bullet points. Be concise."),
                    ("human", f"Summarize this (part {i + 1}):\n\n{chunk}")
                ])
                chain = prompt | llm
                result = chain.invoke({})
                summaries.append(result.content)

            # Final consolidation
            combined = "\n\n".join(summaries)
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Combine these partial summaries into one coherent summary with key bullet points."),
                ("human", combined)
            ])
            chain = prompt | llm
            final = chain.invoke({})
            return f"📝 Summary:\n\n{final.content}"
        else:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a precise summarizer. Summarize the following text into key bullet points. Be concise but comprehensive."),
                ("human", f"Summarize this:\n\n{text}")
            ])
            chain = prompt | llm
            result = chain.invoke({})
            return f"📝 Summary:\n\n{result.content}"

    except Exception as e:
        return f"Summarization failed: {e}"


SUMMARIZER_TOOLS = [summarize_text]
