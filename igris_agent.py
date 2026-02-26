"""
╔══════════════════════════════════════════════════════════════════╗
║  IGRIS AI AGENT — Terminal-Based AI Agent (Refactored)          ║
║                                                                  ║
║  Fixes Applied:                                                  ║
║  • Issue #1: OpenClaw skill base (7 skills added)                ║
║  • Issue #2: Model upgraded to llama-3.3-70b-versatile           ║
║  • Issue #3: LangGraph workflow + Pydantic AI + doc reader       ║
║  • Issue #4: System control (shutdown/reboot/sleep/lock)         ║
║  • Issue #5: Atomic memory saves (no more corruption)            ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import sys

# ──────────────────────────────────────────────
# Imports — config & memory
# ──────────────────────────────────────────────
from config import (
    GROQ_API_KEY,
    MODEL_NAME,
    MODEL_TEMPERATURE,
    MODEL_MAX_TOKENS,
    FALLBACK_MODEL_NAME,
    FALLBACK_MAX_TOKENS,
    SYSTEM_PROMPT,
)
from memory_manager import load_memory, save_memory, save_exchange

# ──────────────────────────────────────────────
# Imports — LangChain
# ──────────────────────────────────────────────
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent

# ──────────────────────────────────────────────
# Imports — Skills  (Issue #1)
# ──────────────────────────────────────────────
from skills.system_control import SYSTEM_CONTROL_TOOLS
from skills.web_search import WEB_SEARCH_TOOLS
from skills.file_manager import FILE_MANAGER_TOOLS
from skills.document_reader import DOCUMENT_READER_TOOLS
from skills.cloud_upload import CLOUD_UPLOAD_TOOLS
from skills.math_solver import MATH_SOLVER_TOOLS
from skills.summarizer import SUMMARIZER_TOOLS

# ──────────────────────────────────────────────
# Imports — LangGraph workflow  (Issue #3)
# ──────────────────────────────────────────────
from graph.workflow import process_with_workflow


# ══════════════════════════════════════════════
# Boot sequence
# ══════════════════════════════════════════════
def print_banner():
    """Print the startup banner."""
    banner = """
    ╔═══════════════════════════════════════════════╗
    ║           ⚔️  IGRIS AI AGENT  ⚔️              ║
    ║       Terminal-Based AI • Shadow Knight       ║
    ╠═══════════════════════════════════════════════╣
    ║  Model  : {model:<35s} ║
    ║  Tokens : {tokens:<35s} ║
    ║  Skills : {skills:<35s} ║
    ╚═══════════════════════════════════════════════╝
    """.format(
        model=MODEL_NAME,
        tokens=str(MODEL_MAX_TOKENS),
        skills=str(len(get_all_tools())),
    )
    print(banner)


def get_all_tools():
    """Collect all skill tools into a single list."""
    tools = []
    tools.extend(SYSTEM_CONTROL_TOOLS)
    tools.extend(WEB_SEARCH_TOOLS)
    tools.extend(FILE_MANAGER_TOOLS)
    tools.extend(DOCUMENT_READER_TOOLS)
    tools.extend(CLOUD_UPLOAD_TOOLS)
    tools.extend(MATH_SOLVER_TOOLS)
    tools.extend(SUMMARIZER_TOOLS)
    return tools


def print_skills():
    """Print available skills and their tools."""
    skill_map = {
        "System Control": SYSTEM_CONTROL_TOOLS,
        "Web Search": WEB_SEARCH_TOOLS,
        "File Manager": FILE_MANAGER_TOOLS,
        "Document Reader": DOCUMENT_READER_TOOLS,
        "Cloud Upload": CLOUD_UPLOAD_TOOLS,
        "Math Solver": MATH_SOLVER_TOOLS,
        "Summarizer": SUMMARIZER_TOOLS,
    }
    print("\n  📋 Available Skills:")
    print("  " + "─" * 45)
    for skill_name, tools in skill_map.items():
        tool_names = ", ".join(t.name for t in tools)
        print(f"  • {skill_name}: {tool_names}")
    print("  " + "─" * 45)
    print()


def create_agent():
    """
    Create the LangChain agent with all tools and tuned prompts.

    Issue #2: Model upgraded + prompt tuned
    Issue #1: All skill tools registered
    """
    # Validate API key
    if not GROQ_API_KEY:
        print("ERROR: GROQ_API_KEY not set.")
        print("Set it via environment variable: export GROQ_API_KEY='your-key-here'")
        print("Or on Windows: set GROQ_API_KEY=your-key-here")
        sys.exit(1)

    # Initialize the LLM  (Issue #2 — upgraded model)
    try:
        llm = ChatGroq(
            model=MODEL_NAME,
            api_key=GROQ_API_KEY,
            temperature=MODEL_TEMPERATURE,
            max_tokens=MODEL_MAX_TOKENS,
        )
    except Exception as e:
        print(f"Primary model ({MODEL_NAME}) failed: {e}")
        print(f"Falling back to {FALLBACK_MODEL_NAME}...")
        llm = ChatGroq(
            model=FALLBACK_MODEL_NAME,
            api_key=GROQ_API_KEY,
            temperature=MODEL_TEMPERATURE,
            max_tokens=FALLBACK_MAX_TOKENS,
        )

    # Collect all tools  (Issue #1)
    tools = get_all_tools()

    # Build prompt  (Issue #2 — tuned system prompt)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Create the agent
    agent = create_tool_calling_agent(llm, tools, prompt)

    # Load memory  (Issue #5 — safe memory loading)
    memory = load_memory()

    # Create executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=5,
        return_intermediate_steps=True,
    )

    return agent_executor, memory


# ══════════════════════════════════════════════
# Main loop
# ══════════════════════════════════════════════
def main():
    """Main entry point for the Igris AI Agent."""
    print_banner()
    print_skills()

    print("  Loading memory and initializing agent...")
    agent_executor, memory = create_agent()
    print("  ⚔️  Igris stands at thy command, Your Majesty!\n")

    print("  Commands:")
    print("    • Type your message to chat")
    print("    • 'skills' — list available skills")
    print("    • 'clear'  — clear conversation history")
    print("    • 'quit'   — save memory and exit")
    print("  " + "═" * 45)
    print()

    while True:
        try:
            user_input = input("  Your Majesty: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Saving memory before exit...")
            save_memory(memory)
            print("  Igris: Fare thee well, Your Majesty. I await thy next summons.")
            break

        if not user_input:
            continue

        # ── Meta commands ──
        if user_input.lower() == "quit":
            print("  Saving thy memory before I rest...")
            save_memory(memory)
            print("  Igris: Fare thee well, Your Majesty. I await thy next summons.")
            break

        if user_input.lower() == "skills":
            print_skills()
            continue

        if user_input.lower() == "clear":
            memory.chat_memory.messages.clear()
            save_memory(memory)
            print("  Memory cleared. A fresh start, Your Majesty.\n")
            continue

        # ── Process through LangGraph workflow  (Issue #3) ──
        workflow_state = process_with_workflow(user_input)
        intent = workflow_state.get("intent", "chat")

        if intent != "chat":
            print(f"  [Intent: {intent}]")

        # ── Execute via agent  (Issues #1, #2, #3, #4) ──
        try:
            result = agent_executor.invoke({
                "input": user_input,
                "chat_history": memory.chat_memory.messages,
            })

            response = result.get("output", "I could not generate a response.")

            # Show intermediate tool usage if any
            steps = result.get("intermediate_steps", [])
            if steps:
                for action, observation in steps:
                    tool_name = action.tool if hasattr(action, "tool") else "unknown"
                    print(f"  [Tool: {tool_name}]")

            print(f"\n  Igris: {response}\n")

        except Exception as e:
            error_msg = str(e)
            if "rate_limit" in error_msg.lower() or "429" in error_msg:
                print("  Igris: The Groq servers are overwhelmed. Please wait a moment and try again.")
            elif "authentication" in error_msg.lower() or "401" in error_msg:
                print("  Igris: API key is invalid. Please check your GROQ_API_KEY.")
            else:
                print(f"  Igris: My liege, an error occurred — {e}")
            print()

        # ── Real-time memory save  (Issue #5) ──
        save_exchange(memory)


if __name__ == "__main__":
    main()
