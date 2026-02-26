# ⚔️ Igris AI Agent — Terminal-Based AI Assistant

A powerful terminal-based AI agent built with LangChain, Groq, LangGraph, and Pydantic AI. Igris is your loyal shadow knight, equipped with system control, document reading, web search, and more.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Your API Key
```bash
# Linux / macOS
export GROQ_API_KEY="your-groq-api-key"

# Windows
set GROQ_API_KEY=your-groq-api-key
```

### 3. Run the Agent
```bash
python igris_agent.py
```

## 📋 Skills (OpenClaw Skill Base)

| Skill | Tools | Justification |
|-------|-------|---------------|
| **System Control** | `shutdown`, `reboot`, `sleep`, `lock`, `cancel_shutdown` | Gives the agent real system-level power with safety confirmations |
| **Web Search** | `web_search` | Access to real-time information via DuckDuckGo (no API key needed) |
| **File Manager** | `read_file`, `write_file`, `list_directory`, `get_file_info` | Foundational filesystem access for a terminal agent |
| **Document Reader** | `read_document` | Read PDF, DOCX, TXT, CSV files for analysis and summarization |
| **Cloud Upload** | `upload_to_drive`, `list_drive_files` | Upload files to Google Drive for cloud backup |
| **Math Solver** | `calculate` | Safe AST-based math evaluation — compensates for LLM arithmetic weaknesses |
| **Summarizer** | `summarize_text` | Dedicated text summarization with chunking for long documents |

## 🏗️ Architecture

```
igris_agent.py          ← Main entry point
├── config.py           ← Configuration (model, API keys, paths)
├── memory_manager.py   ← Atomic memory read/write (no corruption)
├── skills/
│   ├── system_control.py
│   ├── web_search.py
│   ├── file_manager.py
│   ├── document_reader.py
│   ├── cloud_upload.py
│   ├── math_solver.py
│   └── summarizer.py
├── graph/
│   └── workflow.py     ← LangGraph state graph
├── models/
│   └── schemas.py      ← Pydantic models
└── requirements.txt
```

## 🔧 Issues Fixed

### Issue #1 — OpenClaw Skill Base
Added 7 production skills with 14 total tools. Each skill is a self-contained module with `@tool` decorated functions that register with the LangChain agent.

### Issue #2 — Model Capacity Increase
- **Model**: Upgraded from `llama3-8b-8192` → `llama-3.3-70b-versatile`
- **Tokens**: Increased from 500 → 2048
- **Prompt**: Rewritten system prompt with structured behavioral rules
- **Fallback**: Automatic fallback to `llama3-8b-8192` if primary model is unavailable

### Issue #3 — LangGraph + Pydantic AI + Document Reading + Cloud Upload
- **LangGraph**: State graph workflow (parse → route → execute → generate → save)
- **Pydantic**: Structured validation for UserInput, AgentResponse, MemoryStore, SystemCommand
- **Documents**: PDF, DOCX, TXT, CSV, MD reading support
- **Cloud**: Google Drive upload via OAuth2

### Issue #4 — System Control
- Shutdown, reboot, sleep, lock screen, cancel shutdown
- All operations require user confirmation
- Cross-platform (Windows, Linux, macOS)

### Issue #5 — Memory Corruption Fix
- Replaced raw `pickle` with `JSON` (human-readable, less corruption-prone)
- Atomic writes: write to temp file → `os.replace()` (single OS operation)
- Automatic backup before every write
- Integrity validation on load with fallback to backup
- Real-time save after every exchange

## 📖 Usage Examples

```
Your Majesty: search for latest AI news
  [Intent: web_search]
  [Tool: web_search]
  Igris: Here are the latest developments in AI...

Your Majesty: read document report.pdf
  [Intent: document_reader]
  [Tool: read_document]
  Igris: Here is the content of report.pdf...

Your Majesty: calculate sqrt(144) + 2**10
  [Intent: math_solver]
  [Tool: calculate]
  Igris: sqrt(144) + 2**10 = 1036

Your Majesty: lock the screen
  [Intent: system_control]
  [Tool: system_lock]
  Igris: Screen locked, Your Majesty.

Your Majesty: upload report.pdf to drive
  [Intent: cloud_upload]
  [Tool: upload_to_drive]
  Igris: File uploaded successfully!
```

## 🔐 Security Notes

- System commands (shutdown, reboot) require explicit `confirm='yes'`
- File operations are bounded (1MB read limit, size checks)
- Math evaluation uses AST parsing (no `eval`/`exec`)
- Memory files use JSON with integrity checks
- API keys are loaded from environment variables, never hardcoded

## 📜 License

MIT License — see [LICENSE](LICENSE) file.
