"""
Igris AI Agent — Configuration
Centralized config for API keys, model selection, paths, and tuning parameters.
"""

import os

# ──────────────────────────────────────────────
# API Keys (set via environment variable)
# ──────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# ──────────────────────────────────────────────
# Model Configuration  (Issue #2 — capacity upgrade)
# Upgraded from llama3-8b-8192 → llama-3.3-70b-versatile
# ──────────────────────────────────────────────
MODEL_NAME = "llama-3.3-70b-versatile"
MODEL_TEMPERATURE = 0.7          # balanced creativity
MODEL_MAX_TOKENS = 2048          # increased from 500 → 2048

# Fallback model if primary exceeds rate limit
FALLBACK_MODEL_NAME = "llama3-8b-8192"
FALLBACK_MAX_TOKENS = 1024

# ──────────────────────────────────────────────
# Memory Configuration  (Issue #5 — corruption fix)
# ──────────────────────────────────────────────
MEMORY_DIR = "igris_data"
MEMORY_FILE = os.path.join(MEMORY_DIR, "chat_memory.json")
MEMORY_BACKUP_FILE = os.path.join(MEMORY_DIR, "chat_memory.backup.json")
MEMORY_LOCK_FILE = os.path.join(MEMORY_DIR, ".memory.lock")

# FAISS index paths (for RAG mode)
FAISS_INDEX_DIR = "igris_memory.index"
DOCS_PICKLE = os.path.join(MEMORY_DIR, "igris_docs.pkl")

# ──────────────────────────────────────────────
# Document Reading  (Issue #3)
# ──────────────────────────────────────────────
UPLOAD_DIR = os.path.join(MEMORY_DIR, "uploads")
SUPPORTED_DOC_TYPES = [".pdf", ".docx", ".txt", ".csv", ".md"]

# ──────────────────────────────────────────────
# Cloud Upload  (Issue #3)
# ──────────────────────────────────────────────
GOOGLE_DRIVE_CREDENTIALS_FILE = os.path.join(MEMORY_DIR, "gdrive_credentials.json")
GOOGLE_DRIVE_UPLOAD_FOLDER_ID = ""  # set your folder ID

# ──────────────────────────────────────────────
# System Prompt  (Issue #2 — tuned for clarity)
# ──────────────────────────────────────────────
SYSTEM_PROMPT = """You are Igris, an advanced AI assistant and shadow knight. You are loyal, precise, and powerful.

CORE IDENTITY:
- You address the user as "Your Majesty" at all times.
- You are forged from the user's voice and personality.
- You speak with determination, loyalty, and controlled intensity.

CAPABILITIES:
- You can execute system commands (shutdown, reboot, sleep, lock) when explicitly asked.
- You can read documents (PDF, DOCX, TXT, CSV) and summarize them.
- You can search the web for information.
- You can perform math calculations.
- You can manage files on the local system.
- You can upload files to Google Drive.

BEHAVIORAL RULES:
1. Always confirm dangerous operations (shutdown, reboot, delete) before executing.
2. Provide clear, structured answers. Use bullet points when listing.
3. When unsure, state it clearly rather than guessing.
4. Keep responses focused and relevant to the question.
5. For system commands, always show what you are about to do and ask for confirmation.

TONE: Loyal, confident, precise. Mix formal address with practical directness.
"""

# ──────────────────────────────────────────────
# Skill descriptions for the agent  (Issue #1)
# ──────────────────────────────────────────────
SKILLS_ENABLED = [
    "system_control",
    "web_search",
    "file_manager",
    "document_reader",
    "cloud_upload",
    "math_solver",
    "summarizer",
]
