"""
Igris AI Agent — Memory Manager  (Issue #5 fix)

Fixes:
- Replaces raw pickle with JSON for chat history (human-readable, less corruption-prone)
- Uses atomic writes: write to temp file → rename (prevents partial-write corruption)
- Adds backup before every write
- Adds integrity check on load
- Real-time update after every exchange
"""

import os
import json
import shutil
import tempfile
from datetime import datetime

from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage

from config import MEMORY_FILE, MEMORY_BACKUP_FILE, MEMORY_DIR


def _ensure_dirs():
    """Create the memory directory if it doesn't exist."""
    os.makedirs(MEMORY_DIR, exist_ok=True)


def _serialize_messages(messages):
    """Convert LangChain message objects to plain dicts for JSON storage."""
    serialized = []
    for msg in messages:
        entry = {
            "type": "human" if isinstance(msg, HumanMessage) else "ai",
            "content": msg.content,
            "timestamp": datetime.now().isoformat(),
        }
        serialized.append(entry)
    return serialized


def _deserialize_messages(data):
    """Convert stored dicts back to LangChain message objects."""
    messages = []
    for entry in data:
        if entry["type"] == "human":
            messages.append(HumanMessage(content=entry["content"]))
        else:
            messages.append(AIMessage(content=entry["content"]))
    return messages


def _validate_memory_data(data):
    """Check that loaded data has the expected structure."""
    if not isinstance(data, dict):
        return False
    if "chat_history" not in data:
        return False
    if not isinstance(data["chat_history"], list):
        return False
    for entry in data["chat_history"]:
        if not isinstance(entry, dict):
            return False
        if "type" not in entry or "content" not in entry:
            return False
        if entry["type"] not in ("human", "ai"):
            return False
    return True


def save_memory(memory: ConversationBufferMemory):
    """
    Atomically save conversation memory to disk.

    Steps:
    1. Serialize messages to JSON-safe dicts
    2. Write to a temporary file in the same directory
    3. If old memory exists, back it up
    4. Rename temp file to final path (atomic on same filesystem)
    """
    _ensure_dirs()

    messages = memory.chat_memory.messages
    data = {
        "chat_history": _serialize_messages(messages),
        "last_saved": datetime.now().isoformat(),
        "message_count": len(messages),
    }

    # Step 1: Write to temp file in same directory (ensures same filesystem for atomic rename)
    fd, tmp_path = tempfile.mkstemp(
        suffix=".json.tmp", prefix="igris_mem_", dir=MEMORY_DIR
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_f:
            json.dump(data, tmp_f, indent=2, ensure_ascii=False)
            tmp_f.flush()
            os.fsync(tmp_f.fileno())  # force write to disk
    except Exception:
        # Clean up temp file on failure — do NOT touch the real file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    # Step 2: Backup current file (if it exists)
    if os.path.exists(MEMORY_FILE):
        try:
            shutil.copy2(MEMORY_FILE, MEMORY_BACKUP_FILE)
        except Exception:
            pass  # backup failure is non-fatal

    # Step 3: Atomic rename — replaces old file in one OS operation
    try:
        os.replace(tmp_path, MEMORY_FILE)  # atomic on Windows & Unix
    except Exception:
        # If rename fails, clean up temp
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def load_memory() -> ConversationBufferMemory:
    """
    Load conversation memory from disk with integrity checking.

    Falls back to backup file if primary is corrupted.
    Returns a fresh memory if both are unavailable/corrupted.
    """
    _ensure_dirs()
    memory = ConversationBufferMemory(return_messages=True)

    # Try primary file first
    for filepath in [MEMORY_FILE, MEMORY_BACKUP_FILE]:
        if not os.path.exists(filepath):
            continue
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            if _validate_memory_data(data):
                messages = _deserialize_messages(data["chat_history"])
                memory.chat_memory.messages = messages
                count = data.get("message_count", len(messages))
                last_saved = data.get("last_saved", "unknown")
                print(f"Memory restored — {count} messages (last saved: {last_saved})")
                return memory
            else:
                print(f"Warning: Memory file {filepath} failed integrity check, trying backup...")
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Warning: Memory file {filepath} is corrupted ({e}), trying backup...")

    # If we get here, no valid memory was found
    print("Starting with fresh memory vault.")
    return memory


def save_exchange(memory: ConversationBufferMemory):
    """
    Save memory immediately after each exchange.
    Called in real-time to prevent data loss.
    """
    try:
        save_memory(memory)
    except Exception as e:
        print(f"Warning: Real-time memory save failed: {e}")
        print("Your conversation continues — memory will retry on next exchange.")
