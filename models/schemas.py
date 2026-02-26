"""
Igris AI Agent — Pydantic AI Schemas  (Issue #3 — Pydantic AI support)

Structured I/O validation using Pydantic models.
Ensures type safety and validation for all data flowing through the agent.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field, field_validator
from datetime import datetime


class UserInput(BaseModel):
    """Validated user input to the agent."""
    text: str = Field(..., min_length=1, max_length=10000, description="User's message text")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the message was sent")
    mode: Literal["chat", "translate", "recall", "command"] = Field(
        default="chat", description="Input processing mode"
    )

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Input text cannot be empty or only whitespace")
        return v.strip()


class AgentResponse(BaseModel):
    """Validated agent response."""
    content: str = Field(..., description="The response text from the agent")
    tool_used: Optional[str] = Field(None, description="Name of the tool used, if any")
    tool_result: Optional[str] = Field(None, description="Raw result from the tool, if any")
    timestamp: datetime = Field(default_factory=datetime.now)
    tokens_used: Optional[int] = Field(None, description="Number of tokens consumed")


class MemoryEntry(BaseModel):
    """A single conversation memory entry."""
    role: Literal["human", "ai"] = Field(..., description="Who said this")
    content: str = Field(..., description="The message content")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class MemoryStore(BaseModel):
    """The full memory store structure for validation."""
    chat_history: List[MemoryEntry] = Field(default_factory=list)
    last_saved: str = Field(default_factory=lambda: datetime.now().isoformat())
    message_count: int = Field(default=0, ge=0)

    @field_validator("message_count")
    @classmethod
    def count_matches_history(cls, v, info):
        history = info.data.get("chat_history", [])
        if history and v != len(history):
            return len(history)  # auto-correct
        return v


class DocumentInfo(BaseModel):
    """Metadata about a loaded document."""
    file_path: str = Field(..., description="Absolute path to the document")
    file_type: str = Field(..., description="Extension of the file")
    file_size: int = Field(..., ge=0, description="Size in bytes")
    page_count: Optional[int] = Field(None, description="Number of pages (for PDFs)")
    word_count: Optional[int] = Field(None, description="Approximate word count")
    content_preview: str = Field(default="", description="First 500 chars of content")


class SystemCommand(BaseModel):
    """Validated system command request."""
    action: Literal["shutdown", "reboot", "sleep", "lock", "cancel_shutdown"] = Field(
        ..., description="System action to perform"
    )
    confirmed: bool = Field(default=False, description="Whether the user confirmed the action")
    delay_seconds: int = Field(default=30, ge=0, le=3600, description="Delay before action")


class SkillInfo(BaseModel):
    """Information about an available skill."""
    name: str = Field(..., description="Skill identifier")
    description: str = Field(..., description="What the skill does")
    tools: List[str] = Field(default_factory=list, description="Tool names provided by this skill")
    enabled: bool = Field(default=True)


class AgentState(BaseModel):
    """
    The state object used by LangGraph workflow.
    Tracks the full lifecycle of a single user interaction.
    """
    user_input: str = Field(default="", description="Raw user input")
    parsed_input: Optional[UserInput] = Field(None, description="Validated input")
    intent: str = Field(default="chat", description="Detected intent / routing decision")
    tool_calls: List[str] = Field(default_factory=list, description="Tools that were called")
    tool_results: List[str] = Field(default_factory=list, description="Results from tool calls")
    response: str = Field(default="", description="Final response to user")
    error: Optional[str] = Field(None, description="Error message if something failed")
    should_save_memory: bool = Field(default=True, description="Whether to persist this exchange")
