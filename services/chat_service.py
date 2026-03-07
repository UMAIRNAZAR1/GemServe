# services/chat_service.py
import json
import os
from db.database import get_session_messages, check_session_has_files
from db.vector_store import query_relevant_chunks
from services.llm_service import (
    _call_ollama_chat,
    OLLAMA_FAST_MODEL,
    OLLAMA_THINKING_MODEL,
)
from utils.config import (
    SYSTEM_PROMPT,
    MAX_HISTORY_MESSAGES_NO_FILES,
    MAX_HISTORY_MESSAGES_WITH_FILES,
    MAX_RAG_CHUNKS,
)
from utils.helpers import estimate_tokens

# Seed exchange for the 270m fast model — anchors GemServe identity on fresh sessions
_FAST_SEED = [
    {"role": "user",      "content": "Who are you?"},
    {"role": "assistant", "content": "I'm GemServe, your offline AI desktop assistant. How can I help?"},
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_user_name() -> str | None:
    try:
        with open("user_data.json", "r") as f:
            return json.load(f).get("name")
    except Exception:
        return None


def _get_user_notes() -> str | None:
    try:
        with open("user_notes.json", "r") as f:
            return json.load(f).get("notes", "").strip() or None
    except Exception:
        return None


# ── Message builders ──────────────────────────────────────────────────────────

def build_messages_thinking(session_id: str, user_query: str) -> list:
    messages = []

    # System prompt from config
    system = SYSTEM_PROMPT
    name = _get_user_name()
    if name:
        system += f"\nThe user's name is {name}."
    notes = _get_user_notes()
    if notes:
        system += f"\nUser notes: {notes}"
    messages.append({"role": "system", "content": system})

    # RAG context
    if check_session_has_files(session_id):
        chunks = query_relevant_chunks(session_id, user_query, n_results=MAX_RAG_CHUNKS)
        if chunks and chunks["documents"][0]:
            rag_text = "\n\n".join(
                f"[From {chunks['metadatas'][0][i]['filename']}]\n{chunk}"
                for i, chunk in enumerate(chunks["documents"][0])
            )
            messages.append({"role": "system", "content": f"Document context:\n{rag_text}"})

    # Chat history
    limit = MAX_HISTORY_MESSAGES_WITH_FILES if check_session_has_files(session_id) else MAX_HISTORY_MESSAGES_NO_FILES
    for role, content, _ in get_session_messages(session_id, limit=limit):
        messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_query})

    print(f"📊 Thinking tokens: ~{estimate_tokens(' '.join(m['content'] for m in messages))}")
    return messages


def build_messages_fast(session_id: str, user_query: str) -> list:
    messages = []

    # System prompt from config
    system = SYSTEM_PROMPT
    name = _get_user_name()
    if name:
        system += f"\nThe user's name is {name}."
    messages.append({"role": "system", "content": system})

    # Seed on fresh session, history otherwise
    history = get_session_messages(session_id, limit=4)
    if not history:
        messages.extend(_FAST_SEED)
    else:
        for role, content, _ in history:
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_query})

    print(f"📊 Fast tokens: ~{estimate_tokens(' '.join(m['content'] for m in messages))}")
    return messages


# ── Public API ────────────────────────────────────────────────────────────────

def get_chat_response(session_id, user_query: str, mode: str = "fast") -> str:
    if mode == "thinking":
        messages = build_messages_thinking(session_id, user_query)
        model, timeout = OLLAMA_THINKING_MODEL, 180
    else:
        messages = build_messages_fast(session_id, user_query)
        model, timeout = OLLAMA_FAST_MODEL, 60

    return _call_ollama_chat(messages, model, timeout)


# Backward-compat alias
def build_context_prompt(session_id, user_query: str) -> str:
    messages = build_messages_thinking(session_id, user_query)
    return "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in messages)