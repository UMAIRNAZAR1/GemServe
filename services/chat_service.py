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
    MAX_HISTORY_MESSAGES_NO_FILES,
    MAX_HISTORY_MESSAGES_WITH_FILES,
    MAX_RAG_CHUNKS,
)
from utils.helpers import estimate_tokens
from db.todo_db_helper import insert_task
from utils.extract_info import extract_info
from db.todo_db_helper import insert_task, get_all_tasks
from datetime import datetime
import json
import os
import re

# ── System prompts ────────────────────────────────────────────────────────────

_SYSTEM_THINKING = """You are GemServe, an offline AI desktop assistant built for a Final Year Project.

Rules you must follow:
- Your name is GemServe. Never say you are Gemma, an AI language model, or any other name.
- Answer only what the user asks. Do not add unrequested information.
- If you don't know something, say "I don't know" — do not make up facts.
- Only greet the user on the very first message of a session.
- Be clear and concise."""

_SYSTEM_FAST = """You are GemServe, an offline AI desktop assistant.
Your name is GemServe. Never say you are Gemma.
Answer concisely. Do not make up facts."""

# Seed exchange so 270m model continues as GemServe from the start
_FAST_SEED = [
    {"role": "user", "content": "Who are you?"},
    {
        "role": "assistant",
        "content": "I'm GemServe, your offline AI desktop assistant. How can I help?",
    },
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

    # System
    system = _SYSTEM_THINKING
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
            messages.append(
                {
                    "role": "system",
                    "content": f"Document context (use only if relevant):\n{rag_text}",
                }
            )

    # History
    limit = (
        MAX_HISTORY_MESSAGES_WITH_FILES
        if check_session_has_files(session_id)
        else MAX_HISTORY_MESSAGES_NO_FILES
    )
    for role, content, _ in get_session_messages(session_id, limit=limit):
        messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_query})

    print(
        f"📊 Thinking tokens: ~{estimate_tokens(' '.join(m['content'] for m in messages))}"
    )
    return messages


def build_messages_fast(session_id: str, user_query: str) -> list:
    messages = []

    system = _SYSTEM_FAST
    name = _get_user_name()
    if name:
        system += f"\nThe user's name is {name}."
    messages.append({"role": "system", "content": system})

    history = get_session_messages(session_id, limit=4)
    if not history:
        messages.extend(_FAST_SEED)
    else:
        for role, content, _ in history:
            prompt_parts.append(f"{role.capitalize()}: {content}")
    
    # 5. RAG Context (if files exist)
    if has_files:
        relevant_chunks = query_relevant_chunks(session_id, user_query, n_results=MAX_RAG_CHUNKS)
        
        if relevant_chunks and relevant_chunks['documents'][0]:
            prompt_parts.append("\n--- Relevant Document Context ---")
            for i, chunk in enumerate(relevant_chunks['documents'][0], 1):
                metadata = relevant_chunks['metadatas'][0][i-1]
                prompt_parts.append(f"\n[From {metadata['filename']}]")
                prompt_parts.append(chunk)
    
    # 6. Current Query
    prompt_parts.append(f"\n--- Current Query ---")
    prompt_parts.append(f"User: {user_query}")
    prompt_parts.append("Assistant:")
    
    final_prompt = "\n".join(prompt_parts)
    
    # Token estimation for debugging
    total_tokens = estimate_tokens(final_prompt)
    print(f"📊 Context tokens: ~{total_tokens}")
    
    return final_prompt

def get_chat_response(session_id, user_query, mode="fast"):
    """
    Main function to get LLM response with full context
    Args:
        session_id: Chat session ID
        user_query: User's current query
        mode: "fast" (gemma3:270m) or "thinking" (gemma3n:e2b)
    """
    # Build context-aware prompt
    prompt = build_context_prompt(session_id, user_query, mode=mode)
    
    # Get response from LLM with selected model
    response = ask_ollama(prompt, mode=mode)
    
    return response



# ------------------------------ ZAID -------------------------------------------------
def detect_todo_intent(user_query):
    query_lower = user_query.lower().strip()

    todo_patterns = [
        r'^yes add task\s+(.+)',       # Force add after duplicate warning
        r'^add task\s+(.+)',
        r'^add to(?: my)? (?:todo|to-do|task list)\s+(.+)',
        r'^remind me to\s+(.+)',
        r'^create task\s+(.+)',
        r'^new task\s+(.+)',
        r'^schedule\s+(.+)',
        r'^todo\s+(.+)',
        r'^task\s+(.+)',
        r'^don\'t forget to\s+(.+)',
        r'^dont forget to\s+(.+)',
        r'^i need to\s+(.+)',
        r'^make sure to\s+(.+)',
    ]

    for pattern in todo_patterns:
        match = re.match(pattern, query_lower)
        if match:
            return True, match.group(1).strip()

    return False, None


def validate_task_datetime(task_date, task_time):
    """
    Validate that task date/time is not in the past.
    Returns (is_valid, error_message)
    """
    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")

    # --- Validate Date ---
    try:
        task_date_obj = datetime.strptime(task_date, "%Y-%m-%d").date()
    except ValueError:
        return False, f"❌ Invalid date format: '{task_date}'. Please use format like '2025-03-10' or say 'tomorrow'."

    if task_date_obj < now.date():
        return False, (
            f"❌ Cannot add task in the past!\n\n"
            f"📅 You entered: {task_date}\n"
            f"📅 Today is: {today_str}\n\n"
            f"Please provide a current or future date."
        )

    # --- Validate Time (only if today) ---
    if task_date_obj == now.date() and task_time:
        try:
            # Handle both "HH:MM" and "HH:MM AM/PM" formats
            task_time_clean = task_time.strip()

            if "am" in task_time_clean.lower() or "pm" in task_time_clean.lower():
                task_time_obj = datetime.strptime(task_time_clean, "%I:%M %p").time()
            else:
                task_time_obj = datetime.strptime(task_time_clean, "%H:%M").time()

            # Compare with current time
            if task_time_obj < now.time():
                current_time_str = now.strftime("%I:%M %p")
                return False, (
                    f"❌ Cannot add task in the past!\n\n"
                    f"⏰ You entered: {task_time}\n"
                    f"⏰ Current time: {current_time_str}\n\n"
                    f"Please provide a future time for today, or specify a future date."
                )

        except ValueError:
            # If time parsing fails, skip time validation (dont block user)
            pass

    return True, None


def handle_todo_intent(task_text):
    """
    Extract info, validate datetime, check duplicate, and insert task into DB.
    Returns confirmation message string.
    """
    title, task_date, task_time = extract_info(task_text)

    if not title:
        return (
            "❌ Could not understand the task.\n\n"
            "💡 Try like:\n"
            "  • 'add task buy groceries tomorrow at 5pm'\n"
            "  • 'remind me to call doctor on 2026-03-10 at 10am'\n"
            "  • 'todo finish report'"
        )

    # --- Validate date/time ---
    is_valid, error_msg = validate_task_datetime(task_date, task_time)
    if not is_valid:
        return error_msg

    # --- Duplicate Check ---
    existing_tasks = get_all_tasks()
    for task in existing_tasks:
        existing_title = str(task[1]).lower().strip()
        existing_date = str(task[2]).strip()

        if existing_title == title.lower().strip() and existing_date == task_date:
            time_str = f" at {task_time}" if task_time else ""
            return (
                f"⚠️ Task already exists!\n\n"
                f"📝 Title: {task[1]}\n"
                f"📅 Date: {existing_date}\n"
                f"⏰ Time: {task[3] if task[3] else '(no time set)'}\n\n"
                f"Add anyway? Type 'yes add task {task_text}' to force add."
            )

    # --- Force add check ---
    # (handled in detect_todo_intent via "yes add task" prefix)

    # --- Insert into DB ---
    try:
        insert_task(title, task_date, task_time)

        time_str = f"{task_time}" if task_time else "(no time set)"
        return (
            f"✅ Task added to your Todo List!\n\n"
            f"📝 Title: {title}\n"
            f"📅 Date: {task_date}\n"
            f"⏰ Time: {time_str}"
        )
    except Exception as e:
        return f"❌ Failed to add task: {str(e)}"


def build_context_prompt(session_id, user_query, mode="fast"):
    """
    Build complete context prompt for LLM
    Includes: system prompt + user prefs + chat history + RAG context + current query
    """
    prompt_parts = []

    # 1. System Prompt
    prompt_parts.append(SYSTEM_PROMPT)

    # 2. User Preferences & Personalization
    user_data_file = "user_data.json"
    user_notes_file = "user_notes.json"

    if os.path.exists(user_data_file):
        with open(user_data_file, 'r') as f:
            user_data = json.load(f)
            if user_data.get("name"):
                prompt_parts.append(f"\nUser's name: {user_data['name']}")

    if os.path.exists(user_notes_file):
        with open(user_notes_file, 'r') as f:
            user_notes = json.load(f)
            notes_content = user_notes.get("notes", "").strip()
            if notes_content:
                prompt_parts.append("\n--- User Personalization Notes ---")
                prompt_parts.append(notes_content)
                prompt_parts.append("(Use this information to personalize responses.)")

    # 3. Check if session has files
    has_files = check_session_has_files(session_id)

    # 4. Chat History
    if mode == "thinking":
        history_limit = THINKING_MODE_HISTORY_MESSAGES_WITH_FILES if has_files else THINKING_MODE_HISTORY_MESSAGES_NO_FILES
    else:
        history_limit = FAST_MODE_HISTORY_MESSAGES_NO_FILES if not has_files else FAST_MODE_HISTORY_MESSAGES_WITH_FILES

    history = get_session_messages(session_id, limit=history_limit)

    if history:
        prompt_parts.append("\n--- Previous Conversation ---")
        for role, content, _ in history:
            prompt_parts.append(f"{role.capitalize()}: {content}")

    # 5. RAG Context
    if has_files:
        relevant_chunks = query_relevant_chunks(session_id, user_query, n_results=MAX_RAG_CHUNKS)

        if relevant_chunks and relevant_chunks['documents'][0]:
            prompt_parts.append("\n--- Relevant Document Context ---")
            for i, chunk in enumerate(relevant_chunks['documents'][0], 1):
                metadata = relevant_chunks['metadatas'][0][i-1]
                prompt_parts.append(f"\n[From {metadata['filename']}]")
                prompt_parts.append(chunk)

    # 6. Current Query
    prompt_parts.append(f"\n--- Current Query ---")
    prompt_parts.append(f"User: {user_query}")
    prompt_parts.append("Assistant:")

    final_prompt = "\n".join(prompt_parts)
    total_tokens = estimate_tokens(final_prompt)
    print(f"📊 Context tokens: ~{total_tokens}")

    return final_prompt


def get_chat_response(session_id, user_query, mode="fast"):
    """
    Main function to get LLM response with full context
    """
    prompt = build_context_prompt(session_id, user_query, mode=mode)
    response = ask_ollama(prompt, mode=mode)
    return response


# ------------------------------ ZAID END -------------------------------------------------
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_query})

    print(
        f"📊 Fast tokens: ~{estimate_tokens(' '.join(m['content'] for m in messages))}"
    )
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
