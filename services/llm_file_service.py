# services/llm_file_service.py
import re
from services.file_service import (
    open_file,
    delete_file,
    create_file,
    find_files_by_name,
    search_in_cache,
)


# ---------------------------------------------------------------------------
# Intent detection — no LLM needed, regex is fast and reliable for this task
# ---------------------------------------------------------------------------

# Action keyword groups
_OPEN_WORDS = r"\b(open|launch|start|show|view|display|run|access|load)\b"
_DELETE_WORDS = r"\b(delete|remove|trash|erase|get rid of|wipe)\b"
_CREATE_WORDS = r"\b(create|make|new|generate|touch|add)\b"
_SEARCH_WORDS = r"\b(find|search|locate|look for|where is|where are|list)\b"

# Patterns that strongly suggest a FILE is being referenced
# e.g. "notes.txt", "my resume", "the file called X", "a file named X"
_FILE_HINT = r"(\.\w{2,5}|(?:file|document|folder|photo|image|video)\s+(?:called|named|titled)?\s*\S+|my\s+\S+)"


def _extract_filename(text: str) -> str | None:
    """
    Extract a filename from a natural language message.
    Handles:
      - "Open Muhammd_Talha_Resume.pdf"   → "Muhammd_Talha_Resume.pdf"
      - "Open Talha DMC.pdf"              → "Talha DMC.pdf"
      - "Delete notes.txt"                → "notes.txt"
      - "Find my resume"                  → "resume"
      - bare "README.md"                  → "README.md"
    """
    # Step 1: strip leading action verb + optional articles so we work on just the name part
    # e.g. "Open the file Muhammd_Talha_Resume.pdf" → "Muhammd_Talha_Resume.pdf"
    cleaned = re.sub(
        r"^(?:open|delete|remove|trash|erase|create|make|find|search|locate|launch|show|view|look\s+for)\s+"
        r"(?:the\s+|my\s+|a\s+|me\s+)?(?:file\s+)?",
        "",
        text.strip(),
        flags=re.I,
    )

    # Step 2: if what remains contains an extension, grab everything up to that extension
    # This correctly handles spaces and underscores: "Talha DMC.pdf" or "Muhammd_Talha_Resume.pdf"
    m = re.search(r"^([\w\-. ]+?\.\w{2,5})\b", cleaned)
    if m:
        return m.group(1).strip()

    # Step 3: quoted string anywhere in original text — e.g. open "my notes"
    m = re.search(r'["\']([^"\']+)["\']', text)
    if m:
        candidate = m.group(1).strip()
        candidate = re.sub(
            r"^(?:open|delete|remove|create|make|find|search|launch|show|view)\s+",
            "",
            candidate,
            flags=re.I,
        )
        return candidate if candidate else None

    # Step 4: "file/document called/named X"
    m = re.search(
        r'(?:file|document|folder)\s+(?:called|named|titled)\s+"?([^"]+?)"?\s*$',
        text,
        re.I,
    )
    if m:
        return m.group(1).strip()

    # Step 5: fallback — everything after the action verb (no extension found)
    m = re.search(
        r"(?:open|delete|remove|create|make|find|search|launch|show|view)\s+"
        r"(?:the\s+|my\s+|a\s+)?([A-Za-z0-9_\-. ]{2,60}?)(?:\s+file|\s+document|$)",
        text,
        re.I,
    )
    if m:
        candidate = m.group(1).strip()
        if candidate.lower() not in (
            "file",
            "document",
            "folder",
            "me",
            "it",
            "this",
            "",
        ):
            return candidate

    return None


def is_file_operation_request(text: str) -> tuple[bool, float]:
    """
    Determine whether the user's message is a file operation request.
    Returns (is_file_op: bool, confidence: float).

    Rules:
    - Action word + file noun/extension  → confident yes
    - Bare filename with extension (e.g. user replied "README.md") → yes if pending
    - Generic sentences like "create a poem" → no
    """
    t = text.strip().lower()

    has_action = bool(
        re.search(
            r"\b(?:open|launch|start|delete|remove|trash|erase|create|make|find|search|locate|look for|where is)\b",
            t,
        )
    )

    # Hard file reference: has a file extension
    has_extension = bool(re.search(r"\.\w{2,5}\b", t))

    # File noun: word "file", "document", "folder" etc.
    has_file_noun = bool(
        re.search(r"\b(?:file|document|folder|photo|image|video)\b", t)
    )

    # Case 1: action + (extension or file noun) → definite file op
    if has_action and (has_extension or has_file_noun):
        return True, 0.9

    # Case 2: bare filename typed alone (user is replying with just "README.md")
    # Only treat as file op if the entire message looks like a filename
    if has_extension and re.fullmatch(r"[\w\-. ]+\.\w{2,5}", t.strip()):
        return True, 0.85

    return False, 0.0


def parse_user_intent(text: str) -> dict:
    """
    Parse the user's message and return a structured intent dict.
    Uses regex — fast, offline, no Ollama dependency.
    """
    t = text.lower().strip()

    if re.search(_DELETE_WORDS, t):
        action = "delete"
    elif re.search(_OPEN_WORDS, t):
        action = "open"
    elif re.search(_CREATE_WORDS, t):
        action = "create"
    elif re.search(_SEARCH_WORDS, t):
        action = "search"
    else:
        # No action word — if the whole message looks like a filename, default to open
        if re.fullmatch(r"[\w\-. ]+\.\w{2,5}", t.strip()):
            action = "open"
        else:
            action = "unknown"

    filename = _extract_filename(text)  # preserve original casing

    confidence = (
        0.9
        if (action != "unknown" and filename)
        else 0.6 if action != "unknown" else 0.0
    )

    return {
        "action": action,
        "filename": filename,
        "confidence": confidence,
        "reasoning": f"Detected action='{action}', filename='{filename}'",
    }


# ---------------------------------------------------------------------------
# Smart file finder — tries multiple variants so spaces vs underscores don't matter
# ---------------------------------------------------------------------------


def _smart_find(filename: str, session_id=None) -> list:
    """
    Search for a file trying multiple name variants:
      1. Exact name as given          ("Talha DMC.pdf")
      2. Spaces replaced by _         ("Talha_DMC.pdf")
      3. _ replaced by spaces         ("Muhammd Talha Resume.pdf")
      4. Each word fragment separately (finds "Muhammd_Talha_Resume.pdf" from "Talha")
    Returns a deduplicated list of matching paths.
    """
    seen = set()
    found = []

    def _add(paths):
        for p in paths:
            if p not in seen:
                seen.add(p)
                found.append(p)

    # Split name and extension
    if "." in filename:
        dot_idx = filename.rfind(".")
        name_part = filename[:dot_idx]
        ext_part = filename[dot_idx:]  # includes the dot, e.g. ".pdf"
    else:
        name_part = filename
        ext_part = ""

    # Build variants
    variants = set()
    variants.add(filename)  # original
    variants.add(name_part.replace(" ", "_") + ext_part)  # spaces -> underscore
    variants.add(name_part.replace("_", " ") + ext_part)  # underscore -> spaces

    for v in variants:
        r = find_files_by_name(v, session_id=None)
        _add(r["files"])

    # If still nothing, try each word fragment individually
    if not found:
        words = re.split(r"[\s_]+", name_part)
        for word in words:
            if len(word) >= 3:
                r = find_files_by_name(word + ext_part, session_id=None)
                _add(r["files"])

    return found


# ---------------------------------------------------------------------------
# Main handler
# ---------------------------------------------------------------------------


def handle_llm_file_command(user_prompt: str, session_id=None) -> dict:
    """
    Interpret a natural language file request and execute the appropriate
    file_service function.

    Returns:
        status  : "success" | "error" | "select" | "confirm" | "ask_location" | "clarify"
        message : Human-readable text to show in the chat bubble
        action  : Inferred action string
        data    : Extra payload (files list, filename, etc.)
    """
    intent = parse_user_intent(user_prompt)

    if intent["confidence"] < 0.6 or intent["action"] == "unknown":
        return {
            "status": "clarify",
            "message": (
                "🤔 I'm not sure which file operation you want.\n\n"
                "You can say things like:\n"
                "  • 'Open resume.pdf'\n"
                "  • 'Delete notes.txt'\n"
                "  • 'Create report.docx'\n"
                "  • 'Find photo.jpg'"
            ),
            "action": None,
            "confidence": intent["confidence"],
        }

    action = intent["action"]
    filename = intent["filename"]

    if not filename:
        return {
            "status": "error",
            "message": (
                f"❌ I understand you want to **{action}** a file, "
                "but I couldn't work out the filename.\n\n"
                "Please include the filename, e.g.:\n"
                f"  • '{action.capitalize()} notes.txt'"
            ),
            "action": action,
        }

    # ---- OPEN ----
    if action == "open":
        cache_matches = search_in_cache(session_id, filename) if session_id else []

        if cache_matches:
            if len(cache_matches) == 1:
                result = open_file(cache_matches[0], session_id)
                return {
                    "status": result["status"],
                    "message": result["message"],
                    "action": "open",
                }
            return _multi_select_response(cache_matches, "open", filename)

        files = _smart_find(filename, session_id)
        if not files:
            return {
                "status": "error",
                "message": f"❌ '{filename}' not found on any drive.",
                "action": "open",
            }
        if len(files) == 1:
            result = open_file(files[0], session_id)
            return {
                "status": result["status"],
                "message": result["message"],
                "action": "open",
            }
        return _multi_select_response(files[:20], "open", filename)

    # ---- DELETE ----
    elif action == "delete":
        cache_matches = search_in_cache(session_id, filename) if session_id else []

        if cache_matches:
            if len(cache_matches) == 1:
                return _delete_confirm(cache_matches[0])
            return _multi_select_response(cache_matches, "delete", filename)

        files = _smart_find(filename, session_id)
        if not files:
            return {
                "status": "error",
                "message": f"❌ '{filename}' not found on any drive.",
                "action": "delete",
            }
        if len(files) == 1:
            return _delete_confirm(files[0])
        return _multi_select_response(files[:20], "delete", filename)

    # ---- CREATE ----
    elif action == "create":
        return {
            "status": "ask_location",
            "message": (
                f"📝 Where should I create **'{filename}'**?\n\n"
                "  1️⃣  Desktop (default)\n"
                "  2️⃣  Custom path\n\n"
                "Type **1**, **2**, or **cancel**"
            ),
            "action": "create",
            "data": {"filename": filename, "operation": "create"},
        }

    # ---- SEARCH ----
    elif action == "search":
        files = _smart_find(filename, session_id)
        if not files:
            return {
                "status": "error",
                "message": f"❌ No files matching '{filename}' found.",
                "action": "search",
            }

        files_list = "\n".join(f"  {i}. {f}" for i, f in enumerate(files[:20], 1))
        extra = f"\n  … and {len(files) - 20} more" if len(files) > 20 else ""
        return {
            "status": "success",
            "message": f"🔍 Found {len(files)} file(s) matching '{filename}':\n\n{files_list}{extra}",
            "action": "search",
            "data": {"files": files, "count": len(files)},
        }

    return {
        "status": "error",
        "message": f"❌ Unknown action: {action}",
        "action": action,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _delete_confirm(filepath: str) -> dict:
    return {
        "status": "confirm",
        "message": f"🗑️ Delete this file?\n📂 {filepath}\n\nType **yes** to confirm or **no** to cancel",
        "action": "delete",
        "data": {"files": [filepath], "operation": "delete"},
    }


def _multi_select_response(files: list, operation: str, filename: str) -> dict:
    numbered = "\n".join(f"  {i}. {f}" for i, f in enumerate(files, 1))
    return {
        "status": "select",
        "message": (
            f"📂 Found {len(files)} file(s) matching '{filename}'.\n\n"
            f"{numbered}\n\n"
            f"Enter the number to {operation}, or **cancel**"
        ),
        "action": operation,
        "data": {"files": files, "operation": operation, "filename": filename},
    }


# ---------------------------------------------------------------------------
# Follow-up response processor (user replies to a pending prompt)
# ---------------------------------------------------------------------------


def process_file_response(response_text: str, pending_action: dict) -> dict:
    """
    Handle the user's reply to a multi-step file prompt
    (file selection, delete confirmation, location choice, custom path).
    """
    state = pending_action.get("state", "select")
    files = pending_action.get("files", [])
    operation = pending_action.get("operation", "")
    r = response_text.strip().lower()

    # ---- File selection ----
    if state == "select":
        if r in ("cancel", "c", "no"):
            return {
                "status": "success",
                "message": "❌ Operation cancelled.",
                "handled": True,
            }
        try:
            choice = int(r)
            if 1 <= choice <= len(files):
                selected = files[choice - 1]
                if operation == "open":
                    result = open_file(selected)
                    return {
                        "status": result["status"],
                        "message": result["message"],
                        "action": "open",
                        "handled": True,
                    }
                elif operation == "delete":
                    return {
                        "status": "confirm",
                        "message": f"🗑️ Delete this file?\n📂 {selected}\n\nType **yes** to confirm or **no** to cancel",
                        "action": "delete_confirm",
                        "data": {"file": selected},
                        "handled": True,
                    }
            return {
                "status": "error",
                "message": f"❌ Please enter a number between 1 and {len(files)}.",
                "handled": False,
            }
        except ValueError:
            return {
                "status": "error",
                "message": "❌ Invalid input — please enter a number or 'cancel'.",
                "handled": False,
            }

    # ---- Delete confirmation ----
    elif state == "delete_confirm":
        if r in ("yes", "y"):
            file_to_delete = pending_action.get("file")
            result = delete_file(file_to_delete)
            return {
                "status": result["status"],
                "message": result["message"],
                "action": "delete",
                "handled": True,
            }
        elif r in ("no", "n", "cancel"):
            return {
                "status": "success",
                "message": "❌ Delete cancelled.",
                "handled": True,
            }
        return {
            "status": "error",
            "message": "❌ Please type **yes** or **no**.",
            "handled": False,
        }

    # ---- Create — choose location ----
    elif state == "location":
        filename = pending_action.get("filename", "")
        if r in ("1", "desktop"):
            result = create_file(filename)
            return {
                "status": result["status"],
                "message": result["message"],
                "action": "create",
                "handled": True,
            }
        elif r in ("2", "custom"):
            return {
                "status": "ask_custom_path",
                "message": "📁 Enter the full path where you want to create the file (or **cancel**):",
                "action": "create_custom",
                "handled": True,
            }
        elif r in ("cancel", "c"):
            return {
                "status": "success",
                "message": "❌ Creation cancelled.",
                "handled": True,
            }
        return {
            "status": "error",
            "message": "❌ Please enter **1** for Desktop, **2** for custom path, or **cancel**.",
            "handled": False,
        }

    # ---- Create — custom path provided ----
    elif state == "custom_path":
        if r in ("cancel", "c"):
            return {
                "status": "success",
                "message": "❌ Creation cancelled.",
                "handled": True,
            }
        filename = pending_action.get("filename", "")
        result = create_file(filename, custom_path=response_text.strip())
        return {
            "status": result["status"],
            "message": result["message"],
            "action": "create",
            "handled": True,
        }

    return {
        "status": "error",
        "message": "❌ Unexpected state. Please try your request again.",
        "handled": False,
    }
