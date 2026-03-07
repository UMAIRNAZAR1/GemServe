# services/llm_service.py
import requests
from utils.config import OLLAMA_BASE_URL, OLLAMA_FAST_MODEL, OLLAMA_THINKING_MODEL

# Backward-compat: some imports expect OLLAMA_MODEL
try:
    from utils.config import OLLAMA_MODEL
except ImportError:
    OLLAMA_MODEL = OLLAMA_FAST_MODEL

_FALLBACK = (
    "I'm not sure how to respond to that. Could you rephrase or ask something else?"
)


def _call_ollama_chat(
    messages: list, model: str, timeout: int = 60, retries: int = 2
) -> str:
    """
    Call Ollama /api/chat with a proper messages array.
    Correctly separates system prompt from conversation so the model
    does NOT re-introduce itself on every reply.

    messages format:
        [{"role": "system",    "content": "..."},
         {"role": "user",      "content": "..."},
         {"role": "assistant", "content": "..."},
         {"role": "user",      "content": "current query"}]
    """
    url = f"{OLLAMA_BASE_URL}/api/chat"
    data = {"model": model, "messages": messages, "stream": False}

    for attempt in range(1, retries + 1):
        try:
            response = requests.post(url, json=data, timeout=timeout)
            response.raise_for_status()
            text = response.json().get("message", {}).get("content", "").strip()
            if text:
                return text
            print(f"⚠️ Empty response from {model} (attempt {attempt}/{retries})")
        except requests.exceptions.ConnectionError:
            return "❌ Cannot connect to Ollama. Make sure Ollama is running."
        except requests.exceptions.Timeout:
            return "❌ Request timed out. The model might be loading — try again."
        except Exception as e:
            return f"❌ Error: {str(e)}"

    return _FALLBACK


def _call_ollama(prompt: str, model: str, timeout: int = 60, retries: int = 2) -> str:
    """
    Legacy /api/generate caller — kept for backward compatibility.
    Used by ask_ollama() and file intent parsing.
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"
    data = {"model": model, "prompt": prompt, "stream": False}

    for attempt in range(1, retries + 1):
        try:
            response = requests.post(url, json=data, timeout=timeout)
            response.raise_for_status()
            text = response.json().get("response", "").strip()
            if text:
                return text
            print(f"⚠️ Empty response from {model} (attempt {attempt}/{retries})")
        except requests.exceptions.ConnectionError:
            return "❌ Cannot connect to Ollama. Make sure Ollama is running."
        except requests.exceptions.Timeout:
            return "❌ Request timed out. The model might be loading — try again."
        except Exception as e:
            return f"❌ Error: {str(e)}"

    return _FALLBACK


def ask_ollama(prompt: str) -> str:
    """Send a plain prompt using the fast model. Used by file intent parsing."""
    return _call_ollama(prompt, OLLAMA_FAST_MODEL, timeout=30)


def get_chat_response(session_id, user_query: str, mode: str = "fast") -> str:
    """Lightweight wrapper — used only if chat_service is not available."""
    model = OLLAMA_FAST_MODEL if mode == "fast" else OLLAMA_THINKING_MODEL
    timeout = 60 if mode == "fast" else 180
    return _call_ollama(user_query, model, timeout)
