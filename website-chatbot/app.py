"""Minimal company website chatbot.

The bot answers questions using ONLY the content of the files in
``company_info/``. There is no vector database and no RAG pipeline: the files
are small, so their full text is placed in the system prompt (with prompt
caching so the repeated context stays cheap). If an answer is not in the files,
the bot says so instead of making something up.

Run locally:

    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=sk-ant-...
    uvicorn app:app --reload

then open http://localhost:8000
"""

import os
from pathlib import Path

import anthropic
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# --------------------------------------------------------------------------- #
# Configuration (override with environment variables)
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).parent
COMPANY_NAME = os.environ.get("COMPANY_NAME", "شركتنا")
# claude-haiku-4-5 -> cheapest/fastest. Switch to claude-sonnet-5 or
# claude-opus-4-8 for higher answer quality (see README for prices).
MODEL = os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5")
KNOWLEDGE_DIR = Path(os.environ.get("KNOWLEDGE_DIR", BASE_DIR / "company_info"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "1024"))
MAX_HISTORY = 20  # how many past messages to keep per conversation

# --------------------------------------------------------------------------- #
# Load the knowledge files once at startup
# --------------------------------------------------------------------------- #
def load_knowledge() -> str:
    """Concatenate every .md / .txt file under KNOWLEDGE_DIR into one string."""
    parts = []
    for path in sorted(KNOWLEDGE_DIR.glob("**/*")):
        if path.is_file() and path.suffix.lower() in {".md", ".txt"}:
            parts.append(f"### File: {path.name}\n{path.read_text(encoding='utf-8')}")
    return "\n\n".join(parts) if parts else "(No company information files found.)"


KNOWLEDGE = load_knowledge()

SYSTEM_PROMPT = f"""You are the customer assistant for {COMPANY_NAME}.

Rules:
- Answer using ONLY the information inside <company_information> below.
- If the answer is not there, clearly say you don't have that information and
  suggest contacting the company. Never invent facts, prices, or policies.
- Reply in the SAME language the user writes in (Arabic or English).
- Keep answers short, friendly, and professional.

<company_information>
{KNOWLEDGE}
</company_information>"""

# --------------------------------------------------------------------------- #
# Anthropic client (created lazily so the module imports without an API key)
# --------------------------------------------------------------------------- #
_client: anthropic.Anthropic | None = None


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    return _client


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
app = FastAPI(title=f"{COMPANY_NAME} Chatbot")


class ChatRequest(BaseModel):
    messages: list[dict]  # [{"role": "user"|"assistant", "content": "..."}]


@app.post("/chat")
def chat(req: ChatRequest):
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in req.messages
        if m.get("role") in ("user", "assistant") and m.get("content")
    ][-MAX_HISTORY:]

    if not history:
        return {"reply": "مرحباً! كيف أقدر أساعدك؟"}

    try:
        response = get_client().messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    # Cache the company info so repeated calls are ~90% cheaper.
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=history,
        )
        reply = "".join(b.text for b in response.content if b.type == "text")
        return {"reply": reply or "عذراً، لم أفهم سؤالك. ممكن تعيد صياغته؟"}
    except anthropic.APIError as exc:  # network / auth / rate-limit, etc.
        return {"reply": "عذراً، صار خطأ مؤقت. حاول مرة ثانية بعد لحظات.",
                "error": str(exc)}


# Serve the chat widget at "/" (registered after /chat so the API wins).
app.mount("/", StaticFiles(directory=BASE_DIR / "static", html=True), name="static")
