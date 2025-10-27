import os, json, requests, re, gc
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langdetect import detect
from fastapi.middleware.cors import CORSMiddleware  # ✅ NEW

# -------------------------------------------------------------
# Ollama + FastAPI Config
# -------------------------------------------------------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

app = FastAPI(title="CCA Bilingual Chatbot API", version="2.2")

# ✅ Enable CORS so frontend can access backend
origins = [
    "https://cca-project-frontend.onrender.com",  # your Streamlit frontend
    "http://localhost:8501",                      # optional: local Streamlit dev
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------
# Load lightweight resources only
# -------------------------------------------------------------
with open("prompts/system_en.txt", "r", encoding="utf-8") as f:
    SYSTEM_EN = f.read().strip()
with open("prompts/system_hi.txt", "r", encoding="utf-8") as f:
    SYSTEM_HI = f.read().strip()

# -------------------------------------------------------------
# Input Schema
# -------------------------------------------------------------
class ChatIn(BaseModel):
    query: str
    lang: str | None = None  # "en", "hi", or auto-detect


# -------------------------------------------------------------
# Lazy Retrieval Loader
# -------------------------------------------------------------
def get_retriever():
    """
    Dynamically import and initialize the retriever only when needed.
    This prevents large FAISS + transformer models from loading at startup.
    """
    from .retrieve import HybridRetriever
    retriever = HybridRetriever()
    return retriever


def build_context(docs):
    """Format retrieved documents with citations."""
    blocks, cites = [], []
    for d in docs:
        org = d["meta"].get("org", "Unknown")
        year = d["meta"].get("year", "NA")
        title = d["meta"].get("title", "")
        blocks.append(f"Source [{org}, {year}, {title}]: {d['text']}")
        cites.append({"org": org, "year": year, "title": title})
    return "\n\n".join(blocks), cites


# -------------------------------------------------------------
# Chat Endpoint (Streaming)
# -------------------------------------------------------------
@app.post("/chat")
def chat(inp: ChatIn):
    q = inp.query.strip()
    if not q:
        return StreamingResponse(iter(["Please ask a question."]), media_type="text/plain")

    # Auto-detect language
    lang = inp.lang
    if not lang:
        try:
            lang = "hi" if detect(q) == "hi" else "en"
        except Exception:
            lang = "en"

    system_prompt = SYSTEM_HI if lang == "hi" else SYSTEM_EN

    # Lazy load retriever here (saves memory on startup)
    retriever = get_retriever()

    # Retrieve top relevant docs
    try:
        docs = retriever.retrieve(q)
    except Exception as e:
        docs = [{"text": f"[Retriever failed: {e}]", "meta": {"org": "N/A", "year": "N/A", "title": ""}}]

    context, cites = build_context(docs)

    # Build final prompt
    prompt = f"""{system_prompt}

[Retrieved context]
{context}

[User question]
{q}

[Instructions]
- Answer in {"Hindi" if lang == "hi" else "English"}.
- Start with concise actionable steps.
- Follow with a short explanation citing organization + year.
"""

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": True,
        "options": {"temperature": 0.2}
    }

    print("\n==================== Prompt Sent to Ollama ====================")
    print(prompt[:1000] + ("..." if len(prompt) > 1000 else ""))
    print("==============================================================\n")

    def stream_response():
        collected_text = ""
        try:
            with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=None) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if line:
                        try:
                            data = json.loads(line.decode("utf-8"))
                            if "response" in data:
                                chunk = data["response"]
                                collected_text += chunk
                                yield chunk
                        except json.JSONDecodeError:
                            continue
        except requests.exceptions.RequestException as e:
            yield f"\n[❌ Ollama request failed: {e}]"
        except Exception as e:
            yield f"\n[⚠️ Unexpected error: {e}]"

        # Cleanup output
        cleaned_resp = re.sub(r"\[\[END_JSON\]\].*|\{.*\}\]\]$", "", collected_text).strip()
        cleaned_resp = re.sub(r"(?s)```json.*?```", "", cleaned_resp).strip()

        final_json = json.dumps({
            "answer": cleaned_resp,
            "sources": cites
        }, ensure_ascii=False)

        yield f"\n[[END_JSON]]{final_json}[[END_JSON]]"

        # Explicit memory cleanup
        del retriever
        gc.collect()

    return StreamingResponse(stream_response(), media_type="text/plain")


# -------------------------------------------------------------
# Health Check Endpoint
# -------------------------------------------------------------
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model": MODEL,
        "ollama_url": OLLAMA_URL,
        "retriever_status": "lazy-loaded"
    }


# -------------------------------------------------------------
# Root Endpoint
# -------------------------------------------------------------
@app.get("/")
def root():
    return {
        "message": "🌿 CCA Bilingual Chatbot API is running 🚀",
        "endpoints": {
            "/chat": "POST — Stream bilingual chatbot responses",
            "/health": "GET — Check server and model status"
        }
    }
