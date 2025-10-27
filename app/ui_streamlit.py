import streamlit as st
import requests
import json
import re

st.set_page_config(page_title="🌿 CCA – Bilingual Chatbot", layout="wide")

st.title("🌿 Climate-Controlled Agriculture (CCA) – Bilingual Chatbot")
st.markdown("Ask in **English** or **Hindi** about greenhouse or hydroponic systems.")

# -----------------------------------------------------
# Backend URL
# -----------------------------------------------------
API_URL = "https://cca-bot-backend.onrender.com/chat"

# -----------------------------------------------------
# Language Selection
# -----------------------------------------------------
lang_choice = st.radio("Choose Language", ["Auto", "English", "Hindi"], horizontal=True)
lang_map = {"Auto": None, "English": "en", "Hindi": "hi"}

# -----------------------------------------------------
# Query Box
# -----------------------------------------------------
query = st.text_area("💬 Ask your question here:", "", height=100)
ask = st.button("Ask")

# -----------------------------------------------------
# Chat Handling
# -----------------------------------------------------
if ask and query.strip():
    st.info("⏳ Query sent to model... please wait.")

    payload = {"query": query, "lang": lang_map[lang_choice]}

    try:
        with requests.post(API_URL, json=payload, stream=True, timeout=None) as r:
            if r.status_code != 200:
                st.error(f"❌ Server error: {r.status_code} {r.text}")
            else:
                placeholder = st.empty()
                response_text = ""
                json_buffer = ""

                for chunk in r.iter_content(chunk_size=None):
                    if not chunk:
                        continue

                    piece = chunk.decode("utf-8", errors="ignore")

                    # If [[END_JSON]] appears, buffer JSON section only
                    if "[[END_JSON]]" in piece:
                        json_buffer += piece
                    else:
                        response_text += piece
                        cleaned = re.sub(r"\[\[END_JSON\]\].*|\{.*\}\]\]$", "", response_text).strip()
                        placeholder.markdown(cleaned)

                # ✅ Extract final JSON block safely
                full_json_block = None
                if "[[END_JSON]]" in json_buffer:
                    try:
                        json_str = re.search(r"\[\[END_JSON\]\](.*?)\[\[END_JSON\]\]", json_buffer, re.S).group(1)
                        full_json_block = json.loads(json_str)
                    except Exception as e:
                        st.warning(f"⚠️ Could not parse final JSON cleanly: {e}")

                st.success("✅ Done!")

                # ✅ Display citations cleanly
                if full_json_block and "sources" in full_json_block:
                    cites = full_json_block["sources"]
                    if cites:
                        st.markdown("### 📚 Sources")
                        for c in cites:
                            org = c.get("org", "Unknown")
                            year = c.get("year", "NA")
                            title = c.get("title", "")
                            st.markdown(f"- **{org}** ({year}) — {title}")

    except requests.exceptions.RequestException as e:
        st.error(f"Server error: {e}")

else:
    st.caption("👆 Type a question and click **Ask** to start.")
