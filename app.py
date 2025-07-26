import streamlit as st
import pandas as pd
import numpy as np
import faiss
import google.generativeai as genai

# ---------------------- Streamlit Configuration ----------------------
st.set_page_config(page_title="📚 CSV RAG Chat with Gemini", layout="wide")
st.title("📚 Conversational RAG on CSV using Google Gemini + FAISS")

# ---------------------- Helper Functions ----------------------
def chunk_text(text, size=2048):
    return [text[i:i + size] for i in range(0, len(text), size)]

@st.cache_data(show_spinner=False)
def build_faiss_index(chunks, embed_function):
    embeddings = np.vstack([embed_function(chunk) for chunk in chunks]).astype("float32")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def get_gemini_embedding_fn(api_key):
    genai.configure(api_key=api_key)

    model = genai.get_model("embedding-001")

    def embed_fn(text):
        result = model.embed_content(
            content=text,
            task_type="retrieval_document"
        )
        return result["embedding"]

    return embed_fn


def get_answer_from_gemini(api_key, context, query, chat_history=[]):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-pro")

    prompt = f"""
You are a helpful assistant. Use the following context from a CSV file to answer user questions.

Context:
---------------------
{context}
---------------------

Question: {query}
Answer:"""

    history = [{"role": "user", "parts": [prompt]}]
    if chat_history:
        history.extend(chat_history)

    response = model.generate_content(history)
    return response.text

# ---------------------- Main App ----------------------
def main():
    # --- Sidebar: API and File ---
    api_key = st.sidebar.text_input("🔑 Google Gemini API Key", type="password")
    if not api_key:
        st.sidebar.warning("Please provide your Gemini API key.")
        st.stop()

    uploaded_file = st.sidebar.file_uploader("📂 Upload a CSV file", type=["csv"])
    if not uploaded_file:
        st.info("Upload a CSV file to begin.")
        st.stop()

    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("📄 Uploaded CSV Preview")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"❌ Could not read CSV file: {e}")
        st.stop()

    # --- Column selection ---
    selected_col = st.sidebar.selectbox("📌 Select text column for QA", df.columns)
    if not selected_col:
        st.warning("Please select a valid text column.")
        st.stop()

    text = " ".join(df[selected_col].astype(str).dropna().tolist())
    chunks = chunk_text(text)
    st.success(f"✅ Split column into {len(chunks)} chunks.")

    # --- Build index ---
    embed_fn = get_gemini_embedding_fn(api_key)
    faiss_index = build_faiss_index(chunks, embed_fn)

    # --- Initialize Chat Memory ---
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --- Chat UI ---
    st.subheader("💬 Ask anything about your CSV")
    with st.form("chat_form"):
        query = st.text_input("Your question", placeholder="e.g. What is this CSV about?")
        submitted = st.form_submit_button("Ask")

    if submitted and query:
        try:
            query_embedding = np.array([embed_fn(query)]).astype("float32")
            _, indices = faiss_index.search(query_embedding, 3)
            retrieved_chunks = [chunks[i] for i in indices[0]]
            context = "\n\n".join(retrieved_chunks)

            answer = get_answer_from_gemini(api_key, context, query, st.session_state.chat_history)

            # Update session chat history
            st.session_state.messages.append(("user", query))
            st.session_state.messages.append(("assistant", answer))
            st.session_state.chat_history.append({"role": "user", "parts": [query]})
            st.session_state.chat_history.append({"role": "model", "parts": [answer]})

        except Exception as e:
            st.error(f"❌ Failed to generate answer: {e}")

    # --- Display Chat Messages ---
    for role, msg in st.session_state.messages:
        if role == "user":
            st.chat_message("🧍 User").markdown(msg)
        else:
            st.chat_message("🤖 Assistant").markdown(msg)

if __name__ == "__main__":
    main()
