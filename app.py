import streamlit as st
import pandas as pd
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss

st.set_page_config(page_title="DataPilot AI RAG", page_icon="📊", layout="wide")

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def query_ai(prompt):
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return completion.choices[0].message.content

def build_vector_store(df):
    text_data = []
    for _, row in df.iterrows():
        text_data.append(" | ".join(map(str, row.values)))

    model = load_embedding_model()
    embeddings = model.encode(text_data, convert_to_numpy=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype("float32"))

    return model, index, text_data

def retrieve_context(query, model, index, text_data, k=5):
    query_embedding = model.encode([query], convert_to_numpy=True)

    distances, indices = index.search(
        query_embedding.astype("float32"), k
    )

    retrieved_rows = []
    for idx in indices[0]:
        if idx < len(text_data):
            retrieved_rows.append(text_data[idx])

    return "\n".join(retrieved_rows)

st.title("📊 DataPilot AI - RAG Edition")
st.write("Upload CSV → Build Embeddings → Retrieve Context → Ask Groq")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding_errors="ignore")

    st.success("CSV Uploaded Successfully")
    st.dataframe(df.head())

    st.subheader("Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Values", int(df.isnull().sum().sum()))
    c4.metric("Duplicates", int(df.duplicated().sum()))

    st.subheader("Statistical Summary")
    try:
        st.dataframe(df.describe())
    except Exception:
        st.info("No numeric columns")

    with st.spinner("Building Vector Store..."):
        model, index, text_data = build_vector_store(df)

    st.success("RAG Vector Store Ready")

    user_question = st.text_input("Ask a question about your dataset")

    if user_question:
        context = retrieve_context(
            user_question,
            model,
            index,
            text_data,
            k=5
        )

        with st.expander("Retrieved Context"):
            st.code(context)

        prompt = f"""
You are an expert data analyst.

Use ONLY the provided context.

Context:
{context}

Question:
{user_question}

Instructions:
- Be accurate.
- Mention useful insights.
- Be concise and professional.
- If information is unavailable, say so.

Answer:
"""

        with st.spinner("Analyzing..."):
            answer = query_ai(prompt)

        st.subheader("AI Insight")
        st.write(answer)
