import streamlit as st
import pandas as pd
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="DataPilot AI RAG", page_icon="📊", layout="wide")

# -----------------------------
# Custom Styling
# -----------------------------
st.markdown("""
<style>
.main-header {
    text-align: center;
    padding: 20px;
}
.feature-box {
    background-color: #262730;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #3a3b45;
    margin-bottom: 10px;
}
.stButton>button {
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Groq Setup
# -----------------------------
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

def query_ai(prompt):
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return completion.choices[0].message.content

# -----------------------------
# Embedding Model
# -----------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

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
        query_embedding.astype("float32"),
        k
    )

    retrieved_rows = []

    for idx in indices[0]:
        if idx < len(text_data):
            retrieved_rows.append(text_data[idx])

    return "\n".join(retrieved_rows)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.title("🚀 DataPilot AI RAG")

    st.markdown("""
### Features

✅ Data Cleaning

✅ Statistics

✅ Visualizations

✅ CSV Download

✅ RAG Search

✅ Groq AI Insights

### Tech Stack

- Streamlit
- Pandas
- FAISS
- Sentence Transformers
- Groq LLM
""")

# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div class="main-header">
<h1>📊 DataPilot AI - RAG Edition</h1>
<h4>Clean Data • Visualize • Ask Questions using RAG</h4>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "📂 Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        df = pd.read_csv(
            uploaded_file,
            encoding_errors="ignore"
        )

    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()

    st.success("✅ CSV Uploaded Successfully")

    # Dataset Overview
    st.subheader("📋 Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Values", int(df.isnull().sum().sum()))
    c4.metric("Duplicates", int(df.duplicated().sum()))

    # Raw Data
    st.subheader("📂 Raw Data")
    st.dataframe(df, use_container_width=True)

    # Cleaning
    st.subheader("🧹 Data Cleaning")

    if st.checkbox("Remove Missing Values"):
        df.dropna(inplace=True)
        st.success("Missing values removed")

    if st.checkbox("Fill Missing Values"):

        fill_option = st.selectbox(
            "Choose Fill Method",
            ["Mean", "Median", "Mode"]
        )

        for col in df.select_dtypes(
            include=["float64", "int64"]
        ).columns:

            if fill_option == "Mean":
                df[col] = df[col].fillna(df[col].mean())

            elif fill_option == "Median":
                df[col] = df[col].fillna(df[col].median())

            elif fill_option == "Mode":
                df[col] = df[col].fillna(df[col].mode()[0])

        st.success(f"Filled using {fill_option}")

    if st.checkbox("Remove Duplicate Rows"):
        df.drop_duplicates(inplace=True)
        st.success("Duplicates removed")

    if st.checkbox("Drop Columns"):

        cols_to_drop = st.multiselect(
            "Select Columns",
            df.columns
        )

        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)
            st.success("Columns dropped")

    if st.checkbox("Rename Columns"):

        selected_cols = st.multiselect(
            "Columns to Rename",
            df.columns
        )

        for col in selected_cols:

            new_name = st.text_input(
                f"Rename {col}"
            )

            if new_name:
                df.rename(
                    columns={col: new_name},
                    inplace=True
                )

    if st.checkbox("Reset Index"):
        df.reset_index(drop=True, inplace=True)
        st.success("Index reset")

    # Cleaned Data
    st.subheader("📊 Cleaned Dataset")
    st.dataframe(df, use_container_width=True)

    # Statistics
    st.subheader("📈 Statistical Summary")

    try:
        st.dataframe(
            df.describe(),
            use_container_width=True
        )
    except Exception:
        st.warning("No numeric columns found")

    # Visualization
    st.subheader("📊 Data Visualization")

    numeric_cols = df.select_dtypes(
        include=["float64", "int64"]
    ).columns.tolist()

    if numeric_cols:

        chart_type = st.selectbox(
            "Chart Type",
            ["Line Chart", "Bar Chart", "Area Chart"]
        )

        column_to_plot = st.selectbox(
            "Select Column",
            numeric_cols
        )

        if chart_type == "Line Chart":
            st.line_chart(df[column_to_plot])

        elif chart_type == "Bar Chart":
            st.bar_chart(df[column_to_plot])

        elif chart_type == "Area Chart":
            st.area_chart(df[column_to_plot])

    # Download
    st.subheader("📥 Download Cleaned CSV")

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Cleaned CSV",
        csv,
        "cleaned_data.csv",
        "text/csv"
    )

    # Build RAG on cleaned data
    with st.spinner("Building RAG Vector Store..."):
        model, index, text_data = build_vector_store(df)

    st.success("✅ RAG Vector Store Ready")

    # AI Section
    st.subheader("🤖 Ask AI About Your Dataset")

    user_question = st.text_input(
        "Ask a question about your dataset"
    )

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

        st.subheader("🧠 AI Insight")
        st.write(answer)

else:
    st.info("⬆️ Upload a CSV file to begin analysis.")
