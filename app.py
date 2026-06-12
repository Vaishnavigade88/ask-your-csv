import streamlit as st
import pandas as pd
import google.generativeai as genai

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="DataPilot AI",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Custom Styling
# -----------------------------
st.markdown("""
<style>
.main-header {
    text-align: center;
    padding: 20px;
}

.metric-card {
    background-color: #262730;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
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
# Gemini Setup
# -----------------------------
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

model = genai.GenerativeModel("gemini-1.5-flash")


def query_gemini(prompt):
    response = model.generate_content(prompt)
    return response.text


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.title("🚀 DataPilot AI")

    st.markdown("""
### About

AI-Powered CSV Analytics Platform

### Features

✅ Upload CSV Files

✅ Data Cleaning

✅ Statistics

✅ Interactive Charts

✅ Gemini AI Insights

### Tech Stack

- Python
- Streamlit
- Pandas
- Gemini AI
""")

    st.success("Ready for Analysis")

# -----------------------------
# Landing Page
# -----------------------------
st.markdown("""
<div class="main-header">
<h1>📊 DataPilot AI</h1>
<h4>Transform CSV Files into Actionable Insights using Gemini AI</h4>
</div>
""", unsafe_allow_html=True)

st.info(
    "🤖 Upload a CSV file and ask questions in natural language. "
    "Clean data, generate statistics, visualize trends, and get AI-powered insights."
)

# Feature Cards
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-box">
    <h3>🧹 Data Cleaning</h3>
    Handle missing values, duplicates, and column operations.
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-box">
    <h3>📈 Visualizations</h3>
    Generate charts instantly from your data.
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-box">
    <h3>🤖 Gemini AI</h3>
    Ask questions and get intelligent insights.
    </div>
    """, unsafe_allow_html=True)

st.divider()

# -----------------------------
# Upload CSV
# -----------------------------
uploaded_file = st.file_uploader(
    "📂 Upload Your CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    try:
    df = pd.read_csv(uploaded_file)
    except:
        try:
            df = pd.read_csv(uploaded_file, encoding="latin1")
        except:
            try:
                df = pd.read_csv(uploaded_file, sep=";")
            except Exception as e:
                st.error(f"Could not read file: {e}")
                st.stop()

    st.success("✅ File Uploaded Successfully!")

    # -----------------------------
    # Dataset Overview
    # -----------------------------
    st.subheader("📋 Dataset Overview")

    rows = df.shape[0]
    cols = df.shape[1]
    missing = int(df.isnull().sum().sum())
    duplicates = int(df.duplicated().sum())

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Rows", rows)
    c2.metric("Columns", cols)
    c3.metric("Missing Values", missing)
    c4.metric("Duplicates", duplicates)

    # -----------------------------
    # Raw Data
    # -----------------------------
    st.subheader("📂 Raw Data")
    st.dataframe(df, use_container_width=True)

    # -----------------------------
    # Cleaning Options
    # -----------------------------
    st.subheader("🧹 Data Cleaning")

    if st.checkbox("Remove Missing Values"):
        df.dropna(inplace=True)
        st.success("Missing values removed.")

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

        st.success(f"Missing values filled using {fill_option}")

    if st.checkbox("Remove Duplicate Rows"):
        df.drop_duplicates(inplace=True)
        st.success("Duplicate rows removed.")

    if st.checkbox("Drop Columns"):

        cols_to_drop = st.multiselect(
            "Select Columns",
            df.columns
        )

        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)
            st.success("Columns removed successfully.")

    if st.checkbox("Rename Columns"):

        selected_cols = st.multiselect(
            "Select Columns to Rename",
            df.columns
        )

        for col in selected_cols:

            new_name = st.text_input(
                f"Rename '{col}' to:"
            )

            if new_name:
                df.rename(
                    columns={col: new_name},
                    inplace=True
                )

    if st.checkbox("Reset Index"):
        df.reset_index(drop=True, inplace=True)
        st.success("Index reset successfully.")

    # -----------------------------
    # Cleaned Data
    # -----------------------------
    st.subheader("📊 Cleaned Dataset")
    st.dataframe(df, use_container_width=True)

    # -----------------------------
    # Statistics
    # -----------------------------
    st.subheader("📈 Statistical Summary")

    try:
        st.dataframe(
            df.describe(),
            use_container_width=True
        )
    except:
        st.warning("No numerical columns found.")

    # -----------------------------
    # Visualizations
    # -----------------------------
    st.subheader("📊 Data Visualization")

    numeric_cols = df.select_dtypes(
        include=["float64", "int64"]
    ).columns.tolist()

    if len(numeric_cols) > 0:

        chart_type = st.selectbox(
            "Chart Type",
            [
                "Line Chart",
                "Bar Chart",
                "Area Chart"
            ]
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

    else:
        st.warning(
            "No numeric columns available for visualization."
        )

    # -----------------------------
    # Download CSV
    # -----------------------------
    st.subheader("📥 Download Cleaned Data")

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Cleaned CSV",
        data=csv,
        file_name="cleaned_data.csv",
        mime="text/csv"
    )

    # -----------------------------
    # AI Section
    # -----------------------------
    st.subheader("🤖 Ask Gemini About Your Data")

    user_question = st.text_input(
        "Ask a question about your dataset"
    )

    if user_question:

        with st.spinner("Gemini is analyzing your data..."):

            prompt = f"""
You are an expert data analyst.

Analyze the dataset below and answer the user's question.

Dataset:
{df.head(100).to_string(index=False)}

Question:
{user_question}

Instructions:
- Be accurate.
- Mention useful insights.
- Be concise and professional.
- If information is unavailable, say so.

Answer:
"""

            try:
                answer = query_gemini(prompt)

                st.success("🧠 Gemini Insight")
                st.write(answer)

            except Exception as e:
                st.error(f"Error: {e}")

else:
    st.info("⬆️ Upload a CSV file to begin analysis.")
