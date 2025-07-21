import streamlit as st
import pandas as pd
import os
import requests

st.set_page_config(page_title="CSV Monster Bot", layout="wide")

# Load CSV File
st.title("📊 CSV Monster Bot (Local AI Powered)")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    # Show raw data
    st.subheader("📂 Raw Data")
    st.dataframe(df)

    # Drop Nulls
    st.subheader("🧹 Cleaned Data (Nulls Removed)")
    df_cleaned = df.dropna()
    st.dataframe(df_cleaned)
    st.subheader("🧼 Data Cleaning Options")

    # Show original data
    st.write("🔍 Raw Data Preview:")
    st.dataframe(df.head())

    if st.checkbox("Drop rows with missing values"):
        df.dropna(inplace=True)
        st.success("Dropped all rows with missing values.")

    if st.checkbox("Fill missing values"):
        fill_option = st.selectbox("Select method", ["Mean", "Median", "Mode"])
        for col in df.select_dtypes(include=["float", "int"]):
            if fill_option == "Mean":
                df[col].fillna(df[col].mean(), inplace=True)
            elif fill_option == "Median":
                df[col].fillna(df[col].median(), inplace=True)
            elif fill_option == "Mode":
                df[col].fillna(df[col].mode()[0], inplace=True)
        st.success(f"Filled NaNs with {fill_option.lower()}.")

    if st.checkbox("Remove duplicate rows"):
        df.drop_duplicates(inplace=True)
        st.success("Duplicate rows removed.")

    if st.checkbox("Drop selected columns"):
        cols_to_drop = st.multiselect("Select columns to drop", df.columns)
        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)
            st.success(f"Dropped columns: {', '.join(cols_to_drop)}")

    if st.checkbox("Rename columns"):
        old_cols = st.multiselect("Select columns to rename", df.columns)
        for col in old_cols:
            new_name = st.text_input(f"Rename '{col}' to:", key=col)
            if new_name:
                df.rename(columns={col: new_name}, inplace=True)
        if old_cols:
            st.success("Renamed selected columns.")

    if st.checkbox("Reset index"):
        df.reset_index(drop=True, inplace=True)
        st.success("Index has been reset.")

    # Show cleaned data
    st.subheader("📊 Cleaned Data Preview:")
    st.dataframe(df.head())

    # Show stats
    st.subheader("📈 Quick Statistics")
    st.write(df_cleaned.describe())

    # Visualizations
    st.subheader("📊 Data Visualizations")
    numeric_cols = df_cleaned.select_dtypes(include=['float64', 'int64']).columns.tolist()

    if numeric_cols:
        chart_type = st.selectbox("Choose a chart type", ["Line Chart", "Bar Chart", "Area Chart"])
        column_to_plot = st.selectbox("Choose a column to visualize", numeric_cols)

        if chart_type == "Line Chart":
            st.line_chart(df_cleaned[column_to_plot])
        elif chart_type == "Bar Chart":
            st.bar_chart(df_cleaned[column_to_plot])
        elif chart_type == "Area Chart":
            st.area_chart(df_cleaned[column_to_plot])
    else:
        st.warning("No numeric columns found for visualization.")

    # Ollama function
    def query_ollama(prompt, model="tinyllama"):
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            }
        )
        return response.json()["response"]

    # Q&A Feature
    st.subheader("❓ Ask Questions about Your Data")
    user_question = st.text_input("Type your question (e.g. What is the average writing score?)")

    if user_question:
        with st.spinner("Thinking..."):
            prompt = f"""You are a data analyst. Given the following student performance data, answer the user's question briefly.

Data:
{df_cleaned.head(50).to_string(index=False)}

Question: {user_question}
Answer:"""

            try:
                answer = query_ollama(prompt)
                st.success("🧠 Answer:")
                st.write(answer)
            except Exception as e:
                st.error(f"❌ Error: {e}")

else:
    st.info("⬆️ Upload a CSV file to get started.")
