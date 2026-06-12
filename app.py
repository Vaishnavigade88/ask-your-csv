import streamlit as st
import pandas as pd
import google.generativeai as genai

st.set_page_config(page_title="CSV Monster Bot", layout="wide")

# Gemini Configuration
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

model = genai.GenerativeModel("gemini-1.5-flash")

def query_gemini(prompt):
    response = model.generate_content(prompt)
    return response.text

# App Title
st.title("📊 CSV Monster Bot (Gemini AI Powered)")

# File Upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.success("✅ File uploaded successfully!")

    # Raw Data
    st.subheader("📂 Raw Data")
    st.dataframe(df)

    # Cleaned Data
    st.subheader("🧹 Cleaned Data (Nulls Removed)")
    df_cleaned = df.dropna()
    st.dataframe(df_cleaned)

    # Data Cleaning Options
    st.subheader("🧼 Data Cleaning Options")

    st.write("🔍 Raw Data Preview:")
    st.dataframe(df.head())

    if st.checkbox("Drop rows with missing values"):
        df.dropna(inplace=True)
        st.success("Dropped all rows with missing values.")

    if st.checkbox("Fill missing values"):
        fill_option = st.selectbox(
            "Select method",
            ["Mean", "Median", "Mode"]
        )

        for col in df.select_dtypes(include=["float64", "int64"]).columns:

            if fill_option == "Mean":
                df[col] = df[col].fillna(df[col].mean())

            elif fill_option == "Median":
                df[col] = df[col].fillna(df[col].median())

            elif fill_option == "Mode":
                df[col] = df[col].fillna(df[col].mode()[0])

        st.success(f"Filled missing values using {fill_option}.")

    if st.checkbox("Remove duplicate rows"):
        df.drop_duplicates(inplace=True)
        st.success("Duplicate rows removed.")

    if st.checkbox("Drop selected columns"):
        cols_to_drop = st.multiselect(
            "Select columns to drop",
            df.columns
        )

        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)
            st.success(f"Dropped columns: {', '.join(cols_to_drop)}")

    if st.checkbox("Rename columns"):

        old_cols = st.multiselect(
            "Select columns to rename",
            df.columns
        )

        for col in old_cols:

            new_name = st.text_input(
                f"Rename '{col}' to:",
                key=col
            )

            if new_name:
                df.rename(
                    columns={col: new_name},
                    inplace=True
                )

        if old_cols:
            st.success("Columns renamed successfully.")

    if st.checkbox("Reset index"):
        df.reset_index(drop=True, inplace=True)
        st.success("Index reset successfully.")

    # Final Cleaned Data
    st.subheader("📊 Cleaned Data Preview")
    st.dataframe(df.head())

    # Statistics
    st.subheader("📈 Quick Statistics")

    try:
        st.write(df.describe())
    except:
        st.write("No numerical columns available.")

    # Visualizations
    st.subheader("📊 Data Visualizations")

    numeric_cols = df.select_dtypes(
        include=["float64", "int64"]
    ).columns.tolist()

    if numeric_cols:

        chart_type = st.selectbox(
            "Choose a chart type",
            ["Line Chart", "Bar Chart", "Area Chart"]
        )

        column_to_plot = st.selectbox(
            "Choose a column to visualize",
            numeric_cols
        )

        if chart_type == "Line Chart":
            st.line_chart(df[column_to_plot])

        elif chart_type == "Bar Chart":
            st.bar_chart(df[column_to_plot])

        elif chart_type == "Area Chart":
            st.area_chart(df[column_to_plot])

    else:
        st.warning("No numeric columns found for visualization.")

    # AI Q&A Section
    st.subheader("❓ Ask Questions about Your Data")

    user_question = st.text_input(
        "Type your question"
    )

    if user_question:

        with st.spinner("🤖 Gemini is thinking..."):

            prompt = f"""
You are a professional data analyst.

Dataset:
{df.head(50).to_string(index=False)}

Answer the following question based on the dataset.

Question:
{user_question}

Answer:
"""

            try:
                answer = query_gemini(prompt)

                st.success("🧠 AI Answer")
                st.write(answer)

            except Exception as e:
                st.error(f"Error: {e}")

else:
    st.info("⬆️ Upload a CSV file to get started.")
