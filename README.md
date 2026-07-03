# 📊 CSV Insight AI

A powerful and interactive tool built using **Streamlit** that lets users upload CSV files, explore data, perform data cleaning (nulls, duplicates, outliers), get quick statistics, visualize data, and even interact with it using **AI chat-based analysis** — all in one seamless experience!

---

## 🚀 Features

- ✅ Upload and preview any CSV file
- 📉 Automatic data analysis (shape, nulls, dtypes, stats, etc.)
- 🧹 One-click **Data Cleaning**:
  - Remove null values
  - Drop duplicate rows
  - Detect and remove outliers using IQR
- 📊 Visualize Data with:
  - Histograms
  - Correlation heatmaps
  - Box plots
- 🤖 AI-Powered Chat with CSV (Ask questions in natural language!)
- 📁 Download cleaned data as CSV

---

## ⚙️ Tech Stack

- Python 🐍
- Pandas 🧮
- Matplotlib / Seaborn 📈
- Streamlit 🧼
- OpenAI LLMs (for AI chat)
- scikit-learn (optional: for outlier detection)

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Vaishnavigade88/ask-your-csv.git
cd ask-your-csv
2. Create Virtual Environment (optional)
bash
Copy code
python -m venv venv
source venv/bin/activate  # for Windows: venv\Scripts\activate
3. Install Dependencies
bash
Copy code
pip install -r requirements.txt
4. Run the App
bash
Copy code
streamlit run main.py
🧠 AI Feature
The bot uses an LLM to allow users to chat with their CSV file and ask questions like:

"What are the top 5 rows with highest sales?"

"Average age of customers?"

"Count of rows with missing email?"

💡Note: Currently integrated with OpenAI GPT via API — no HuggingFace used.

📂 Folder Structure
bash
Copy code
csv-monster-bot/
│
├── main.py                # Main Streamlit app
├── requirements.txt       # Dependencies
├── cleaned_data.csv       # Output after cleaning
├── utils/                 # Custom helper functions (optional)
├── .gitignore
└── README.md              # This file

📝 Future Improvements
Add user login for session management

Export visuals as images

More outlier detection methods

Add chatbot memory (context-aware questions)

🙋‍♀️ Author
Vaishnavi Gade
📧 vaishnavigadeyshu@gmail.com
🌱 Passionate about AI, Data, and Full-Stack Development

