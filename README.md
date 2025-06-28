# 💼 GL Commentary App

A powerful Streamlit-based app that generates executive-level and analyst-level financial commentary from SAP general ledger data. It uses Azure OpenAI (GPT-4) to summarize large volumes of account-level variances between reporting periods.

## 🚀 Features

- 📊 Upload GL Mapping and Transaction dump (Excel)
- 🔍 Compare two periods and compute changes
- 🧠 AI-generated commentary (Analyst + CFO views)
- 🧵 Fast response using stream generation
- 📤 Export the final report to Excel

## 🧠 Powered By

- Azure OpenAI (GPT-4 via Microsoft Cloud)
- Streamlit for frontend
- Pandas for data transformation

## ⚠️ Credentials

**❗ Do not expose your API keys or Microsoft credentials in the codebase.**  
Use `.env` files or environment variables in deployment. See below.

## 📦 Setup (Local)

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
