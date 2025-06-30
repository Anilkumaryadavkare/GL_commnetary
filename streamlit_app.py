import pandas as pd
import streamlit as st
from openai import AzureOpenAI
import calendar
import hashlib
import io
from datetime import datetime
from typing import List
from concurrent.futures import ThreadPoolExecutor

from auth_module import auth_gate
auth_gate()  # Will show login/signup if not logged in

from dotenv import load_dotenv
import os

# load_dotenv()

# --- Streamlit Config ---
st.set_page_config(page_title="GL Analytics Pro", layout="wide", page_icon="💼")

# --- Azure Config ---
# AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
# AZURE_OPENAI_MODEL_NAME = os.getenv("AZURE_OPENAI_MODEL_NAME")
# AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
# AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")


# --- Azure OpenAI Configuration ---
try:
    AZURE_OPENAI_ENDPOINT = st.secrets["AZURE_OPENAI_ENDPOINT"]
    AZURE_OPENAI_MODEL_NAME = st.secrets["AZURE_OPENAI_MODEL_NAME"]
    AZURE_OPENAI_API_KEY = st.secrets["AZURE_OPENAI_API_KEY"]
    AZURE_OPENAI_API_VERSION = st.secrets["AZURE_OPENAI_API_VERSION"]
except AttributeError as e:
    st.error(f"Azure OpenAI secrets missing or incorrectly configured: {e}")
    st.stop()

# --- Helper Functions ---
def initialize_openai():
    return AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT
    )

def standardize_columns(df, file_type):
    column_map = {
        'GL_mapping': {
            'Account Number': 'Account_Number',
            'Text for B/S P&L item': 'Text_for_BS_PL_item'
        },
        'Transaction': {
            'G/L': 'G_L',
            'Year/month': 'Year_month',
            'Amount in DC': 'Amount_in_DC',
            'Typ': 'Typ',
            'Text': 'Text'
        }
    }
    df.columns = [col.strip() for col in df.columns]
    for original, new in column_map[file_type].items():
        if original in df.columns:
            df.rename(columns={original: new}, inplace=True)
    return df

def load_and_cache_gl_mapping(uploaded_file):
    file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
    if 'gl_mapping' not in st.session_state or st.session_state.gl_mapping_hash != file_hash:
        df = pd.read_excel(uploaded_file)
        df = standardize_columns(df, 'GL_mapping')
        if not all(col in df.columns for col in ['Account_Number', 'Text_for_BS_PL_item']):
            st.error("Missing required columns in GL Mapping.")
            st.stop()
        st.session_state.gl_mapping = df
        st.session_state.gl_mapping_hash = file_hash
    return st.session_state.gl_mapping

def detect_available_periods(df):
    return sorted(df['Year_month'].dropna().unique(), reverse=True)

def calculate_period_balances(df, end_period):
    filtered = df[df['Year_month'] <= end_period]
    return filtered.groupby('G_L')['Amount_in_DC'].sum().reset_index().rename(columns={'Amount_in_DC': 'Cumulative_Amount'})

def format_period_name(period_str):
    try:
        year, month = period_str.strip().split('_')
        return f"{calendar.month_abbr[int(month)]}'{year[2:]}"
    except:
        return period_str.strip()

def extract_top_contributors(transactions, current_period):
    df = transactions[transactions['Year_month'] == current_period]
    grouped = df.groupby(['G_L', 'Text'])['Amount_in_DC'].sum().reset_index()
    typs = df.groupby(['G_L', 'Typ'])['Amount_in_DC'].sum().reset_index()
    result = {}
    for acc in grouped['G_L'].unique():
        top_texts = grouped[grouped['G_L'] == acc].nlargest(5, 'Amount_in_DC')['Text'].tolist()
        top_typs = typs[typs['G_L'] == acc].nlargest(2, 'Amount_in_DC')['Typ'].astype(str).tolist()
        result[acc] = {
            'top_texts': [str(t) for t in top_texts if pd.notna(t)],
            'top_typs': [str(t) for t in top_typs if pd.notna(t)]
        }
    return result

@st.cache_data(show_spinner=False)
def generate_commentary_batch(rows: list[dict], extra_details: dict):
    client = initialize_openai()
    cfo_comments = []
    analyst_comments = []

    def generate_comment(row):
        acc = row['Account Description']
        acc_num = row['Account Number']
        change = row['Change']
        top = extra_details.get(acc_num, {'top_texts': [], 'top_typs': []})
        items = ', '.join(str(x) for x in top['top_texts']) if top['top_texts'] else 'No major entries'
        typs = ', '.join(str(x) for x in top['top_typs']) if top['top_typs'] else 'N/A'

        cfo_prompt = f"""
        Provide a concise 2-line commentary.
        Account: {acc}
        Change: {change:,.2f}
        Avoid markdown. Be executive.
        Type(s): {typs}
        Top Items: {items}
        """

        analyst_prompt = f"""
        You are a financial analyst creating commentary.
        Account: {acc}
        Change: {change:,.2f}
        Explain possible drivers using type and material.
        Top materials: {items}
        Types: {typs}
        Professional tone. No markdown. Max 3 lines.
        """
        try:
            cfo_res = client.chat.completions.create(
                model=AZURE_OPENAI_MODEL_NAME,
                messages=[{"role": "system", "content": "You are a CFO."}, {"role": "user", "content": cfo_prompt}],
                temperature=0.3,
                max_tokens=120
            )
            analyst_res = client.chat.completions.create(
                model=AZURE_OPENAI_MODEL_NAME,
                messages=[{"role": "system", "content": "You are a senior financial analyst."}, {"role": "user", "content": analyst_prompt}],
                temperature=0.3,
                max_tokens=150
            )
            return cfo_res.choices[0].message.content.strip(), analyst_res.choices[0].message.content.strip()
        except Exception as e:
            return f"Error: {e}", f"Error: {e}"

    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(generate_comment, rows))

    cfo_comments = [r[0] for r in results]
    analyst_comments = [r[1] for r in results]
    return cfo_comments, analyst_comments

def generate_comparison_report(gl_mapping, transactions, current_period, prev_period):
    curr_bal = calculate_period_balances(transactions, current_period)
    prev_bal = calculate_period_balances(transactions, prev_period)
    report_df = pd.merge(gl_mapping, curr_bal, left_on='Account_Number', right_on='G_L', how='left')
    prev_bal.rename(columns={"Cumulative_Amount": "Cumulative_Amount_prev"}, inplace=True)
    report_df = pd.merge(report_df, prev_bal[['G_L', 'Cumulative_Amount_prev']], left_on='Account_Number', right_on='G_L', how='left')
    report_df['Period_Difference'] = report_df['Cumulative_Amount'].fillna(0) - report_df['Cumulative_Amount_prev'].fillna(0)

    report_df.rename(columns={
        'Account_Number': 'Account Number',
        'Text_for_BS_PL_item': 'Account Description',
        'Period_Difference': 'Change'
    }, inplace=True)

    top_contributors = extract_top_contributors(transactions, current_period)
    rows = report_df.to_dict(orient='records')
    cfo_comms, analyst_comms = generate_commentary_batch(rows, top_contributors)
    report_df['CFO Commentary'] = cfo_comms
    report_df['Analyst Commentary'] = analyst_comms

    curr_label = format_period_name(current_period)
    prev_label = format_period_name(prev_period)

    report_df.rename(columns={
        'Cumulative_Amount': curr_label,
        'Cumulative_Amount_prev': prev_label
    }, inplace=True)

    return report_df[[
        'Account Number', 'Account Description',
        curr_label, prev_label,
        'Change', 'CFO Commentary', 'Analyst Commentary'
    ]]

# --- UI ---
st.title("📊 GL Commentary Generator")

with st.expander("📁 Upload Files", expanded=True):
    col1, col2 = st.columns(2)
    gl_file = col1.file_uploader("GL Mapping File", type="xlsx")
    tx_file = col2.file_uploader("Transaction Dump", type="xlsx")

if gl_file and tx_file:
    gl_df = load_and_cache_gl_mapping(gl_file)
    tx_df = pd.read_excel(tx_file)
    tx_df = standardize_columns(tx_df, 'Transaction')

    periods = detect_available_periods(tx_df)
    if len(periods) < 2:
        st.error("At least 2 periods required.")
        st.stop()

    idx = st.select_slider("Select Periods to Compare", options=list(range(len(periods)-1)),
                           format_func=lambda i: f"{format_period_name(periods[i])} vs {format_period_name(periods[i+1])}")
    p1, p2 = periods[idx], periods[idx+1]

    with st.spinner("🔍 Generating commentary..."):
        report_df = generate_comparison_report(gl_df, tx_df, p1, p2)
        st.session_state.report_df = report_df
        st.success("✅ Report ready!")

if 'report_df' in st.session_state:
    st.markdown("### 🔍 Filter Results")
    col1, col2 = st.columns(2)
    with col1:
        min_val = st.number_input("Minimum Change", value=100000, step=10000)
    with col2:
        accounts = st.multiselect("Account Filter", st.session_state.report_df['Account Number'].unique())

    df = st.session_state.report_df.copy()
    df = df[df['Change'].abs() >= min_val]
    if accounts:
        df = df[df['Account Number'].isin(accounts)]

    st.dataframe(df, use_container_width=True)

    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)
    st.download_button("Download Excel", data=excel_buffer.getvalue(), file_name="GL_Commentary_Report.xlsx")
