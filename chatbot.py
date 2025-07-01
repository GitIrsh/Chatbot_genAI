import streamlit as st
import pandas as pd
import duckdb
from transformers import pipeline
from dateutil.relativedelta import relativedelta
import datetime
import torch

# Load datasets with caching
@st.cache_data
def load_data():
    bo_tbl = pd.read_csv("data/sample_bo_tbl_large.csv")
    sub_details = pd.read_csv("data/sample_sub_details_large.csv")
    revenue = pd.read_csv("data/sample_revenue_large.csv")
    return bo_tbl, sub_details, revenue

# Load LLM model with caching
@st.cache_resource
def load_model():
    # Check if GPU is available
    device = 0 if torch.cuda.is_available() else -1
    
    return pipeline(
        "text-generation", 
        model="HuggingFaceH4/zephyr-7b-beta",
        torch_dtype=torch.bfloat16 if device == 0 else torch.float32,
        device=device
    )
# Build prompt with date handling
def build_prompt(question):
    today = datetime.date.today()
    last_month = today - relativedelta(months=1)
    
    return f"""
You are a data analyst converting questions to DuckDB SQL. Use these tables:
1. bo_tbl(country, date, total_mins, international_mins, sms, total_data_usage, payg_amount)
2. sub_details(country, channel, date, subs, netadds, churn)
3. revenue(country, channel, date, revenue, net_revenue)

Rules:
- Use SQL format: SELECT ... FROM ... WHERE ...
- TODAY = '{today}'
- LAST_MONTH = '{last_month.strftime("%Y-%m")}-01' to '{last_month.strftime("%Y-%m-%d")}'
- When comparing channels, use 'Online' vs 'Retail'

Examples:
Q: "Churn in AAA last month" 
A: SELECT SUM(churn) FROM sub_details WHERE country='AAA' AND date BETWEEN '{last_month.strftime("%Y-%m")}-01' AND '{last_month.strftime("%Y-%m-%d")}'

Q: "Compare net revenue across channels for DDD"
A: SELECT channel, SUM(net_revenue) FROM revenue WHERE country='DDD' GROUP BY channel

Q: "Highest data usage in Q2"
A: SELECT country, SUM(total_data_usage) FROM bo_tbl WHERE EXTRACT(QUARTER FROM date) = 2 GROUP BY country ORDER BY SUM(total_data_usage) DESC LIMIT 5

Question: {question}
SQL: 
"""

# Execute query
def run_query(sql, dfs):
    conn = duckdb.connect()
    conn.register("bo_tbl", dfs[0])
    conn.register("sub_details", dfs[1])
    conn.register("revenue", dfs[2])
    return conn.execute(sql).fetchdf()

# Streamlit app
def main():
    st.set_page_config(page_title="Data Analyst Chatbot", layout="wide")
    
    # Sidebar with info
    with st.sidebar:
        st.header("📊 Data Chatbot")
        st.markdown("Ask questions about:")
        st.markdown("- Daily usage & transactions")
        st.markdown("- Subscriber statistics")
        st.markdown("- Revenue metrics")
        st.divider()
        st.caption("Example queries:")
        st.code("What was the churn in AAA last month?")
        st.code("Compare net revenue across channels for DDD")
        st.code("Which country had highest data usage in Q2?")
    
    # Main content
    st.title("📈 Structured Data Analyst Chatbot")
    
    # Initialize session state
    if "history" not in st.session_state:
        st.session_state.history = []
    
    # Load resources
    dfs = load_data()
    qa_pipeline = load_model()
    
    # User input
    question = st.chat_input("Ask a question about the data...")
    
    if question:
        with st.spinner("Analyzing your question..."):
            try:
                # Generate SQL
                prompt = build_prompt(question)
                llm_response = qa_pipeline(
                    prompt, 
                    max_new_tokens=200,
                    temperature=0.1
                )[0]['generated_text']
                
                # Extract SQL from response
                sql = llm_response.split("SQL:")[-1].strip().split(";")[0] + ";"
                
                # Execute query
                result = run_query(sql, dfs)
                
                # Store in history
                st.session_state.history.append({
                    "question": question,
                    "sql": sql,
                    "result": result
                })
                
            except Exception as e:
                st.error(f"Error processing your question: {str(e)}")
    
    # Display results
    if st.session_state.history:
        latest = st.session_state.history[-1]
        
        st.subheader("Result")
        st.dataframe(latest["result"], use_container_width=True)
        
        with st.expander("See SQL Query"):
            st.code(latest["sql"], language="sql")
        
        with st.expander("Query History"):
            for i, entry in enumerate(reversed(st.session_state.history)):
                st.markdown(f"**Q{i+1}:** {entry['question']}")
                st.code(entry["sql"], language="sql")

if __name__ == "__main__":
    main()