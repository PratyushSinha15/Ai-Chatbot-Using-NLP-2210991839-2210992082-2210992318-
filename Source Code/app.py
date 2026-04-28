import os
import re
import pandas as pd
import sqlite3
import streamlit as st
from dotenv import load_dotenv

# LangChain / LLM imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

# ----------------------------
# SQLite Setup
# ----------------------------
DB_FILE = "data.db"

def get_connection():
    return sqlite3.connect(DB_FILE)

def load_csv_to_sqlite(file, table_name):
    df = pd.read_csv(file)
    conn = get_connection()
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()

def load_excel_to_sqlite(file, table_name):
    df = pd.read_excel(file)
    conn = get_connection()
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()

def run_sql(sql):
    conn = get_connection()
    df = pd.read_sql_query(sql, conn)
    conn.close()
    return df

# ----------------------------
# LLM Setup
# ----------------------------
API_KEY = os.getenv("API_KEY")
os.environ['GOOGLE_API_KEY'] = API_KEY or "AIzaSyBY-hNMJZQpwxWnJFtc6VyGHAsvU-BALFI"

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key="AIzaSyBY-hNMJZQpwxWnJFtc6VyGHAsvU-BALFI"
)

# SQLite URI
SQLALCHEMY_URI = "sqlite:///data.db"

db = SQLDatabase.from_uri(SQLALCHEMY_URI)
generate_query = create_sql_query_chain(llm, db)
execute_query = QuerySQLDataBaseTool(db=db)

# ----------------------------
# Prompt
# ----------------------------
prompt = PromptTemplate.from_template("""
You are a SQL assistant for SQLite database.

If result is empty → say "No result found"

Question: {question}
SQL Query: {query}
SQL Result: {result}

Answer:
""")

chain = prompt | llm | StrOutputParser()

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="SQLite Chatbot", layout="wide")
st.title("SQLite Chatbot (CSV + Excel Support)")

# Upload Section
st.sidebar.header("Upload Data")

csv_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
excel_file = st.sidebar.file_uploader("Upload Excel", type=["xlsx"])

if csv_file:
    load_csv_to_sqlite(csv_file, "my_table")
    st.sidebar.success("CSV Loaded into SQLite as 'my_table'")

if excel_file:
    load_excel_to_sqlite(excel_file, "my_table")
    st.sidebar.success("Excel Loaded into SQLite as 'my_table'")

# User Query
user_input = st.text_input("Ask your question")

if st.button("Run Query"):
    if user_input.strip():
        with st.spinner("Processing..."):
            try:
                # Generate SQL
                sql = generate_query.invoke({"question": user_input})

                if isinstance(sql, dict):
                    sql = sql.get("query", "")

                # Execute SQL
                df = run_sql(sql)

                # Generate Answer
                result_text = df.head(50).to_csv(index=False) if not df.empty else "No rows"
                answer = chain.invoke({
                    "question": user_input,
                    "query": sql,
                    "result": result_text
                })

                # Display
                st.markdown("### Answer")
                st.write(answer)

                st.code(sql)

                if not df.empty:
                    st.dataframe(df)

                    # Export buttons
                    st.download_button(
                        "Download CSV",
                        df.to_csv(index=False),
                        "result.csv",
                        "text/csv"
                    )

                    st.download_button(
                        "Download Excel",
                        df.to_excel(index=False),
                        "result.xlsx"
                    )
                else:
                    st.warning("No result found")

            except Exception as e:
                st.error(f"Error: {e}")

st.caption("SQLite + CSV + Excel + LLM 🚀")