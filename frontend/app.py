import streamlit as st

st.set_page_config(page_title="OMNIMIND", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

st.sidebar.title("🧠 OMNIMIND")
st.sidebar.markdown("*Multi-Agent AI Research Platform*")
st.sidebar.divider()
st.sidebar.markdown("**Navigation:**")
st.sidebar.page_link("pages/01_upload.py", label="📤 Upload Documents")
st.sidebar.page_link("pages/02_query.py", label="💬 Query")
st.sidebar.page_link("pages/03_analytics.py", label="📊 Analytics")

st.title("Welcome to OMNIMIND 🧠")
st.markdown("""
A multi-agent AI research platform with intelligent query routing across:

- 📄 **RAG Retrieval** — searches your uploaded documents
- 🔍 **Web Search Agent** — fetches live information (Tavily + LangChain ReAct)
- 🤖 **GPT-4 Reasoning** — deep multi-hop analysis

Use the sidebar to upload documents and ask questions.
""")

st.divider()
col1, col2, col3 = st.columns(3)
col1.metric("Avg Latency Target", "< 3s")
col2.metric("Cost Reduction", "~40%")
col3.metric("Supported Formats", "PDF / PNG / JPG / TIFF")
