"""Analytics dashboard for OMNIMIND routing and cost metrics."""
import os
import streamlit as st
import httpx
import pandas as pd
import plotly.express as px

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8080")

st.set_page_config(page_title="Analytics | OMNIMIND", page_icon="📊", layout="wide")
st.title("📊 Analytics")
st.markdown("Routing distribution and cost-savings dashboard.")

if st.button("🔄 Refresh"):
    st.rerun()


def fetch_analytics() -> dict | None:
    """GET analytics from the backend."""
    try:
        with httpx.Client(timeout=30) as client:
            resp = client.get(f"{BACKEND_URL}/api/v1/analytics")
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPError as e:
        st.error(f"Failed to fetch analytics: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error fetching analytics: {e}")
        return None


data = fetch_analytics()

if not data:
    st.warning("No analytics data available.")
    st.stop()

routing_counter = data.get("routing_counter") or {}
total_queries = int(data.get("total_queries") or sum(routing_counter.values()) or 0)
distribution = data.get("distribution") or {}

rag_count = int(routing_counter.get("rag", 0))
web_count = int(routing_counter.get("web_search", 0))
reasoning_count = int(routing_counter.get("reasoning", 0))


def pct(part: int) -> float:
    if total_queries <= 0:
        return 0.0
    return (part / total_queries) * 100.0


rag_pct = distribution.get("rag", pct(rag_count))
web_pct = distribution.get("web_search", pct(web_count))
reasoning_pct = distribution.get("reasoning", pct(reasoning_count))

try:
    rag_pct = float(rag_pct)
    web_pct = float(web_pct)
    reasoning_pct = float(reasoning_pct)
except (TypeError, ValueError):
    rag_pct, web_pct, reasoning_pct = pct(rag_count), pct(web_count), pct(reasoning_count)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Queries", f"{total_queries:,}")
m2.metric("RAG %", f"{rag_pct:.1f}%")
m3.metric("Web %", f"{web_pct:.1f}%")
m4.metric("Reasoning %", f"{reasoning_pct:.1f}%")

st.divider()

if routing_counter:
    df = pd.DataFrame(
        [{"route": k, "count": v} for k, v in routing_counter.items() if v]
    )
    if not df.empty:
        fig = px.pie(
            df,
            names="route",
            values="count",
            title="Routing Distribution",
            hole=0.4,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No routing data to plot yet.")
else:
    st.info("No routing counter data available.")

st.divider()
st.subheader("💰 Cost Savings Estimate")

naive_cost = total_queries * 0.03
actual_cost = (rag_count * 0.001) + (web_count * 0.002) + (reasoning_count * 0.03)
savings_pct = ((naive_cost - actual_cost) / naive_cost * 100.0) if naive_cost > 0 else 0.0

c1, c2, c3 = st.columns(3)
c1.metric("Naive cost (all GPT-4)", f"${naive_cost:.2f}")
c2.metric("Actual cost (routed)", f"${actual_cost:.2f}")
c3.metric("Estimated cost savings vs all-GPT-4", f"{savings_pct:.1f}%")

st.caption(
    "Cost model assumptions: RAG $0.001/query, Web Search $0.002/query, "
    "Reasoning $0.03/query, Naive baseline $0.03/query."
)
