"""Chat query page with multi-agent routing."""
import os
import streamlit as st
import httpx

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8080")

st.set_page_config(page_title="Query | OMNIMIND", page_icon="💬", layout="wide")
st.title("💬 Query OMNIMIND")
st.markdown("Ask anything — OMNIMIND will route your question to the best agent.")

ROUTE_ICONS = {"rag": "🗂️", "web_search": "🌐", "reasoning": "🤖"}
ROUTE_LABELS = {"rag": "RAG Retrieval", "web_search": "Web Search", "reasoning": "GPT-4 Reasoning"}

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.markdown("### Query Settings")
    force_route = st.selectbox(
        "Force route",
        ["auto", "rag", "web_search", "reasoning"],
        index=0,
        help="Override the automatic router with a specific agent.",
    )
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()


def render_assistant_meta(meta: dict) -> None:
    """Render route badge, latency, and sources for an assistant message."""
    if not meta:
        return
    route = meta.get("route", "unknown")
    icon = ROUTE_ICONS.get(route, "❓")
    label = ROUTE_LABELS.get(route, route)
    st.caption(f"{icon} **{label}**")

    routing_reason = meta.get("routing_reason")
    if routing_reason:
        st.caption(f"_Routing reason: {routing_reason}_")

    latency = meta.get("latency_ms")
    if latency is not None:
        try:
            st.caption(f"{float(latency):.0f}ms")
        except (TypeError, ValueError):
            st.caption(f"{latency}ms")

    sources = meta.get("sources") or []
    if sources:
        with st.expander(f"📎 Sources ({len(sources)})"):
            for s in sources:
                if isinstance(s, dict):
                    title = s.get("title") or s.get("source") or s.get("url") or str(s)
                    url = s.get("url")
                    if url:
                        st.markdown(f"- [{title}]({url})")
                    else:
                        st.markdown(f"- {title}")
                else:
                    st.markdown(f"- {s}")


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_assistant_meta(msg.get("meta") or {})


def call_query_api(prompt: str, force: str) -> dict | None:
    """POST to the query endpoint and return the parsed JSON."""
    payload = {
        "query": prompt,
        "force_route": None if force == "auto" else force,
    }
    try:
        with httpx.Client(timeout=120) as client:
            resp = client.post(f"{BACKEND_URL}/api/v1/query", json=payload)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPError as e:
        st.error(f"Query failed: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error during query: {e}")
        return None


prompt = st.chat_input("Ask anything about your documents...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("OMNIMIND is thinking..."):
        result = call_query_api(prompt, force_route)

    if result is not None:
        answer = result.get("answer") or result.get("response") or "_(no answer returned)_"
        meta = {
            "route": result.get("route", "unknown"),
            "sources": result.get("sources") or [],
            "latency_ms": result.get("latency_ms"),
            "routing_reason": result.get("routing_reason"),
        }
        st.session_state.messages.append({"role": "assistant", "content": answer, "meta": meta})
    else:
        st.session_state.messages.append(
            {"role": "assistant", "content": "_Sorry, the query failed. Please try again._", "meta": {}}
        )

    if len(st.session_state.messages) > 20:
        st.session_state.messages = st.session_state.messages[-20:]

    st.rerun()
