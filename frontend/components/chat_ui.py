"""Reusable chat UI components for OMNIMIND."""
import streamlit as st

ROUTE_ICONS = {"rag": "🗂️", "web_search": "🌐", "reasoning": "🤖"}
ROUTE_LABELS = {"rag": "RAG Retrieval", "web_search": "Web Search", "reasoning": "GPT-4 Reasoning"}


def render_route_badge(route: str) -> None:
    icon = ROUTE_ICONS.get(route, "❓")
    label = ROUTE_LABELS.get(route, route)
    st.caption(f"{icon} **{label}**")


def render_sources(sources: list) -> None:
    if not sources:
        return
    with st.expander(f"📎 Sources ({len(sources)})"):
        for s in sources:
            st.markdown(f"- {s}")
