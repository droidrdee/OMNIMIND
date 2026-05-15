"""Reusable document preview component."""
import streamlit as st
import pandas as pd


def render_doc_table(docs: list) -> None:
    if not docs:
        st.info("No documents ingested yet. Upload some to get started.")
        return
    df = pd.DataFrame(docs)
    st.dataframe(df, use_container_width=True, hide_index=True)
