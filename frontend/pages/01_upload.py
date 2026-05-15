"""Document upload and management page."""
import os
import streamlit as st
import httpx
import pandas as pd

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8080")

st.set_page_config(page_title="Upload | OMNIMIND", page_icon="📤", layout="wide")
st.title("📤 Upload Documents")
st.markdown("Upload PDFs or images to be ingested into the OMNIMIND knowledge base.")


def fetch_documents() -> list:
    """Fetch list of ingested documents from the backend."""
    try:
        with httpx.Client(timeout=30) as client:
            resp = client.get(f"{BACKEND_URL}/api/v1/documents")
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict):
                return data.get("documents", []) or data.get("data", []) or []
            return data or []
    except httpx.HTTPError as e:
        st.error(f"Failed to fetch documents: {e}")
        return []
    except Exception as e:
        st.error(f"Unexpected error fetching documents: {e}")
        return []


def delete_document(doc_id: str) -> bool:
    """Delete a document by ID."""
    try:
        with httpx.Client(timeout=30) as client:
            resp = client.delete(f"{BACKEND_URL}/api/v1/documents/{doc_id}")
            resp.raise_for_status()
            return True
    except httpx.HTTPError as e:
        st.error(f"Failed to delete document {doc_id}: {e}")
        return False
    except Exception as e:
        st.error(f"Unexpected error deleting document: {e}")
        return False


def ingest_file(name: str, content: bytes, content_type: str) -> dict | None:
    """POST a single file to the ingest endpoint."""
    try:
        with httpx.Client(timeout=120) as client:
            files = {"file": (name, content, content_type)}
            resp = client.post(f"{BACKEND_URL}/api/v1/ingest", files=files)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPError as e:
        st.error(f"Failed to ingest {name}: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error ingesting {name}: {e}")
        return None


if "doc_list" not in st.session_state:
    st.session_state.doc_list = fetch_documents()

uploaded_files = st.file_uploader(
    "Upload documents",
    type=["pdf", "png", "jpg", "jpeg", "tiff", "tif"],
    accept_multiple_files=True,
)

col_a, col_b = st.columns([1, 5])
with col_a:
    ingest_btn = st.button("Ingest Files", type="primary", disabled=not uploaded_files)

if ingest_btn and uploaded_files:
    progress = st.progress(0.0, text="Starting ingestion...")
    total = len(uploaded_files)
    for idx, uf in enumerate(uploaded_files):
        progress.progress(idx / total, text=f"Ingesting {uf.name}...")
        content = uf.read()
        content_type = uf.type or "application/octet-stream"
        result = ingest_file(uf.name, content, content_type)
        if result is not None:
            chunks = result.get("chunks_created", result.get("chunks", "?"))
            method = result.get("method", result.get("ingestion_method", "unknown"))
            st.success(
                f"**{uf.name}** ingested — chunks: `{chunks}` | method: :blue-background[{method}]"
            )
        progress.progress((idx + 1) / total, text=f"Completed {uf.name}")
    progress.empty()
    st.session_state.doc_list = fetch_documents()

st.divider()
st.subheader("📁 Ingested Documents")

refresh_col, _ = st.columns([1, 6])
with refresh_col:
    if st.button("🔄 Refresh"):
        st.session_state.doc_list = fetch_documents()
        st.rerun()

docs = st.session_state.doc_list or []

if not docs:
    st.info("No documents ingested yet. Upload some files above to get started.")
else:
    try:
        df = pd.DataFrame(docs)
        st.dataframe(df, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Failed to render documents table: {e}")

    st.markdown("**Delete documents:**")
    for i, doc in enumerate(docs):
        doc_id = str(doc.get("doc_id") or doc.get("id") or doc.get("document_id") or "")
        if not doc_id:
            continue
        name = doc.get("filename") or doc.get("name") or doc_id
        c1, c2 = st.columns([5, 1])
        with c1:
            st.caption(f"`{doc_id}` — {name}")
        with c2:
            if st.button(f"Delete {doc_id[:8]}", key=f"del_{doc_id}_{i}"):
                if delete_document(doc_id):
                    st.session_state.doc_list = fetch_documents()
                    st.rerun()
