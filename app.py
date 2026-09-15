
import html
import json
import re
import tempfile
from pathlib import Path

import streamlit as st

from src.config import config
from src.pipeline import RAGPipeline
from src.ingest import DocumentIngestionPipeline

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SAMPLE_DIR = ROOT / "sample_data"

PAGE_MARKER = re.compile(r"\[Page (\d+)\]")

st.set_page_config(
    page_title="Hybrid RAG Engine",
    page_icon="🔎",
    layout="wide",
)



st.markdown(
    """
    <style>
    :root {
        --bg: #0e1013;
        --bg-alt: #14171c;
        --card: #191c22;
        --card-alt: #1e222a;
        --border: #2a2e37;
        --border-soft: #23262e;
        --text: #e7e9ee;
        --text-dim: #9aa0ac;
        --text-faint: #6b7280;
        --accent: #22d3ee;
        --accent-soft: rgba(34, 211, 238, 0.12);
        --good: #34d399;
        --good-soft: rgba(52, 211, 153, 0.12);
        --bad: #f87171;
        --bad-soft: rgba(248, 113, 113, 0.12);
    }

    #MainMenu, footer, header {visibility: hidden;}

    .stApp {
        background: var(--bg);
        color: var(--text);
    }

    section.main > div.block-container {
        max-width: 980px;
        padding-top: 2.4rem;
        padding-bottom: 3rem;
    }

    /* ---------------- Header ---------------- */
    .app-eyebrow {
        display: inline-block;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--accent);
        background: var(--accent-soft);
        border: 1px solid rgba(34, 211, 238, 0.25);
        border-radius: 999px;
        padding: 4px 12px;
        margin-bottom: 14px;
    }

    .app-title {
        font-size: 34px;
        font-weight: 750;
        color: var(--text);
        letter-spacing: -0.01em;
        margin: 0 0 8px 0;
        line-height: 1.15;
    }

    .app-subtitle {
        font-size: 14.5px;
        color: var(--text-dim);
        line-height: 1.6;
        max-width: 720px;
        margin-bottom: 18px;
    }

    .pipeline-strip {
        font-size: 13px;
        color: var(--text-dim);
        background: var(--card);
        border: 1px solid var(--border);
        border-left: 3px solid var(--accent);
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 26px;
        font-family: "SFMono-Regular", Consolas, Menlo, monospace;
        letter-spacing: 0.01em;
    }

    .pipeline-strip b { color: var(--text); font-weight: 600; }

    /* ---------------- Tabs ---------------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 1px solid var(--border);
        margin-bottom: 22px;
    }

    .stTabs [data-baseweb="tab"] {
        color: var(--text-dim);
        font-size: 14px;
        font-weight: 550;
        padding: 8px 4px;
    }

    .stTabs [aria-selected="true"] {
        color: var(--text) !important;
    }

    .stTabs [data-baseweb="tab-highlight"] {
        background-color: var(--accent) !important;
    }

    /* ---------------- Generic surfaces ---------------- */
    .section-label {
        font-size: 12px;
        font-weight: 650;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--text-faint);
        margin: 28px 0 10px 0;
    }

    .card-surface, .stTextInput > div > div, .stFileUploader > section {
        background: var(--card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
    }

    /* Upload expander */
    .streamlit-expanderHeader, [data-testid="stExpander"] summary {
        background: var(--card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        color: var(--text) !important;
        font-size: 13.5px !important;
        font-weight: 550 !important;
    }
    [data-testid="stExpander"] {
        border: none !important;
    }
    [data-testid="stExpander"] > div {
        background: var(--bg-alt);
        border: 1px solid var(--border-soft);
        border-top: none;
        border-radius: 0 0 10px 10px;
        padding: 4px 4px 12px 4px;
    }

    .stFileUploader label { color: var(--text-dim) !important; font-size: 13px !important; }
    .stFileUploader small { color: var(--text-faint) !important; }

    /* Text input */
    .stTextInput label { color: var(--text-dim) !important; font-size: 13.5px !important; font-weight: 500; }
    .stTextInput input {
        color: var(--text) !important;
        font-size: 14.5px !important;
    }
    .stTextInput input::placeholder { color: var(--text-faint) !important; }

    /* Buttons */
    .stButton > button {
        background: var(--accent) !important;
        color: #06222a !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 650 !important;
        font-size: 13.5px !important;
        padding: 0.5rem 1.1rem !important;
        transition: opacity 0.15s ease;
    }
    .stButton > button:hover { opacity: 0.88; }
    .stButton > button p { color: #06222a !important; }

    /* Secondary-looking button (Ingest document) gets a quieter treatment
       via container ordering — Streamlit doesn't expose per-button variants
       without a `type` kwarg, so this stays visually consistent by design. */

    /* ---------------- Status badges ---------------- */
    .status-row { display: flex; gap: 12px; margin: 4px 0 18px 0; flex-wrap: wrap; }

    .status-badge {
        flex: 1;
        min-width: 210px;
        display: flex;
        align-items: center;
        gap: 10px;
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 12px 16px;
    }

    .status-badge.ok { border-color: rgba(52, 211, 153, 0.35); }
    .status-badge.bad { border-color: rgba(248, 113, 113, 0.35); }

    .status-dot {
        width: 9px; height: 9px; border-radius: 999px; flex-shrink: 0;
    }
    .status-badge.ok .status-dot { background: var(--good); box-shadow: 0 0 8px rgba(52,211,153,0.6); }
    .status-badge.bad .status-dot { background: var(--bad); box-shadow: 0 0 8px rgba(248,113,113,0.6); }

    .status-text { display: flex; flex-direction: column; gap: 1px; }
    .status-label { font-size: 11.5px; color: var(--text-faint); font-weight: 550; letter-spacing: 0.02em; }
    .status-value { font-size: 15px; font-weight: 650; }
    .status-badge.ok .status-value { color: var(--good); }
    .status-badge.bad .status-value { color: var(--bad); }

    /* ---------------- Answer card ---------------- */
    .answer-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-left: 3px solid var(--accent);
        border-radius: 10px;
        padding: 20px 22px;
        line-height: 1.7;
        color: var(--text);
        font-size: 15px;
    }

    /* ---------------- Source cards ---------------- */
    .source-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 16px 18px;
        margin-bottom: 12px;
    }

    .source-head {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 10px;
        flex-wrap: wrap;
    }

    .source-title {
        font-size: 13.5px;
        font-weight: 650;
        color: var(--text);
    }

    .source-title .idx {
        display: inline-block;
        color: var(--accent);
        font-family: "SFMono-Regular", Consolas, Menlo, monospace;
        margin-right: 6px;
    }

    .source-score {
        font-size: 11.5px;
        color: var(--text-faint);
        font-family: "SFMono-Regular", Consolas, Menlo, monospace;
        white-space: nowrap;
    }

    .source-tags { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 9px; }

    .source-tag {
        font-size: 10.5px;
        font-weight: 600;
        letter-spacing: 0.02em;
        color: var(--text-dim);
        background: var(--bg-alt);
        border: 1px solid var(--border-soft);
        border-radius: 5px;
        padding: 3px 8px;
    }

    .source-tag.method { color: var(--accent); border-color: rgba(34,211,238,0.25); background: var(--accent-soft); }

    .source-text {
        font-size: 12.5px;
        color: var(--text-dim);
        margin-top: 12px;
        line-height: 1.6;
        white-space: pre-wrap;
        border-top: 1px solid var(--border-soft);
        padding-top: 10px;
    }

    /* ---------------- Evaluation table ---------------- */
    .eval-caption { color: var(--text-dim); font-size: 13px; margin-bottom: 16px; }
    [data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }

    /* Misc Streamlit chrome */
    .stAlert { border-radius: 10px !important; }
    hr { border-color: var(--border) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# Cached pipeline access


@st.cache_resource
def get_pipeline():
    return RAGPipeline()


def reload_pipeline():
    """Force the next get_pipeline() call to rebuild indexes from disk."""
    get_pipeline.clear()


def ingest_uploaded_file(uploaded_file) -> dict:
    """Save an uploaded file into sample_data/ and run it through the
    existing DocumentIngestionPipeline (parse -> chunk -> dedup -> index).

    Returns the real, post-ingestion chunk counts from BOTH indexes so the
    caller can verify the upload actually became searchable, instead of
    just assuming success because no exception was raised.
    """
    SAMPLE_DIR.mkdir(exist_ok=True)

    target = SAMPLE_DIR / uploaded_file.name
    if target.exists():
        stem, suffix = target.stem, target.suffix
        counter = 1
        while target.exists():
            target = SAMPLE_DIR / f"{stem}_{counter}{suffix}"
            counter += 1

    target.write_bytes(uploaded_file.getvalue())

    pipeline = DocumentIngestionPipeline()
    pipeline.ingest_file(target)

    return {
        "filename": target.name,
        "source_path": str(target),
        "sparse_total": len(pipeline.sparse_index.indexed_chunks),
        "dense_total": pipeline.dense_index.collection.count(),
    }


def extract_page_number(chunk_text: str):
    match = PAGE_MARKER.search(chunk_text)
    return int(match.group(1)) if match else None


def load_json(filename):
    path = DATA_DIR / filename
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluation_rows(report):
    if not report:
        return []
    systems = report.get("systems", []) if isinstance(report, dict) else report
    if isinstance(systems, dict):
        systems = [{"system": name, "summary": values} for name, values in systems.items()]

    rows = []
    for item in systems:
        if not isinstance(item, dict):
            continue
        name = item.get("system") or item.get("name")
        summary = item.get("summary", item)
        if not name or not isinstance(summary, dict):
            continue

        def metric(*keys):
            for key in keys:
                if key in summary:
                    return summary[key]
            return None

        rows.append(
            {
                "System": name,
                "R@1": metric("R@1", "recall@1"),
                "R@5": metric("R@5", "recall@5"),
                "R@10": metric("R@10", "recall@10"),
                "MRR": metric("MRR", "mrr"),
                "NDCG@5": metric("NDCG@5", "ndcg@5"),
            }
        )
    return rows



# Header


st.markdown('<div class="app-eyebrow">Hybrid Retrieval-Augmented Generation</div>', unsafe_allow_html=True)
st.markdown('<div class="app-title">Hybrid RAG Engine</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Enterprise document search with lexical + semantic retrieval, '
    'reciprocal rank fusion, cross-encoder reranking, and grounded, citation-verified answers — '
    'upload a document, ask a question, and inspect exactly which chunks the answer came from.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="pipeline-strip">
    <b>Sparse BM25</b> + <b>dense vector</b> retrieval &nbsp;→&nbsp;
    <b>Reciprocal Rank Fusion</b> &nbsp;→&nbsp;
    <b>cross-encoder reranking</b> &nbsp;→&nbsp;
    <b>grounded generation</b> (Groq) &nbsp;→&nbsp;
    <b>citation verification</b>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_search, tab_eval = st.tabs(["Search", "Evaluation"])



# Search tab


with tab_search:

    with st.expander("Upload a document to index"):
        uploaded_file = st.file_uploader(
            "PDF, TXT, HTML, or Markdown",
            type=["pdf", "txt", "html", "md"],
        )
        if uploaded_file is not None and st.button("Ingest document"):
            with st.spinner("Parsing, chunking, and indexing..."):
                try:
                    info = ingest_uploaded_file(uploaded_file)
                    reload_pipeline()
                    st.session_state["ingestion_status"] = ("success", info)
                    # Scope subsequent questions to the document 
                    st.session_state["active_document"] = {
                        "filename": info["filename"],
                        "source_path": info["source_path"],
                    }
                except Exception as exc:
                    st.session_state["ingestion_status"] = ("error", str(exc))

        # Persisted across reruns (e.g. after asking a question) so a real
        # ingestion failure — or a successful one — stays visible instead
        # of vanishing the moment the user does anything else.
        status = st.session_state.get("ingestion_status")
        if status:
            kind, payload = status
            if kind == "success":
                st.success(
                    f"Indexed '{payload['filename']}' — sparse index: "
                    f"{payload['sparse_total']} total chunks · dense index: "
                    f"{payload['dense_total']} total chunks. It is now "
                    f"searchable below."
                )
            else:
                st.error(f"Ingestion failed: {payload}")

    st.markdown('<div class="section-label">Ask a question</div>', unsafe_allow_html=True)

    active_document = st.session_state.get("active_document")
    if active_document:
        scope_col, clear_col = st.columns([5, 1])
        with scope_col:
            st.caption(f"Searching only: {active_document['filename']}")
        with clear_col:
            if st.button("Search all documents"):
                st.session_state["active_document"] = None
                active_document = None

    query = st.text_input(
        "Ask a question about the indexed documents",
        placeholder="e.g. What is the purpose of the employee Code of Conduct?",
        label_visibility="collapsed",
    )

    if st.button("Search", type="primary") and query.strip():
        with st.spinner("Retrieving, reranking, and generating an answer..."):
            try:
                scope_source_path = active_document["source_path"] if active_document else None
                result = get_pipeline().query(query.strip(), source_path=scope_source_path)
                st.session_state["last_result"] = result
            except Exception as exc:
                st.error(f"Query failed: {exc}")
                st.session_state["last_result"] = None

    result = st.session_state.get("last_result")

    if result:
        st.markdown('<div class="section-label">Answer</div>', unsafe_allow_html=True)

        context_ok = result.get("is_context_sufficient", False)
        verification = result.get("citation_verification", {}) or {}
        citation_ok = verification.get("is_valid", False)

        st.markdown(
            f"""
            <div class="status-row">
                <div class="status-badge {'ok' if context_ok else 'bad'}">
                    <div class="status-dot"></div>
                    <div class="status-text">
                        <div class="status-label">Context sufficient</div>
                        <div class="status-value">{'Yes' if context_ok else 'No'}</div>
                    </div>
                </div>
                <div class="status-badge {'ok' if citation_ok else 'bad'}">
                    <div class="status-dot"></div>
                    <div class="status-text">
                        <div class="status-label">Citations verified</div>
                        <div class="status-value">{'Yes' if citation_ok else 'No'}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not citation_ok and verification.get("flagged_issues"):
            st.warning(f"Flagged citation issues: {verification['flagged_issues']}")

        answer = result.get("answer", "No answer returned.")
        st.markdown(
            f'<div class="answer-card">{html.escape(str(answer)).replace(chr(10), "<br>")}</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-label">Retrieved &amp; reranked sources</div>', unsafe_allow_html=True)

        reranked = result.get("reranked_results", [])

        if not reranked:
            st.caption("No sources were retrieved for this query.")
        else:
            for idx, item in enumerate(reranked, 1):
                chunk = item.get("chunk")
                if chunk is None:
                    continue

                metadata = chunk.metadata
                page_number = extract_page_number(chunk.page_content)
                section_title = (metadata.custom_attributes or {}).get("section_title")
                contributing = item.get("previous_sources", [])
                score = item.get("rerank_score")
                score_text = f"score {score:.4f}" if isinstance(score, (int, float)) else "score —"

                title = Path(metadata.source_path).name if metadata.source_path else "Document"

                tags = [f'<span class="source-tag">chunk #{metadata.chunk_index}</span>']
                for method in contributing:
                    tags.append(f'<span class="source-tag method">{html.escape(method)}</span>')
                if page_number is not None:
                    tags.append(f'<span class="source-tag">page {page_number}</span>')
                if section_title:
                    tags.append(f'<span class="source-tag">{html.escape(str(section_title))}</span>')

                st.markdown(
                    f"""
                    <div class="source-card">
                        <div class="source-head">
                            <div class="source-title"><span class="idx">[{idx}]</span>{html.escape(title)}</div>
                            <div class="source-score">{html.escape(score_text)}</div>
                        </div>
                        <div class="source-tags">{''.join(tags)}</div>
                        <div class="source-text">{html.escape(chunk.page_content[:800])}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )



# Evaluation tab


with tab_eval:
    st.markdown('<div class="section-label">Retrieval evaluation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="eval-caption">Offline benchmark over the golden query set '
        '(<code>python -m evaluation.evaluate_retrieval</code>) — '
        'Recall@k, MRR, and NDCG@5 for each retrieval strategy.</div>',
        unsafe_allow_html=True,
    )

    report = load_json("retrieval_evaluation.json")

    if not report:
        st.info(
            "No evaluation report found yet. Run:\n\n"
            "`python -m evaluation.evaluate_retrieval`"
        )
    else:
        rows = evaluation_rows(report)
        if not rows:
            st.error("The evaluation report was found, but its metrics could not be parsed.")
        else:
            st.dataframe(rows, hide_index=True, use_container_width=True)
