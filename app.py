import json
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity

from ingestion.arxiv_client import ArxivClient
from ingestion.semantic_scholar_client import SemanticScholarClient
from parsing.pdf_parser import PDFParser, SentenceSplitter
from parsing.section_segmenter import SectionSegmenter
from claims.claim_extractor import ClaimExtractor
from claims.limitation_retriever import LimitationRetriever
from embedding.specter_embedder import SpecterEmbedder
from clustering.hdbscan_cluster import ClaimClusterer
from nli.nli_engine import NLIEngine
from gap_generation.generator import GapGenerator
from config.settings import (
    ARXIV_MAX_RESULTS, MIN_SENTENCE_LENGTH,
    COSINE_SIM_MIN, COSINE_SIM_MAX, NLI_CONFIDENCE_THRESHOLD,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Research Gap Finder",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Hero header */
.hero { padding: 1.5rem 0 0.5rem 0; }
.hero h1 { font-size: 2.2rem; font-weight: 700; margin-bottom: 0.2rem; }
.hero p  { color: #6b7280; font-size: 1rem; margin-top: 0; }

/* Gap cards */
.gap-card {
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin: 0.6rem 0;
    border-left: 5px solid #ccc;
    background: #fafafa;
}
.gap-card.contradiction { border-left-color: #dc2626; background: #fff5f5; }
.gap-card.limitation    { border-left-color: #d97706; background: #fffbeb; }
.gap-card.open_question { border-left-color: #2563eb; background: #eff6ff; }

.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-bottom: 0.5rem;
}
.badge.contradiction { background: #fee2e2; color: #b91c1c; }
.badge.limitation    { background: #fef3c7; color: #b45309; }
.badge.open_question { background: #dbeafe; color: #1d4ed8; }

.gap-text { font-size: 1rem; line-height: 1.6; color: #1f2937; margin: 0.4rem 0; }
.paper-pill {
    display: inline-block;
    background: #f3f4f6;
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.78rem;
    color: #374151;
    margin: 2px 3px 2px 0;
}

/* Metric enhancements */
[data-testid="metric-container"] {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}

/* Section headings */
.section-heading {
    font-size: 1.3rem;
    font-weight: 600;
    color: #111827;
    margin: 1.5rem 0 0.8rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid #e5e7eb;
}
</style>
""", unsafe_allow_html=True)

# ── Pipeline stage labels ─────────────────────────────────────────────────────
STAGES = [
    "Download papers from ArXiv",
    "Enrich via Semantic Scholar",
    "Extract claims (Claude Haiku)",
    "Embed claims (SPECTER2)",
    "Cluster claims (HDBSCAN)",
    "Detect contradictions (NLI)",
    "Retrieve limitations",
    "Synthesize research gaps (Claude Opus)",
]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 Research Gap Finder")
    st.markdown("Powered by **Claude API** + SPECTER2 + HDBSCAN + NLI")
    st.divider()
    query = st.text_input("Research topic:", "retrieval augmented generation",
                          help="Enter any ML/NLP/AI research topic")
    max_papers = st.slider("Papers to analyze:", min_value=3, max_value=100, value=10,
                           help="More papers = richer analysis but longer runtime")
    run = st.button("▶  Run Analysis", type="primary", use_container_width=True)
    st.divider()
    stage_placeholder = st.empty()
    st.caption("Models used:")
    st.caption("• Claim extraction: Claude Haiku 4.5")
    st.caption("• Gap synthesis: Claude Opus 4.7")
    st.caption("• Embeddings: SPECTER2")
    st.caption("• Contradiction: DeBERTa-v3 NLI")


def render_stages(current_stage: int):
    lines = ["**Pipeline progress:**\n"]
    for i, name in enumerate(STAGES):
        if i < current_stage:
            lines.append(f"✅ ~~{i+1}. {name}~~")
        elif i == current_stage and current_stage < len(STAGES):
            lines.append(f"⏳ **{i+1}. {name}**")
        else:
            lines.append(f"○ {i+1}. {name}")
    if current_stage >= len(STAGES):
        lines.append("\n🎉 **Analysis complete!**")
    stage_placeholder.markdown("\n\n".join(lines))


# ── Landing state ─────────────────────────────────────────────────────────────
if not run:
    render_stages(-1)
    st.markdown("""
<div class="hero">
  <h1>🔬 Research Gap Discovery</h1>
  <p>Automatically identify contradictions, limitations, and open questions across academic papers.</p>
</div>
""", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col1.info("**Step 1** — Enter a research topic in the sidebar and choose how many papers to analyze.")
    col2.info("**Step 2** — The pipeline downloads papers, extracts claims with Claude AI, clusters them semantically, and detects contradictions.")
    col3.info("**Step 3** — Claude Opus 4.7 synthesizes grounded research gaps with full citation traceability.")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — Download papers
# ══════════════════════════════════════════════════════════════════════════════
render_stages(0)
with st.spinner("Fetching papers from ArXiv..."):
    arxiv = ArxivClient()
    papers = arxiv.search_and_download(query, max_results=max_papers)

# ── Stage 2 — Semantic Scholar ─────────────────────────────────────────────
render_stages(1)
with st.spinner("Enriching metadata via Semantic Scholar..."):
    scholar = SemanticScholarClient()
    papers, extra_papers = scholar.enrich_and_expand(papers)
    if extra_papers:
        papers.extend(extra_papers)

# ── Parse all papers for the paper explorer ───────────────────────────────────
parser = PDFParser()
segmenter = SectionSegmenter()
splitter = SentenceSplitter()

paper_data = []
for paper in papers:
    text = parser.extract_text(paper["pdf_path"])
    sections = segmenter.segment(text)
    paper_data.append({
        "title": paper["title"],
        "pdf_path": paper["pdf_path"],
        "citation_count": paper.get("citation_count"),
        "year": paper.get("year"),
        "venue": paper.get("venue", ""),
        "sections": sections,
        "text": text,
    })

# ══════════════════════════════════════════════════════════════════════════════
# PAPER EXPLORER (shown while pipeline runs below)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f'<div class="section-heading">📄 Papers Collected ({len(paper_data)})</div>',
            unsafe_allow_html=True)

for i, pd in enumerate(paper_data):
    citations = f" · {pd['citation_count']:,} citations" if pd.get("citation_count") else ""
    year = f" · {pd['year']}" if pd.get("year") else ""
    venue = f" · {pd['venue']}" if pd.get("venue") else ""

    with st.expander(f"**{i+1}.** {pd['title']} {year}{citations}{venue}"):
        section_names = list(pd["sections"].keys())
        st.caption(f"Sections: {', '.join(section_names) if section_names else 'none detected'}")

        # Abstract or introduction preview
        preview_section = next(
            (s for s in ["abstract", "introduction"] if s in pd["sections"]), None
        )
        if preview_section:
            preview = pd["sections"][preview_section][:600].strip()
            if len(pd["sections"][preview_section]) > 600:
                preview += "…"
            st.markdown(f"**{preview_section.title()}:** {preview}")
        else:
            preview = pd["text"][:600].strip()
            if len(pd["text"]) > 600:
                preview += "…"
            st.markdown(preview)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — Claim Extraction (Claude Haiku — batched per paper)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-heading">⚙️ Pipeline</div>', unsafe_allow_html=True)

render_stages(2)
with st.status("Extracting claims with Claude Haiku...", expanded=True) as status:
    extractor = ClaimExtractor()
    all_claims = []
    progress = st.progress(0, text="Starting claim extraction…")

    for idx, pd in enumerate(paper_data):
        progress.progress(
            idx / len(paper_data),
            text=f"Paper {idx+1}/{len(paper_data)}: {pd['title'][:55]}…",
        )
        sentence_items = []
        for section_name, content in pd["sections"].items():
            for s in splitter.split(content):
                if len(s) >= MIN_SENTENCE_LENGTH:
                    sentence_items.append((s, pd["title"], section_name))

        paper_claims = extractor.extract_all(sentence_items)
        all_claims.extend(paper_claims)

    progress.progress(1.0, text=f"Extracted {len(all_claims)} claims")
    status.update(
        label=f"✅ Extracted {len(all_claims):,} claims from {len(paper_data)} papers",
        state="complete",
    )

if not all_claims:
    st.error("No claims extracted. Try a different topic or increase the paper count.")
    st.stop()

# ── Stage 4 — Embedding ───────────────────────────────────────────────────────
render_stages(3)
with st.status("Embedding claims with SPECTER2…", expanded=False) as status:
    embedder = SpecterEmbedder()
    embeddings = embedder.encode(all_claims)
    status.update(label=f"✅ Embedded {len(all_claims):,} claims (768-dim)", state="complete")

# ── Stage 5 — Clustering ──────────────────────────────────────────────────────
render_stages(4)
with st.status("Clustering with HDBSCAN…", expanded=False) as status:
    clusterer = ClaimClusterer()
    labels = clusterer.cluster(embeddings)
    clusters = clusterer.group_clusters(all_claims, labels)
    if not clusters:
        clusters = {0: all_claims}
    status.update(label=f"✅ Formed {len(clusters)} semantic clusters", state="complete")

# ── Stage 6 — Contradiction Detection ────────────────────────────────────────
render_stages(5)
with st.status("Detecting contradictions via NLI…", expanded=False) as status:
    nli = NLIEngine()
    all_contradictions = []

    for cid, cl in clusters.items():
        idxs = [i for i, l in enumerate(labels) if l == cid]
        if len(idxs) < 2:
            continue
        emb = embeddings[idxs]
        for i in range(len(cl)):
            for j in range(i + 1, len(cl)):
                sim = cosine_similarity([emb[i]], [emb[j]])[0][0]
                if sim < COSINE_SIM_MIN or sim > COSINE_SIM_MAX:
                    continue
                rel, conf = nli.predict(cl[i]["text"], cl[j]["text"])
                if conf > NLI_CONFIDENCE_THRESHOLD and rel == "contradiction":
                    all_contradictions.append((cl[i], cl[j], conf))

    status.update(label=f"✅ Found {len(all_contradictions)} contradictions", state="complete")

# ── Stage 7 — Limitations ─────────────────────────────────────────────────────
render_stages(6)
with st.status("Retrieving limitation-adjacent claims…", expanded=False) as status:
    retriever = LimitationRetriever(embedder, clusterer)
    limitation_claims, limitation_clusters = retriever.retrieve(all_claims, embeddings)
    status.update(
        label=f"✅ Found {len(limitation_claims)} limitation claims in {len(limitation_clusters)} clusters",
        state="complete",
    )

# ── Stage 8 — Gap Generation (Claude Opus 4.7) ───────────────────────────────
render_stages(7)
with st.status("Synthesizing research gaps with Claude Opus 4.7…", expanded=True) as status:
    gap_generator = GapGenerator()
    gaps = []

    cluster_tasks = [cl for cl in clusters.values() if len(cl) >= 2]
    limitation_tasks = [cl for cl in limitation_clusters.values() if len(cl) >= 2]
    total_tasks = len(cluster_tasks) + (1 if all_contradictions else 0) + len(limitation_tasks)
    done = 0
    progress = st.progress(0, text="Generating research gaps…")

    for cid, cl in clusters.items():
        if len(cl) < 2:
            continue
        gap = gap_generator.generate_gap(cl)
        gaps.append(gap)
        done += 1
        progress.progress(done / max(total_tasks, 1), text=f"Gap {done}/{total_tasks}")

    if all_contradictions:
        gap = gap_generator.generate_gap([], contradictions=all_contradictions)
        gaps.append(gap)
        done += 1
        progress.progress(done / max(total_tasks, 1), text=f"Gap {done}/{total_tasks}")

    for cid, cl in limitation_clusters.items():
        if len(cl) < 2:
            continue
        gap = gap_generator.generate_gap(cl)
        gap["type"] = "limitation"
        gaps.append(gap)
        done += 1
        progress.progress(done / max(total_tasks, 1), text=f"Gap {done}/{total_tasks}")

    progress.progress(1.0, text="Done")
    status.update(label=f"✅ Synthesized {len(gaps)} research gaps", state="complete")

render_stages(len(STAGES))  # all done

# ══════════════════════════════════════════════════════════════════════════════
# RESULTS DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown(f'<div class="section-heading">📊 Results: "{query}"</div>', unsafe_allow_html=True)

# ── Summary metrics ───────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("📄 Papers", len(papers))
c2.metric("💬 Claims", f"{len(all_claims):,}")
c3.metric("🗂 Clusters", len(clusters))
c4.metric("⚡ Contradictions", len(all_contradictions))
c5.metric("🔍 Research Gaps", len(gaps))

# ── Categorise gaps ───────────────────────────────────────────────────────────
contradiction_gaps = [g for g in gaps if g.get("type") == "contradiction"]
limitation_gaps    = [g for g in gaps if g.get("type") == "limitation"]
open_question_gaps = [g for g in gaps if g.get("type", "open_question") == "open_question"]

# ── Charts ────────────────────────────────────────────────────────────────────
col_chart1, col_chart2 = st.columns([1, 2])

with col_chart1:
    st.markdown("**Gap Type Distribution**")
    labels_pie  = ["Open Questions", "Contradictions", "Limitations"]
    values_pie  = [len(open_question_gaps), len(contradiction_gaps), len(limitation_gaps)]
    colors_pie  = ["#2563eb", "#dc2626", "#d97706"]
    if sum(values_pie) > 0:
        fig_pie = go.Figure(data=[go.Pie(
            labels=labels_pie,
            values=values_pie,
            hole=0.45,
            marker=dict(colors=colors_pie, line=dict(color="white", width=2)),
            textinfo="label+percent",
            hovertemplate="%{label}: %{value} gaps<extra></extra>",
        )])
        fig_pie.update_layout(
            showlegend=False,
            margin=dict(t=10, b=10, l=10, r=10),
            height=230,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

with col_chart2:
    st.markdown("**Top Clusters by Size (claims per cluster)**")
    cluster_sizes = sorted(
        [(cid, len(cl)) for cid, cl in clusters.items()],
        key=lambda x: x[1], reverse=True
    )[:12]
    if cluster_sizes:
        cids, csizes = zip(*cluster_sizes)
        fig_bar = go.Figure(data=[go.Bar(
            x=[f"C{cid}" for cid in cids],
            y=csizes,
            marker_color="#6366f1",
            hovertemplate="Cluster %{x}: %{y} claims<extra></extra>",
        )])
        fig_bar.update_layout(
            xaxis_title=None,
            yaxis_title="Claims",
            margin=dict(t=10, b=30, l=40, r=10),
            height=230,
            plot_bgcolor="white",
            yaxis=dict(gridcolor="#f3f4f6"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# ── Export button ─────────────────────────────────────────────────────────────
export_data = {
    "query": query,
    "papers_analyzed": len(papers),
    "total_claims": len(all_claims),
    "clusters": len(clusters),
    "contradictions": len(all_contradictions),
    "gaps": gaps,
}
col_exp, _ = st.columns([1, 3])
with col_exp:
    st.download_button(
        label="⬇ Export Results (JSON)",
        data=json.dumps(export_data, indent=2),
        file_name=f"research_gaps_{query.replace(' ', '_')}.json",
        mime="application/json",
    )

# ── Gap search/filter ─────────────────────────────────────────────────────────
search_query = st.text_input("🔍 Filter gaps by keyword:", placeholder="e.g. evaluation, dataset, efficiency…")

# ── Gap rendering helper ──────────────────────────────────────────────────────
TYPE_LABELS = {
    "contradiction": "Contradiction",
    "limitation":    "Limitation",
    "open_question": "Open Question",
}
TYPE_CSS = {
    "contradiction": "contradiction",
    "limitation":    "limitation",
    "open_question": "open_question",
}


def render_gap(gap: dict, index: int):
    gap_type  = gap.get("type", "open_question")
    css_class = TYPE_CSS.get(gap_type, "open_question")
    label     = TYPE_LABELS.get(gap_type, "Gap")

    st.markdown(f"""
<div class="gap-card {css_class}">
  <span class="badge {css_class}">{label}</span>
  <div class="gap-text">{gap['gap']}</div>
</div>
""", unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 1])
    with col_a:
        if gap.get("source_papers"):
            pills_html = "".join(f'<span class="paper-pill">{p[:60]}</span>' for p in gap["source_papers"])
            st.markdown(f"**Sources:** {pills_html}", unsafe_allow_html=True)
    with col_b:
        if gap.get("supporting_claims"):
            with st.expander(f"View {len(gap['supporting_claims'])} supporting claims"):
                for c in gap["supporting_claims"]:
                    st.markdown(f"- *{c}*")


def filter_gaps(gap_list: list) -> list:
    if not search_query:
        return gap_list
    kw = search_query.lower()
    return [g for g in gap_list if kw in g.get("gap", "").lower()]


# ── Tabbed gap explorer ───────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    f"All Gaps ({len(gaps)})",
    f"Open Questions ({len(open_question_gaps)})",
    f"Contradictions ({len(contradiction_gaps)})",
    f"Limitations ({len(limitation_gaps)})",
])

with tab1:
    filtered = filter_gaps(gaps)
    if not filtered:
        st.info("No gaps match your search query.")
    for i, gap in enumerate(filtered):
        render_gap(gap, i)

with tab2:
    filtered = filter_gaps(open_question_gaps)
    if filtered:
        for i, gap in enumerate(filtered):
            render_gap(gap, i)
    else:
        st.info("No open question gaps found." if not search_query else "No matches for your search.")

with tab3:
    filtered = filter_gaps(contradiction_gaps)
    if filtered:
        for i, gap in enumerate(filtered):
            render_gap(gap, i)
    else:
        st.info(
            "No contradiction-based gaps found at current thresholds."
            if not search_query else "No matches for your search."
        )

with tab4:
    filtered = filter_gaps(limitation_gaps)
    if filtered:
        for i, gap in enumerate(filtered):
            render_gap(gap, i)
    else:
        st.info("No limitation-based gaps found." if not search_query else "No matches for your search.")

# ── Cluster explorer ──────────────────────────────────────────────────────────
st.divider()
with st.expander("🗂 Explore Semantic Clusters"):
    top_n = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)[:20]
    for cid, cl in top_n:
        st.markdown(f"**Cluster {cid}** — {len(cl)} claims")
        for c in cl[:4]:
            st.markdown(f"&nbsp;&nbsp;· {c['text'][:120]}", unsafe_allow_html=True)
        if len(cl) > 4:
            st.caption(f"… and {len(cl) - 4} more claims")
        st.markdown("---")
