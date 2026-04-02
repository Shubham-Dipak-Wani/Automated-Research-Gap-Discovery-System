import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import json

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

st.set_page_config(page_title="Research Gap Finder", layout="wide")

st.title("Research Gap Discovery System")
st.markdown("Analyze research papers to find contradictions, limitations, and open questions.")

# --- Sidebar config ---
with st.sidebar:
    st.header("Settings")
    query = st.text_input("Research topic:", "retrieval augmented generation")
    max_papers = st.slider("Max papers from ArXiv:", min_value=3, max_value=100, value=10)
    run = st.button("Run Analysis", type="primary", use_container_width=True)
    st.divider()
    st.markdown("""
    **Pipeline stages:**
    1. Download papers
    2. Enrich via Semantic Scholar
    3. Extract claims (Mistral 7B)
    4. Embed claims (SPECTER2)
    5. Cluster claims (HDBSCAN)
    6. Detect contradictions (NLI)
    7. Retrieve limitations
    8. Generate research gaps
    """)

if not run:
    st.info("Enter a research topic and click **Run Analysis** to start.")
    st.stop()

# ============================================================
# PIPELINE — each stage updates the UI as it completes
# ============================================================

# --- Stage 1: Paper Collection ---
with st.status("Downloading papers from ArXiv...", expanded=True) as status:
    arxiv = ArxivClient()
    papers = arxiv.search_and_download(query, max_results=max_papers)
    st.write(f"Downloaded **{len(papers)}** papers from ArXiv")

    status.update(label="Enriching with Semantic Scholar...", state="running")
    scholar = SemanticScholarClient()
    papers, extra_papers = scholar.enrich_and_expand(papers)
    if extra_papers:
        st.write(f"Discovered **{len(extra_papers)}** additional papers via citation graph")
        papers.extend(extra_papers)

    status.update(label=f"Collected {len(papers)} papers", state="complete")

# Show paper list
with st.expander(f"Papers collected ({len(papers)})", expanded=False):
    for i, p in enumerate(papers):
        citation = f" — {p.get('citation_count', '?')} citations" if p.get('citation_count') else ""
        st.write(f"{i+1}. **{p['title']}**{citation}")

# --- Stage 2: Claim Extraction ---
with st.status("Extracting claims from papers...", expanded=True) as status:
    parser = PDFParser()
    segmenter = SectionSegmenter()
    splitter = SentenceSplitter()
    extractor = ClaimExtractor()

    all_claims = []
    progress = st.progress(0, text="Starting claim extraction...")

    for idx, paper in enumerate(papers):
        progress.progress(
            (idx) / len(papers),
            text=f"Paper {idx+1}/{len(papers)}: {paper['title'][:50]}..."
        )

        text = parser.extract_text(paper["pdf_path"])
        sections = segmenter.segment(text)

        for section_name, content in sections.items():
            sentences = splitter.split(content)
            for s in sentences:
                if len(s) < MIN_SENTENCE_LENGTH:
                    continue
                claims = extractor.extract_from_sentence(
                    s, paper_title=paper["title"], section=section_name,
                )
                all_claims.extend(claims)

    progress.progress(1.0, text="Claim extraction complete")
    st.write(f"Extracted **{len(all_claims)}** claims across {len(papers)} papers")
    status.update(label=f"Extracted {len(all_claims)} claims", state="complete")

if not all_claims:
    st.error("No claims extracted. Try a different topic or more papers.")
    st.stop()

# --- Stage 3: Embedding ---
with st.status("Embedding claims with SPECTER2...", expanded=False) as status:
    embedder = SpecterEmbedder()
    embeddings = embedder.encode(all_claims)
    status.update(label=f"Embedded {len(all_claims)} claims (768-d vectors)", state="complete")

# --- Stage 4: Clustering ---
with st.status("Clustering claims with HDBSCAN...", expanded=False) as status:
    clusterer = ClaimClusterer()
    labels = clusterer.cluster(embeddings)
    clusters = clusterer.group_clusters(all_claims, labels)
    if not clusters:
        clusters = {0: all_claims}
    status.update(label=f"Formed {len(clusters)} clusters", state="complete")

# --- Stage 5: Contradiction Detection ---
with st.status("Detecting contradictions via NLI...", expanded=True) as status:
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

    if all_contradictions:
        st.write(f"Found **{len(all_contradictions)}** contradictions")
    else:
        st.write("No contradictions detected at current thresholds")
    status.update(label=f"NLI complete — {len(all_contradictions)} contradictions", state="complete")

# --- Stage 6: Limitation Retrieval ---
with st.status("Retrieving limitation-adjacent claims...", expanded=False) as status:
    retriever = LimitationRetriever(embedder, clusterer)
    limitation_claims, limitation_clusters = retriever.retrieve(all_claims, embeddings)
    status.update(
        label=f"Found {len(limitation_claims)} limitation claims in {len(limitation_clusters)} clusters",
        state="complete"
    )

# --- Stage 7: Gap Generation ---
with st.status("Generating research gaps with Mistral...", expanded=True) as status:
    gap_generator = GapGenerator()
    gaps = []

    # Count total gap generation tasks
    cluster_tasks = [cl for cl in clusters.values() if len(cl) >= 2]
    limitation_tasks = [cl for cl in limitation_clusters.values() if len(cl) >= 2]
    total_tasks = len(cluster_tasks) + (1 if all_contradictions else 0) + len(limitation_tasks)

    progress = st.progress(0, text="Generating gaps...")
    done = 0

    # Gaps from topic clusters
    for cid, cl in clusters.items():
        if len(cl) < 2:
            continue
        gap = gap_generator.generate_gap(cl)
        gaps.append(gap)
        done += 1
        progress.progress(done / max(total_tasks, 1), text=f"Gap {done}/{total_tasks}...")

    # Gaps from contradictions
    if all_contradictions:
        gap = gap_generator.generate_gap([], contradictions=all_contradictions)
        gaps.append(gap)
        done += 1
        progress.progress(done / max(total_tasks, 1), text=f"Gap {done}/{total_tasks}...")

    # Gaps from limitations
    for cid, cl in limitation_clusters.items():
        if len(cl) < 2:
            continue
        gap = gap_generator.generate_gap(cl)
        gap["type"] = "limitation"
        gaps.append(gap)
        done += 1
        progress.progress(done / max(total_tasks, 1), text=f"Gap {done}/{total_tasks}...")

    progress.progress(1.0, text="Gap generation complete")
    status.update(label=f"Generated {len(gaps)} research gaps", state="complete")

# ============================================================
# RESULTS
# ============================================================

st.divider()
st.header("Results")

# --- Summary metrics ---
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Papers", len(papers))
c2.metric("Claims", len(all_claims))
c3.metric("Clusters", len(clusters))
c4.metric("Contradictions", len(all_contradictions))
c5.metric("Gaps Found", len(gaps))

st.divider()

# --- Tabs for different gap types ---
contradiction_gaps = [g for g in gaps if g.get("type") == "contradiction"]
limitation_gaps = [g for g in gaps if g.get("type") == "limitation"]
open_question_gaps = [g for g in gaps if g.get("type", "open_question") == "open_question"]

tab1, tab2, tab3, tab4 = st.tabs([
    f"All Gaps ({len(gaps)})",
    f"Open Questions ({len(open_question_gaps)})",
    f"Contradictions ({len(contradiction_gaps)})",
    f"Limitations ({len(limitation_gaps)})",
])


def render_gap(gap, index):
    gap_type = gap.get("type", "open_question")
    type_colors = {
        "contradiction": "red",
        "limitation": "orange",
        "open_question": "blue",
    }
    type_labels = {
        "contradiction": "Contradiction",
        "limitation": "Limitation",
        "open_question": "Open Question",
    }

    color = type_colors.get(gap_type, "blue")
    label = type_labels.get(gap_type, "Gap")

    with st.container(border=True):
        st.markdown(f"**:{color}[{label}]** — Gap #{index + 1}")
        st.markdown(f"> {gap['gap']}")

        col_a, col_b = st.columns(2)
        with col_a:
            if gap.get("source_papers"):
                st.markdown("**Source Papers:**")
                for p in gap["source_papers"]:
                    st.markdown(f"- {p}")
        with col_b:
            if gap.get("supporting_claims"):
                with st.expander("Supporting Claims"):
                    for c in gap["supporting_claims"]:
                        st.markdown(f"- _{c}_")


with tab1:
    for i, gap in enumerate(gaps):
        render_gap(gap, i)

with tab2:
    if open_question_gaps:
        for i, gap in enumerate(open_question_gaps):
            render_gap(gap, i)
    else:
        st.info("No open question gaps found.")

with tab3:
    if contradiction_gaps:
        for i, gap in enumerate(contradiction_gaps):
            render_gap(gap, i)
    else:
        st.info("No contradiction-based gaps found at current thresholds.")

with tab4:
    if limitation_gaps:
        for i, gap in enumerate(limitation_gaps):
            render_gap(gap, i)
    else:
        st.info("No limitation-based gaps found.")

# --- Cluster explorer ---
st.divider()
with st.expander("Explore Clusters"):
    for cid, cl in clusters.items():
        st.markdown(f"**Cluster {cid}** ({len(cl)} claims)")
        for c in cl[:5]:
            st.markdown(f"- {c['text']}")
        if len(cl) > 5:
            st.caption(f"... and {len(cl) - 5} more claims")
        st.divider()
