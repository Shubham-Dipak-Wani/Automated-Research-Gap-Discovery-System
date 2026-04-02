import streamlit as st
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

st.set_page_config(page_title="Research Gap Finder", layout="wide")

st.title("Research Gap Discovery System")
st.caption("Analyze research papers to find contradictions, limitations, and open questions.")

# --- Sidebar ---
with st.sidebar:
    st.header("Settings")
    query = st.text_input("Research topic:", "retrieval augmented generation")
    max_papers = st.slider("Papers to fetch:", min_value=3, max_value=100, value=10)
    run = st.button("Run Analysis", type="primary", use_container_width=True)

if not run:
    st.info("Enter a research topic in the sidebar and click **Run Analysis**.")
    st.stop()


# ============================================================
# STAGE 1 — Download papers and show them immediately
# ============================================================

with st.spinner("Fetching papers from ArXiv..."):
    arxiv = ArxivClient()
    papers = arxiv.search_and_download(query, max_results=max_papers)

with st.spinner("Enriching with Semantic Scholar..."):
    scholar = SemanticScholarClient()
    papers, extra_papers = scholar.enrich_and_expand(papers)
    if extra_papers:
        papers.extend(extra_papers)

# --- Parse all papers upfront so we can show overviews ---
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

# ============================================================
# SHOW PAPERS — user can browse while pipeline continues below
# ============================================================

st.header(f"Papers ({len(paper_data)})")

for i, pd in enumerate(paper_data):
    citations = f" | {pd['citation_count']} citations" if pd.get("citation_count") else ""
    year = f" | {pd['year']}" if pd.get("year") else ""
    venue = f" | {pd['venue']}" if pd.get("venue") else ""

    with st.expander(f"**{i+1}. {pd['title']}**{year}{citations}{venue}"):
        section_names = list(pd["sections"].keys())
        st.caption(f"Sections found: {', '.join(section_names) if section_names else 'none detected'}")

        # Show abstract or first section as preview
        preview_section = None
        for candidate in ["abstract", "introduction"]:
            if candidate in pd["sections"]:
                preview_section = candidate
                break

        if preview_section:
            preview_text = pd["sections"][preview_section]
            # Clean up: take first ~500 chars
            preview = preview_text[:500].strip()
            if len(preview_text) > 500:
                preview += "..."
            st.markdown(f"**{preview_section.title()}:**")
            st.markdown(preview)
        else:
            # Fallback: show first 500 chars of raw text
            preview = pd["text"][:500].strip()
            if len(pd["text"]) > 500:
                preview += "..."
            st.markdown(preview)

st.divider()

# ============================================================
# STAGE 2 — Claim Extraction (the slow part)
# ============================================================

st.header("Pipeline Progress")

with st.status("Extracting claims from papers...", expanded=True) as status:
    extractor = ClaimExtractor()
    all_claims = []
    progress = st.progress(0, text="Starting claim extraction...")

    for idx, pd in enumerate(paper_data):
        progress.progress(
            idx / len(paper_data),
            text=f"Extracting from paper {idx+1}/{len(paper_data)}: {pd['title'][:50]}..."
        )

        for section_name, content in pd["sections"].items():
            sentences = splitter.split(content)
            for s in sentences:
                if len(s) < MIN_SENTENCE_LENGTH:
                    continue
                claims = extractor.extract_from_sentence(
                    s, paper_title=pd["title"], section=section_name,
                )
                all_claims.extend(claims)

    progress.progress(1.0, text=f"Extracted {len(all_claims)} claims")
    status.update(label=f"Extracted {len(all_claims)} claims from {len(paper_data)} papers", state="complete")

if not all_claims:
    st.error("No claims extracted. Try a different topic or increase the paper count.")
    st.stop()

# ============================================================
# STAGE 3 — Embedding + Clustering (fast)
# ============================================================

with st.status("Embedding and clustering claims...", expanded=False) as status:
    embedder = SpecterEmbedder()
    embeddings = embedder.encode(all_claims)

    clusterer = ClaimClusterer()
    labels = clusterer.cluster(embeddings)
    clusters = clusterer.group_clusters(all_claims, labels)
    if not clusters:
        clusters = {0: all_claims}

    status.update(label=f"Formed {len(clusters)} clusters from {len(all_claims)} claims", state="complete")

# ============================================================
# STAGE 4 — Contradiction Detection
# ============================================================

with st.status("Detecting contradictions via NLI...", expanded=False) as status:
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

    status.update(label=f"Found {len(all_contradictions)} contradictions", state="complete")

# ============================================================
# STAGE 5 — Limitation Retrieval
# ============================================================

with st.status("Retrieving limitation-adjacent claims...", expanded=False) as status:
    retriever = LimitationRetriever(embedder, clusterer)
    limitation_claims, limitation_clusters = retriever.retrieve(all_claims, embeddings)
    status.update(
        label=f"Found {len(limitation_claims)} limitation claims in {len(limitation_clusters)} clusters",
        state="complete",
    )

# ============================================================
# STAGE 6 — Gap Generation
# ============================================================

with st.status("Generating research gaps...", expanded=True) as status:
    gap_generator = GapGenerator()
    gaps = []

    cluster_tasks = [cl for cl in clusters.values() if len(cl) >= 2]
    limitation_tasks = [cl for cl in limitation_clusters.values() if len(cl) >= 2]
    total_tasks = len(cluster_tasks) + (1 if all_contradictions else 0) + len(limitation_tasks)
    done = 0

    progress = st.progress(0, text="Generating research gaps...")

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
    status.update(label=f"Generated {len(gaps)} research gaps", state="complete")


# ============================================================
# RESULTS
# ============================================================

st.divider()
st.header("Results")

# --- Summary ---
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Papers", len(papers))
c2.metric("Claims", len(all_claims))
c3.metric("Clusters", len(clusters))
c4.metric("Contradictions", len(all_contradictions))
c5.metric("Research Gaps", len(gaps))

st.divider()

# --- Categorize gaps ---
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
    type_labels = {
        "contradiction": "Contradiction",
        "limitation": "Limitation",
        "open_question": "Open Question",
    }
    type_colors = {
        "contradiction": "red",
        "limitation": "orange",
        "open_question": "blue",
    }
    label = type_labels.get(gap_type, "Gap")
    color = type_colors.get(gap_type, "blue")

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
