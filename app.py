import streamlit as st
from main import run_pipeline
from config.settings import ARXIV_MAX_RESULTS

st.set_page_config(page_title="Research Gap Finder", layout="wide")

st.title("Research Gap Discovery System")
st.markdown("Analyze research papers to find contradictions, limitations, and open questions.")

col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input("Enter research topic:", "retrieval augmented generation")
with col2:
    max_papers = st.number_input("Max papers:", min_value=3, max_value=100, value=ARXIV_MAX_RESULTS)

run = st.button("Run Analysis")

if run:
    with st.spinner("Running pipeline... (this may take several minutes)"):
        results = run_pipeline(query, max_results=max_papers)

    if not results:
        st.error("Pipeline failed — no claims extracted.")
        st.stop()

    # --- Summary ---
    st.write("## Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Papers", results["papers_analyzed"])
    c2.metric("Claims", results["total_claims"])
    c3.metric("Clusters", results["clusters"])
    c4.metric("Contradictions", results["contradictions"])

    # --- Research Gaps ---
    st.write("## Research Gaps")

    for i, gap in enumerate(results["gaps"]):
        gap_type = gap.get("type", "open_question")
        icon = {"contradiction": "!!", "limitation": ">>", "open_question": "??"}
        label = f"[{icon.get(gap_type, '??')}] Gap {i + 1}: {gap_type.replace('_', ' ').title()}"

        with st.expander(label, expanded=(i < 3)):
            st.success(gap["gap"])

            if gap.get("source_papers"):
                st.write("**Source Papers:**")
                for p in gap["source_papers"]:
                    st.write(f"- {p}")

            if gap.get("supporting_claims"):
                st.write("**Supporting Claims:**")
                for c in gap["supporting_claims"]:
                    st.write(f"- {c}")
