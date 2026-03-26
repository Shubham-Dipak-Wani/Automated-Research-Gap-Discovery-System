import streamlit as st

from ingestion.arxiv_client import ArxivClient
from parsing.pdf_parser import PDFParser, SentenceSplitter
from parsing.section_segmenter import SectionSegmenter
from claims.claim_extractor import ClaimExtractor
from embedding.specter_embedder import SpecterEmbedder
from clustering.hdbscan_cluster import ClaimClusterer
from gap_generation.generator import GapGenerator


st.set_page_config(page_title="Research Gap Finder", layout="wide")

st.title("🧠 Automated Research Gap Discovery System")

query = st.text_input("Enter research topic:", "retrieval augmented generation")

run = st.button("Run Analysis")

if run:
    with st.spinner("Processing papers..."):

        # Initialize components
        client = ArxivClient()
        parser = PDFParser()
        segmenter = SectionSegmenter()
        splitter = SentenceSplitter()
        extractor = ClaimExtractor()
        embedder = SpecterEmbedder()
        clusterer = ClaimClusterer()
        gap_generator = GapGenerator()

        papers = client.search_and_download(query, max_results=3)

        all_claims = []

        st.write("## 📄 Extracted Claims")

        for paper in papers:
            text = parser.extract_text(paper["pdf_path"])
            sections = segmenter.segment(text)

            for content in sections.values():
                sentences = splitter.split(content)

                for s in sentences[:10]:
                    if len(s) < 40:
                        continue

                    claims = extractor.extract_from_sentence(s)

                    for c in claims:
                        st.write("•", c)
                        all_claims.append(c)

        if not all_claims:
            st.warning("No claims extracted.")
            st.stop()

        # Embeddings
        embeddings = embedder.encode(all_claims)

        # Clustering
        labels = clusterer.cluster(embeddings)
        clusters = clusterer.group_clusters(all_claims, labels)

        if not clusters:
            clusters = {0: all_claims}

        st.write("## 🧩 Clusters")

        for cid, cl in clusters.items():
            with st.expander(f"Cluster {cid}"):
                for c in cl:
                    st.write("•", c)

        # Gap Generation
        st.write("## 🚀 Research Gaps")

        for cid, cl in clusters.items():
            st.subheader(f"Cluster {cid}")
            gap = gap_generator.generate_gap(cl)
            st.success(gap)