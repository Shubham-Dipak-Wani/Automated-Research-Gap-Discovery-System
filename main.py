from ingestion.arxiv_client import ArxivClient
from parsing.pdf_parser import PDFParser, SentenceSplitter
from parsing.section_segmenter import SectionSegmenter
from claims.claim_extractor import ClaimExtractor
from embedding.specter_embedder import SpecterEmbedder
from clustering.hdbscan_cluster import ClaimClusterer
from nli.nli_engine import NLIEngine
from gap_generation.generator import GapGenerator

from sklearn.metrics.pairwise import cosine_similarity


def main():
    print("\n=== STARTING PIPELINE ===\n")

    client = ArxivClient()
    parser = PDFParser()
    segmenter = SectionSegmenter()
    splitter = SentenceSplitter()
    extractor = ClaimExtractor()
    embedder = SpecterEmbedder()
    clusterer = ClaimClusterer()
    nli = NLIEngine()
    gap_generator = GapGenerator()

    papers = client.search_and_download("retrieval augmented generation", 3)

    all_claims = []

    # ---- CLAIM EXTRACTION ----
    for idx, paper in enumerate(papers):
        print(f"\n--- Paper {idx+1} ---")

        text = parser.extract_text(paper["pdf_path"])
        sections = segmenter.segment(text)

        for content in sections.values():
            for s in splitter.split(content)[:10]:

                if len(s) < 40:
                    continue

                claims = extractor.extract_from_sentence(s)

                for c in claims:
                    print("-", c)
                    all_claims.append(c)

    # ---- EMBEDDINGS ----
    embeddings = embedder.encode(all_claims)

    # ---- CLUSTERING ----
    labels = clusterer.cluster(embeddings)
    clusters = clusterer.group_clusters(all_claims, labels)

    # 🔥 CLEAN CLUSTERS
    clusters = {
        cid: cl for cid, cl in clusters.items()
        if len(cl) >= 2 and all("acm" not in c.lower() for c in cl)
    }

    # 🔥 FALLBACK
    if not clusters:
        print("\nNo valid clusters → fallback")
        clusters = {0: all_claims}

    print("\n=== CLUSTERS ===")
    for cid, cl in clusters.items():
        print(f"\nCluster {cid}:")
        for c in cl:
            print("-", c)

    # ---- NLI ----
    print("\n=== CONTRADICTIONS ===")
    for cid, cl in clusters.items():
        idxs = [i for i, l in enumerate(labels) if l == cid]
        emb = embeddings[idxs] if idxs else embeddings

        for i in range(len(cl)):
            for j in range(i + 1, len(cl)):

                sim = cosine_similarity([emb[i]], [emb[j]])[0][0]

                if sim < 0.7 or sim > 0.9:
                    continue

                rel, conf = nli.predict(cl[i], cl[j])

                if conf > 0.7 and rel == "contradiction":
                    print("\n❌", cl[i])
                    print("VS", cl[j])

    # ---- GAP GENERATION ----
    print("\n=== RESEARCH GAPS ===")
    for cid, cl in clusters.items():
        print(f"\nCluster {cid}:")
        print(gap_generator.generate_gap(cl))


if __name__ == "__main__":
    main()