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

from sklearn.metrics.pairwise import cosine_similarity
import json


def run_pipeline(query, max_results=None):
    max_results = max_results or ARXIV_MAX_RESULTS
    print(f"\n=== PIPELINE: '{query}' ({max_results} papers) ===\n")

    # --- Init components ---
    arxiv = ArxivClient()
    scholar = SemanticScholarClient()
    parser = PDFParser()
    segmenter = SectionSegmenter()
    splitter = SentenceSplitter()
    extractor = ClaimExtractor()
    embedder = SpecterEmbedder()
    clusterer = ClaimClusterer()
    nli = NLIEngine()
    gap_generator = GapGenerator()

    # --- 1. Paper Collection ---
    print("Downloading papers from ArXiv...")
    papers = arxiv.search_and_download(query, max_results=max_results)
    print(f"Downloaded {len(papers)} papers")

    print("Enriching with Semantic Scholar...")
    papers, extra_papers = scholar.enrich_and_expand(papers)
    if extra_papers:
        print(f"Discovered {len(extra_papers)} additional papers via citation graph")
        papers.extend(extra_papers)

    # --- 2. Claim Extraction ---
    print("\nExtracting claims...")
    all_claims = []

    for idx, paper in enumerate(papers):
        print(f"  Paper {idx + 1}/{len(papers)}: {paper['title'][:60]}...")
        text = parser.extract_text(paper["pdf_path"])
        sections = segmenter.segment(text)

        for section_name, content in sections.items():
            sentences = splitter.split(content)

            for s in sentences:
                if len(s) < MIN_SENTENCE_LENGTH:
                    continue

                claims = extractor.extract_from_sentence(
                    s,
                    paper_title=paper["title"],
                    section=section_name,
                )
                all_claims.extend(claims)

    print(f"\nTotal claims extracted: {len(all_claims)}")

    if not all_claims:
        print("No claims extracted. Exiting.")
        return None

    # --- 3. Embedding ---
    print("\nEmbedding claims...")
    embeddings = embedder.encode(all_claims)

    # --- 4. Clustering ---
    print("Clustering...")
    labels = clusterer.cluster(embeddings)
    clusters = clusterer.group_clusters(all_claims, labels)

    if not clusters:
        print("No clusters formed — using all claims as one group")
        clusters = {0: all_claims}

    print(f"Formed {len(clusters)} clusters")

    # --- 5. NLI Contradiction Detection ---
    print("\nDetecting contradictions...")
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
                    print(f"  Contradiction (conf={conf:.2f}):")
                    print(f"    {cl[i]['text'][:80]}...")
                    print(f"    vs. {cl[j]['text'][:80]}...")

    print(f"Found {len(all_contradictions)} contradictions")

    # --- 6. Limitation Retrieval ---
    print("\nRetrieving limitation-adjacent claims...")
    retriever = LimitationRetriever(embedder, clusterer)
    limitation_claims, limitation_clusters = retriever.retrieve(all_claims, embeddings)
    print(f"Found {len(limitation_claims)} limitation claims in {len(limitation_clusters)} clusters")

    # --- 7. Gap Generation ---
    print("\n=== RESEARCH GAPS ===\n")
    results = {
        "query": query,
        "papers_analyzed": len(papers),
        "total_claims": len(all_claims),
        "clusters": len(clusters),
        "contradictions": len(all_contradictions),
        "gaps": [],
    }

    # Gaps from topic clusters
    for cid, cl in clusters.items():
        if len(cl) < 2:
            continue
        gap = gap_generator.generate_gap(cl)
        results["gaps"].append(gap)
        print(f"Cluster {cid}: {gap['gap']}")
        print(f"  Sources: {', '.join(gap['source_papers'])}\n")

    # Gaps from contradictions
    if all_contradictions:
        gap = gap_generator.generate_gap([], contradictions=all_contradictions)
        results["gaps"].append(gap)
        print(f"Contradiction Gap: {gap['gap']}")
        print(f"  Sources: {', '.join(gap['source_papers'])}\n")

    # Gaps from limitations
    for cid, cl in limitation_clusters.items():
        if len(cl) < 2:
            continue
        gap = gap_generator.generate_gap(cl)
        gap["type"] = "limitation"
        results["gaps"].append(gap)
        print(f"Limitation Gap: {gap['gap']}")
        print(f"  Sources: {', '.join(gap['source_papers'])}\n")

    print(f"\n=== SUMMARY: {len(results['gaps'])} gaps from {len(papers)} papers ===")
    return results


def main():
    results = run_pipeline("retrieval augmented generation")
    if results:
        with open("data/results.json", "w") as f:
            json.dump(results, f, indent=2)
        print("\nResults saved to data/results.json")


if __name__ == "__main__":
    main()
