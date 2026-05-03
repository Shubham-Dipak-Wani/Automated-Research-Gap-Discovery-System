"""Command-line run of the full pipeline.

    python main.py "retrieval augmented generation" --papers 5
"""
import argparse
import json
import os

import pipeline
from config.settings import ARXIV_MAX_RESULTS


def run_pipeline(query, max_results=None):
    # Heavy models are imported here, not at the top, so `--help` stays instant.
    from claims.claim_extractor import ClaimExtractor
    from claims.limitation_retriever import LimitationRetriever
    from clustering.hdbscan_cluster import ClaimClusterer
    from embedding.specter_embedder import SpecterEmbedder
    from gap_generation.generator import GapGenerator
    from ingestion.arxiv_client import ArxivClient
    from ingestion.semantic_scholar_client import SemanticScholarClient
    from nli.nli_engine import NLIEngine
    from parsing.pdf_parser import PDFParser, SentenceSplitter
    from parsing.section_segmenter import SectionSegmenter

    max_results = max_results or ARXIV_MAX_RESULTS
    print(f"\n=== PIPELINE: '{query}' ({max_results} papers) ===\n")

    print("Downloading papers from arXiv...")
    papers = ArxivClient().search_and_download(query, max_results=max_results)
    print(f"Downloaded {len(papers)} papers")

    print("Enriching with Semantic Scholar...")
    papers, extra = SemanticScholarClient().enrich_and_expand(papers)
    if extra:
        print(f"Discovered {len(extra)} more papers through the citation graph")
        papers.extend(extra)

    parsed = pipeline.parse_papers(papers, PDFParser(), SectionSegmenter())

    print("\nExtracting claims (Claude Haiku)...")
    claims = pipeline.extract_claims(
        parsed, SentenceSplitter(), ClaimExtractor(),
        on_paper=lambda i, p, n: print(f"  {i + 1}/{len(parsed)} {p['title'][:60]}: {n} claims"),
    )
    print(f"Total claims: {len(claims)}")
    if not claims:
        print("No claims extracted. Exiting.")
        return None

    print("\nEmbedding (SPECTER2), clustering (HDBSCAN), NLI and limitations...")
    embedder, clusterer = SpecterEmbedder(), ClaimClusterer()
    embeddings = embedder.encode(claims)
    analysis = pipeline.analyze(claims, embeddings, clusterer, NLIEngine(), LimitationRetriever(embedder, clusterer))
    print(f"{len(analysis['clusters'])} clusters, {len(analysis['contradictions'])} contradictions, "
          f"{len(analysis['limitation_claims'])} limitation claims")

    print("\n=== RESEARCH GAPS (Claude Opus) ===\n")
    tasks = pipeline.plan_gap_tasks(claims, analysis)

    def show(done, total, gap):
        print(f"[{done}/{total}] {gap['type']}: {gap['gap']}")
        print(f"   Sources: {', '.join(gap['source_papers'])}\n")

    gaps = pipeline.generate_gaps(tasks, GapGenerator(), on_gap=show)
    print(f"=== {len(gaps)} gaps from {len(parsed)} papers ===")
    return pipeline.summarize(query, parsed, claims, analysis, gaps)


def main():
    ap = argparse.ArgumentParser(description="Find research gaps in the literature on a topic.")
    ap.add_argument("query", nargs="?", default="retrieval augmented generation")
    ap.add_argument("--papers", type=int, default=5, help="papers to fetch from arXiv (default 5)")
    ap.add_argument("--out", default="data/results.json")
    args = ap.parse_args()

    results = run_pipeline(args.query, max_results=args.papers)
    if results:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
