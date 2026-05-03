"""The research gap pipeline, shared by the CLI (main.py) and the web app (app.py).

Each stage is a plain function that takes its collaborators as arguments, so
the CLI, the Streamlit app and the tests all run exactly the same logic.

    papers → parse → claims → embeddings → clusters ─┬─ contradictions (NLI)
                                                    ├─ limitations (seed similarity)
                                                    └─ gaps (Claude Opus, ranked + capped)
"""
from clustering.utils import group_indices, rank_clusters
from config.settings import MAX_LIMITATION_GAPS, MAX_TOPIC_GAPS, MIN_SENTENCE_LENGTH
from nli.pair_selector import find_contradictions
from parsing.pdf_parser import PDFParseError


def parse_papers(papers, parser, segmenter, on_error=print) -> list:
    """Read each PDF into text + sections. Unreadable PDFs are skipped, not fatal."""
    parsed = []
    for paper in papers:
        try:
            text = parser.extract_text(paper["pdf_path"])
        except PDFParseError as e:
            on_error(f"Skipping '{paper['title'][:60]}': {e}")
            continue
        parsed.append({**paper, "text": text, "sections": segmenter.segment(text)})
    return parsed


def sentence_items(paper, splitter, min_length=MIN_SENTENCE_LENGTH) -> list:
    """(sentence, paper title, section) for every sentence long enough to hold a claim."""
    items = []
    for section_name, content in paper["sections"].items():
        for sentence in splitter.split(content):
            if len(sentence) >= min_length:
                items.append((sentence, paper["title"], section_name))
    return items


def extract_claims(parsed_papers, splitter, extractor, on_paper=None) -> list:
    """Batch-extract atomic claims from every paper. on_paper(index, paper, n_claims) reports progress."""
    claims = []
    for i, paper in enumerate(parsed_papers):
        paper_claims = extractor.extract_all(sentence_items(paper, splitter))
        claims.extend(paper_claims)
        if on_paper:
            on_paper(i, paper, len(paper_claims))
    return claims


def analyze(claims, embeddings, clusterer, nli, retriever) -> dict:
    """Cluster claims, find contradictions inside clusters, and pull out limitation themes."""
    labels = clusterer.cluster(embeddings)
    groups = group_indices(labels)
    if not groups:  # everything was noise: treat all claims as one topic
        groups = {0: list(range(len(claims)))}

    contradictions = find_contradictions(groups, claims, embeddings, nli)
    limitation_claims, limitation_clusters = retriever.retrieve(claims, embeddings)
    return {
        "groups": groups,
        "clusters": {cid: [claims[i] for i in idxs] for cid, idxs in groups.items()},
        "contradictions": contradictions,
        "limitation_claims": limitation_claims,
        "limitation_clusters": limitation_clusters,
    }


def _rank_claim_lists(clusters: dict, limit: int) -> list:
    """Rank clusters given as {id: [claims]}: most distinct papers first, then size."""
    sized = [cl for cl in clusters.values() if len(cl) >= 2]
    sized.sort(key=lambda cl: (-len({c["paper_title"] for c in cl}), -len(cl)))
    return sized[:limit]


def plan_gap_tasks(claims, analysis, max_topic=MAX_TOPIC_GAPS, max_limitation=MAX_LIMITATION_GAPS) -> list:
    """Decide which gaps to generate: [(kind, payload)] in the order they'll run."""
    tasks = [("open_question", analysis["clusters"][cid])
             for cid in rank_clusters(analysis["groups"], claims, limit=max_topic)]
    if analysis["contradictions"]:
        tasks.append(("contradiction", analysis["contradictions"]))
    tasks += [("limitation", cl) for cl in _rank_claim_lists(analysis["limitation_clusters"], max_limitation)]
    return tasks


def generate_gaps(tasks, generator, on_gap=None) -> list:
    """Run Claude Opus once per planned task. on_gap(done, total, gap) reports progress."""
    gaps = []
    for done, (kind, payload) in enumerate(tasks, start=1):
        if kind == "contradiction":
            gap = generator.generate_gap([], contradictions=payload)
        else:
            gap = generator.generate_gap(payload)
            gap["type"] = kind
        gaps.append(gap)
        if on_gap:
            on_gap(done, len(tasks), gap)
    return gaps


def summarize(query, papers, claims, analysis, gaps) -> dict:
    return {
        "query": query,
        "papers_analyzed": len(papers),
        "total_claims": len(claims),
        "clusters": len(analysis["clusters"]),
        "contradictions": len(analysis["contradictions"]),
        "gaps": gaps,
    }
