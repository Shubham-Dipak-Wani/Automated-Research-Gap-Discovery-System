import numpy as np

import pipeline
from clustering.utils import group_indices, rank_clusters
from gap_generation.generator import GapGenerator
from nli.pair_selector import candidate_pairs, find_contradictions
from parsing.pdf_parser import PDFParseError
from tests.conftest import FakeNLI, claim


def test_group_indices_drops_noise_and_keeps_claim_order():
    assert group_indices(np.array([0, -1, 2, 0, 2, -1])) == {0: [0, 3], 2: [2, 4]}


def test_cluster_ids_with_gaps_stay_aligned():
    """HDBSCAN can use ids 0, 1, 3. Grouping by index keeps embeddings and claims in sync."""
    groups = group_indices([3, 0, 3, 1])
    assert groups[3] == [0, 2] and groups[0] == [1] and groups[1] == [3]


def test_rank_prefers_cross_paper_clusters_then_size():
    claims = [claim("a", "P1"), claim("b", "P1"), claim("c", "P1"),
              claim("d", "P1"), claim("e", "P2"), claim("f", "P3")]
    groups = {7: [0, 1, 2], 8: [3, 4], 9: [5]}
    assert rank_clusters(groups, claims) == [8, 7]       # 9 has one claim: skipped
    assert rank_clusters(groups, claims, limit=1) == [8]


def test_candidate_pairs_keep_only_the_similarity_window():
    emb = np.array([[1.0, 0.0], [0.8, 0.6], [1.0, 0.0001], [0.0, 1.0]])
    pairs = {(i, j) for i, j, _ in candidate_pairs(emb, lo=0.7, hi=0.9)}
    # (0,1)=0.8 in window; (0,2)≈1.0 near-duplicate; (0,3)=0 unrelated
    assert (0, 1) in pairs and (0, 2) not in pairs and (0, 3) not in pairs


def test_find_contradictions_checks_both_directions_within_clusters():
    claims = [claim("RAG does not reduce hallucination.", "P1"), claim("RAG does reduce hallucination.", "P2"),
              claim("Unrelated cluster claim.", "P3")]
    emb = np.array([[1.0, 0.0], [0.8, 0.6], [0.8, 0.6]])
    nli = FakeNLI(confidence=0.95)
    found = find_contradictions({0: [0, 1], 1: [2]}, claims, emb, nli)
    assert len(found) == 1 and found[0][2] == 0.95
    assert len(nli.calls) == 2  # one pair, both directions


def test_low_confidence_contradictions_are_ignored():
    claims = [claim("X does not help."), claim("X does help.")]
    found = find_contradictions({0: [0, 1]}, claims, np.array([[1.0, 0.0], [0.8, 0.6]]), FakeNLI(confidence=0.6))
    assert found == []


class FakeParser:
    def extract_text(self, path):
        if path == "broken.pdf":
            raise PDFParseError("corrupt")
        return "Abstract\nText."


class FakeSegmenter:
    def segment(self, text):
        return {"abstract": text}


def test_unreadable_pdfs_are_skipped_not_fatal():
    errors = []
    papers = [{"title": "Good", "pdf_path": "ok.pdf"}, {"title": "Bad", "pdf_path": "broken.pdf"}]
    parsed = pipeline.parse_papers(papers, FakeParser(), FakeSegmenter(), on_error=errors.append)
    assert [p["title"] for p in parsed] == ["Good"] and len(errors) == 1


def test_sentence_items_filter_short_sentences():
    class Splitter:
        def split(self, text):
            return ["Short one.", "This sentence is long enough to contain a scientific claim."]

    items = pipeline.sentence_items({"title": "T", "sections": {"results": "..."}}, Splitter(), min_length=40)
    assert items == [("This sentence is long enough to contain a scientific claim.", "T", "results")]


def test_gap_plan_is_ranked_capped_and_typed():
    claims = [claim(f"claim {i}", f"P{i % 3}") for i in range(12)]
    groups = {c: [c * 2, c * 2 + 1] for c in range(6)}
    analysis = {
        "groups": groups,
        "clusters": {cid: [claims[i] for i in idxs] for cid, idxs in groups.items()},
        "contradictions": [(claims[0], claims[1], 0.9)],
        "limitation_clusters": {0: claims[:3], 1: claims[3:4]},
    }
    tasks = pipeline.plan_gap_tasks(claims, analysis, max_topic=4, max_limitation=5)
    kinds = [k for k, _ in tasks]
    assert kinds == ["open_question"] * 4 + ["contradiction", "limitation"]


def test_generate_gaps_reports_progress_and_sets_types():
    class Gen:
        def generate_gap(self, claims, contradictions=None):
            return {"gap": "Research Gap: x", "supporting_claims": [], "source_papers": [],
                    "type": "contradiction" if contradictions else "open_question"}

    progress = []
    gaps = pipeline.generate_gaps([("open_question", [1]), ("limitation", [2]), ("contradiction", [3])],
                                  Gen(), on_gap=lambda d, t, g: progress.append((d, t)))
    assert [g["type"] for g in gaps] == ["open_question", "limitation", "contradiction"]
    assert progress == [(1, 3), (2, 3), (3, 3)]


class _Block:
    def __init__(self, type_, text=""):
        self.type, self.text = type_, text


class FakeClaude:
    def __init__(self, text):
        self.sent = []
        outer = self

        class Messages:
            def create(self, **kwargs):
                outer.sent.append(kwargs)
                return type("R", (), {"content": [_Block("thinking"), _Block("text", text)]})()

        self.messages = Messages()


def test_gap_generator_adds_prefix_skips_thinking_and_caps_claims():
    client = FakeClaude("No study compares X and Y on long documents.")
    claims = [claim("A sufficiently long scientific claim about retrieval number %d here." % i, f"P{i % 2}")
              for i in range(8)] + [claim("Too short.")]
    gap = GapGenerator(client=client).generate_gap(claims)
    assert gap["gap"] == "Research Gap: No study compares X and Y on long documents."
    assert len(gap["supporting_claims"]) == 5 and gap["source_papers"] == ["P0", "P1"]
    assert client.sent[0]["thinking"] == {"type": "adaptive"}


def test_gap_generator_contradiction_prompt_includes_confidence():
    client = FakeClaude("Research Gap: The papers disagree on Z.")
    gap = GapGenerator(client=client).generate_gap([], contradictions=[(claim("Z helps.", "P1"), claim("Z hurts.", "P2"), 0.91)])
    assert gap["type"] == "contradiction" and gap["gap"].startswith("Research Gap: The papers")
    assert "[confidence: 0.91]" in client.sent[0]["messages"][0]["content"]
