import pytest

from evaluation.evaluator import build_examples
from evaluation.metrics import compute_claim_f1, compute_claim_recall, compute_nli_f1, pairwise_similarity
from ingestion import arxiv_client
from ingestion.arxiv_client import ArxivClient

ATOM = """<feed xmlns="http://www.w3.org/2005/Atom">
  <entry><title>Retrieval-Augmented
      Generation for NLP</title>
    <link title="pdf" href="https://arxiv.org/pdf/2005.11401v4"/></entry>
  <entry><title>No PDF here</title><link href="https://arxiv.org/abs/1"/></entry>
  <entry><title>Broken download</title><link title="pdf" href="https://arxiv.org/pdf/bad"/></entry>
</feed>"""


def test_arxiv_feed_is_parsed_titles_cleaned_and_failures_skipped(tmp_path, monkeypatch):
    def fake_download(self, url, save_dir):
        if url.endswith("bad"):
            raise IOError("timeout")
        return str(tmp_path / (url.split("/")[-1] + ".pdf"))

    monkeypatch.setattr(ArxivClient, "_download_pdf", fake_download)
    papers = ArxivClient()._parse_and_download(ATOM, str(tmp_path))
    assert papers == [{"title": "Retrieval-Augmented Generation for NLP", "pdf_path": str(tmp_path / "2005.11401v4.pdf")}]


def test_arxiv_retries_on_429_then_succeeds(monkeypatch, tmp_path):
    responses = iter([type("R", (), {"status_code": 429, "text": ""})(),
                      type("R", (), {"status_code": 200, "text": "<feed xmlns='http://www.w3.org/2005/Atom'/>"})()])
    monkeypatch.setattr(arxiv_client.requests, "get", lambda *a, **k: next(responses))
    monkeypatch.setattr(arxiv_client.time, "sleep", lambda s: None)
    assert ArxivClient().search_and_download("rag", max_results=1, save_dir=str(tmp_path)) == []


def fake_scorer(cands, refs):
    """1.0 when the strings share their first word, else 0.0."""
    return [1.0 if c.split()[0] == r.split()[0] else 0.0 for c, r in zip(cands, refs)]


def test_pairwise_similarity_builds_a_full_matrix():
    sims = pairwise_similarity(["a x", "b y"], ["a z", "c w", "b q"], fake_scorer)
    assert sims == [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]


def test_claim_f1_counts_matches_each_way():
    result = compute_claim_f1(["a 1", "b 2", "z 3"], ["a 9", "b 9"], scorer=fake_scorer)
    assert result["precision"] == pytest.approx(2 / 3) and result["recall"] == 1.0
    assert compute_claim_f1([], ["a"], scorer=fake_scorer)["f1"] == 0.0


def test_claim_recall():
    result = compute_claim_recall([("a gold", ["a found", "x"]), ("b gold", ["c other"]), ("d gold", [])], scorer=fake_scorer)
    assert result == {"recall": pytest.approx(1 / 3), "matched": 1, "total": 3}


def test_nli_f1():
    result = compute_nli_f1(["contradiction", "entailment", "neutral", "contradiction"],
                            ["contradiction", "entailment", "contradiction", "neutral"])
    assert result["precision_contradiction"] == 0.5 and result["recall_contradiction"] == 0.5
    assert result["n"] == 4
    with pytest.raises(ValueError):
        compute_nli_f1([], [])


def test_scifact_rows_are_joined_with_abstracts():
    corpus = [{"doc_id": 10, "abstract": ["S0.", "S1.", "S2.", "S3."]}, {"doc_id": 11, "abstract": ["T0.", "T1."]}]
    claims = [
        {"claim": "A supported claim.", "evidence_label": "SUPPORT", "evidence_doc_id": "10", "evidence_sentences": [1, 3], "cited_doc_ids": [10]},
        {"claim": "A refuted claim.", "evidence_label": "CONTRADICT", "evidence_doc_id": "11", "evidence_sentences": [0], "cited_doc_ids": [11]},
        {"claim": "No evidence claim.", "evidence_label": "", "evidence_doc_id": "", "evidence_sentences": [], "cited_doc_ids": [11]},
        {"claim": "Missing doc.", "evidence_label": "SUPPORT", "evidence_doc_id": "99", "evidence_sentences": [0], "cited_doc_ids": [99]},
    ]
    examples = build_examples(claims, corpus)
    assert examples == [
        {"claim": "A supported claim.", "premise": "S1. S3.", "label": "entailment"},
        {"claim": "A refuted claim.", "premise": "T0.", "label": "contradiction"},
        {"claim": "No evidence claim.", "premise": "T0. T1.", "label": "neutral"},
    ]
