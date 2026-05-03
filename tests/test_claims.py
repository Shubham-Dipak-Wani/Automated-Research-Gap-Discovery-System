import numpy as np

from claims.claim_extractor import ClaimExtractor
from claims.limitation_retriever import LimitationRetriever

BATCH = [
    ("BERT beats GPT-2 on GLUE and uses fewer parameters.", "Paper A", "results"),
    ("We thank the reviewers.", "Paper A", "acknowledgements"),
    ("Doing retrieval first reduces hallucination in long answers.", "Paper B", "results"),
]


def extractor():
    return ClaimExtractor(client=object())  # no API client needed to parse


def test_batch_output_is_mapped_back_to_its_sentence():
    output = (
        "SENTENCE 1:\n- BERT outperforms GPT-2 on the GLUE benchmark.\n- BERT uses fewer parameters than GPT-2.\n"
        "SENTENCE 2: NONE\n"
        "SENTENCE 3:\n- Doing retrieval first reduces hallucination in long-form answers"
    )
    claims = extractor()._parse_batch_output(output, BATCH)
    assert [c["text"] for c in claims] == [
        "BERT outperforms GPT-2 on the GLUE benchmark.",
        "BERT uses fewer parameters than GPT-2.",
        "Doing retrieval first reduces hallucination in long-form answers.",  # period added
    ]
    assert claims[2]["paper_title"] == "Paper B" and claims[2]["source_sentence"] == BATCH[2][0]


def test_words_that_merely_contain_metadata_terms_are_kept():
    """'doing' contains 'doi' and 'nonetheless' contains 'none': both used to be dropped."""
    block = "- Doing more pretraining improves robustness on noisy inputs.\n- Nonetheless, recall drops on rare entities in the test set."
    assert len(extractor()._parse_single_block(block)) == 2


def test_real_metadata_and_short_lines_are_dropped():
    block = "\n".join([
        "- Code is available at https://github.com/x/y for all experiments.",
        "- Copyright 2024 by the Association for Computational Linguistics.",
        "- See arXiv:2005.11401 for the extended version of this paper.",
        "- Contact the first author by email at someone@uni.edu please.",
        "- Too short.",
        "- NONE",
        "- None of the baselines handle documents longer than 4,000 tokens.",
    ])
    assert extractor()._parse_single_block(block) == ["None of the baselines handle documents longer than 4,000 tokens."]


def test_out_of_range_sentence_numbers_are_ignored():
    assert extractor()._parse_batch_output("SENTENCE 9:\n- A claim that points nowhere at all here.", BATCH) == []


def test_api_errors_skip_the_batch_instead_of_crashing():
    class Failing:
        class messages:
            @staticmethod
            def create(**_):
                raise RuntimeError("rate limited")

    assert ClaimExtractor(client=Failing())._process_batch(BATCH) == []


def test_batches_are_split_by_batch_size(monkeypatch):
    seen = []
    ex = extractor()
    monkeypatch.setattr(ex, "_process_batch", lambda batch: seen.append(len(batch)) or [])
    ex.extract_all([("s", "p", "x")] * 23)
    assert seen == [10, 10, 3]


class FakeEmbedder:
    """Seed queries point along axis 0; claims are given explicit vectors."""

    def encode(self, items):
        return np.array([[1.0, 0.0, 0.0]] * len(items))


class FakeClusterer:
    def cluster(self, embeddings):
        return np.zeros(len(embeddings), dtype=int)

    def group_clusters(self, claims, labels):
        return {0: list(claims)}


def test_limitation_retriever_selects_claims_near_seed_queries(make_claim):
    claims = [make_claim("Long documents remain challenging."), make_claim("BERT has 110M parameters."),
              make_claim("Future work should address multilingual retrieval.")]
    embeddings = np.array([[0.9, 0.1, 0.0], [0.0, 0.0, 1.0], [0.8, 0.2, 0.1]])
    found, clusters = LimitationRetriever(FakeEmbedder(), FakeClusterer()).retrieve(claims, embeddings)
    assert [c["text"] for c in found] == [claims[0]["text"], claims[2]["text"]]
    assert len(clusters[0]) == 2


def test_limitation_retriever_handles_no_matches(make_claim):
    found, clusters = LimitationRetriever(FakeEmbedder(), FakeClusterer()).retrieve(
        [make_claim("Unrelated.")], np.array([[0.0, 1.0, 0.0]]))
    assert found == [] and clusters == {}
