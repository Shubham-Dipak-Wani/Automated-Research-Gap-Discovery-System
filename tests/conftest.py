"""Shared fakes. Nothing here loads a model or touches the network."""
import pytest


def claim(text, paper="Paper A", section="results"):
    return {"text": text, "paper_title": paper, "section": section, "source_sentence": text}


class FakeNLI:
    """Says 'contradiction' when one claim contains 'not' and the other doesn't."""

    def __init__(self, confidence=0.9):
        self.confidence = confidence
        self.calls = []

    def predict(self, premise, hypothesis):
        self.calls.append((premise, hypothesis))
        if (" not " in f" {premise} ") != (" not " in f" {hypothesis} "):
            return "contradiction", self.confidence
        return "entailment", 0.8


@pytest.fixture
def make_claim():
    return claim
