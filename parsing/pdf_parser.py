"""PDF text extraction (PyMuPDF) and sentence splitting (spaCy).

Heavy libraries are imported only when used, so the cleaning logic can be
tested without them.
"""
import re

from config.settings import SPACY_MODEL

# A line that is only a references heading, e.g. "References", "7 References", "BIBLIOGRAPHY".
_REFERENCES_HEADING = re.compile(r"^\s*(?:\d+\.?\s*)?(references|bibliography)\s*$", re.IGNORECASE | re.MULTILINE)


class PDFParseError(Exception):
    """Raised when a PDF can't be opened or read (corrupt or partial download)."""


class SentenceSplitter:
    def __init__(self):
        import spacy

        self.nlp = spacy.load(SPACY_MODEL)

    def split(self, text):
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 20]


class PDFParser:
    def extract_text(self, pdf_path):
        import fitz  # PyMuPDF

        try:
            with fitz.open(pdf_path) as doc:
                text = "".join(page.get_text() for page in doc)
        except Exception as e:  # corrupt or truncated download
            raise PDFParseError(f"Could not read {pdf_path}: {e}") from e
        return self._clean_text(text)

    def _clean_text(self, text):
        # Normalize spaces/tabs within lines, but keep paragraph breaks
        text = re.sub(r"[^\S\n]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return strip_references(text).strip()


def strip_references(text: str) -> str:
    """Cut the bibliography so citations don't become "claims".

    Only a line that is *just* a references heading counts, and the last one
    wins (papers sometimes have a references heading in an appendix too, but
    never a real bibliography before the body ends). The old version split on
    the first "references" anywhere, so a sentence like "...with references to
    prior work" in the introduction threw away most of the paper.
    """
    matches = list(_REFERENCES_HEADING.finditer(text))
    if not matches:
        return text
    return text[: matches[-1].start()]
