import fitz  # PyMuPDF
import re
import spacy
from config.settings import SPACY_MODEL


class SentenceSplitter:
    def __init__(self):
        self.nlp = spacy.load(SPACY_MODEL)

    def split(self, text):
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 20]


class PDFParser:
    def extract_text(self, pdf_path):
        doc = fitz.open(pdf_path)
        text = ""

        for page in doc:
            text += page.get_text()

        return self._clean_text(text)

    def _clean_text(self, text):
        # Normalize spaces/tabs within lines, but preserve paragraph breaks
        text = re.sub(r'[^\S\n]+', ' ', text)
        # Collapse 3+ newlines into double newline (paragraph break)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Remove references section (basic heuristic)
        text = re.split(r'\bReferences\b', text, flags=re.IGNORECASE)[0]

        return text.strip()
