import fitz  # PyMuPDF
import re
import spacy

class SentenceSplitter:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

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
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove references section (basic heuristic)
        text = re.split(r'References', text, flags=re.IGNORECASE)[0]

        return text.strip()