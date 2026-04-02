# Pipeline Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align the codebase with the pre-proposal: fix bugs, upgrade models to Mistral 7B via Ollama + SPECTER2, implement missing features (limitation retrieval, Semantic Scholar, citation tracing, SciFact evaluation), centralize config.

**Architecture:** Stateless pipeline with centralized config. Ollama provides LLM inference (claim extraction + gap generation). All claims carry source provenance metadata throughout. Two analysis paths after clustering: NLI contradictions and limitation retrieval.

**Tech Stack:** Python 3.14, Ollama (Mistral 7B), SPECTER2 (sentence-transformers), DeBERTa-v3 (transformers), HDBSCAN, Streamlit, PyMuPDF, spaCy, SciFact (datasets), BERTScore.

---

## File Structure

| File | Responsibility | Action |
|------|---------------|--------|
| `config/settings.py` | All thresholds, model names, URLs | Implement |
| `ingestion/arxiv_client.py` | ArXiv paper download | Minor update (use config) |
| `ingestion/semantic_scholar_client.py` | Semantic Scholar metadata + citation graph | Implement |
| `parsing/pdf_parser.py` | PDF text extraction + sentence splitting | Fix whitespace bug |
| `parsing/section_segmenter.py` | Section detection from text | Fix regex anchoring |
| `claims/claim_extractor.py` | Atomic claim extraction via Ollama | Rewrite |
| `claims/prompts.py` | Few-shot SciFact prompts | Expand |
| `claims/limitation_retriever.py` | Seed query retrieval for limitations | New |
| `embedding/specter_embedder.py` | SPECTER2 claim embeddings | Swap model |
| `clustering/hdbscan_cluster.py` | HDBSCAN clustering | Minor update (use config) |
| `nli/nli_engine.py` | DeBERTa contradiction detection | Fix label order bug |
| `gap_generation/generator.py` | Gap synthesis via Ollama | Rewrite |
| `gap_generation/prompts.py` | Gap generation prompts | Implement |
| `evaluation/evaluator.py` | SciFact evaluation runner | Implement |
| `evaluation/metrics.py` | F1, BERTScore computation | Implement |
| `main.py` | CLI entry point | Refactor to full pipeline |
| `app.py` | Streamlit entry point | Refactor to full pipeline + NLI |
| `requirements.txt` | Dependencies | Add new packages |

---

### Task 1: Centralize Configuration

**Files:**
- Implement: `config/settings.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Implement config/settings.py**

```python
# config/settings.py

# --- Ingestion ---
ARXIV_MAX_RESULTS = 50
SEMANTIC_SCHOLAR_API_URL = "https://api.semanticscholar.org/graph/v1"
PAPER_SAVE_DIR = "data/raw_papers"

# --- Models ---
OLLAMA_MODEL = "mistral"
OLLAMA_URL = "http://localhost:11434/api/generate"
SPECTER_MODEL = "allenai/specter2"
NLI_MODEL = "MoritzLaurer/deberta-v3-base-zeroshot-v1"
SPACY_MODEL = "en_core_web_sm"

# --- Claim Extraction ---
MIN_SENTENCE_LENGTH = 40
MIN_CLAIM_LENGTH = 30

# --- Clustering ---
HDBSCAN_MIN_CLUSTER_SIZE = 2
HDBSCAN_MIN_SAMPLES = 1
HDBSCAN_EPSILON = 0.8

# --- NLI ---
COSINE_SIM_MIN = 0.7
COSINE_SIM_MAX = 0.9
NLI_CONFIDENCE_THRESHOLD = 0.7

# --- Limitation Retrieval ---
LIMITATION_SIMILARITY_THRESHOLD = 0.5
LIMITATION_SEED_QUERIES = [
    "unresolved problem",
    "our approach does not handle",
    "limitation of this work",
    "future work should address",
    "remains challenging",
    "open question in the field",
    "current methods fail to",
    "has not been explored",
]

# --- Gap Generation ---
MIN_CLAIM_LENGTH_FOR_GAP = 50
MAX_CLAIMS_PER_GAP = 5
```

- [ ] **Step 2: Update requirements.txt**

```
requests
tqdm
pymupdf
spacy
transformers
torch
sentence-transformers
scikit-learn
hdbscan
streamlit
bert-score
datasets
```

- [ ] **Step 3: Install new dependencies**

Run: `source .venv/bin/activate && pip install bert-score datasets`

- [ ] **Step 4: Commit**

```bash
git add config/settings.py requirements.txt
git commit -m "Add centralized config and new dependencies

Move all hardcoded thresholds, model names, and URLs into
config/settings.py so nothing is scattered across pipeline files.
Add bert-score and datasets for SciFact evaluation."
```

---

### Task 2: Fix PDF Parser Whitespace Destruction

**Files:**
- Modify: `parsing/pdf_parser.py`

- [ ] **Step 1: Fix _clean_text to preserve paragraph structure**

Replace the entire `parsing/pdf_parser.py` with:

```python
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
```

- [ ] **Step 2: Verify it works**

Run: `source .venv/bin/activate && python -c "from parsing.pdf_parser import PDFParser; p = PDFParser(); text = p.extract_text('data/raw_papers/2504.13684v1.pdf'); print(text[:500]); print('---'); print('Paragraph breaks:', text.count('\n\n'))"`

Expected: Text with paragraph breaks preserved, not a single collapsed line.

- [ ] **Step 3: Commit**

```bash
git add parsing/pdf_parser.py
git commit -m "Fix PDF parser whitespace destruction

Previous implementation collapsed ALL whitespace to single spaces,
destroying paragraph structure. Now preserves double-newlines as
paragraph breaks while normalizing spaces within lines. Also uses
word boundary in References split to avoid false matches."
```

---

### Task 3: Fix Section Segmenter Regex

**Files:**
- Modify: `parsing/section_segmenter.py`

- [ ] **Step 1: Anchor regex patterns to line starts**

Replace the entire `parsing/section_segmenter.py` with:

```python
import re


class SectionSegmenter:
    PATTERNS = [
        "abstract", "introduction", "method", "methods",
        "methodology", "results", "discussion", "conclusion",
        "related work", "background", "evaluation", "experiments"
    ]

    def segment(self, text):
        sections = {}

        for p in self.PATTERNS:
            # Match section headers at line starts, with optional numbering
            pattern = rf'^\s*(?:\d+\.?\s*)?{p}\s*$'
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                sections[p] = match.start()

        sorted_sections = sorted(sections.items(), key=lambda x: x[1])

        segmented = {}

        for i, (name, start) in enumerate(sorted_sections):
            end = sorted_sections[i + 1][1] if i + 1 < len(sorted_sections) else len(text)
            segmented[name] = text[start:end]

        return segmented
```

- [ ] **Step 2: Verify it works**

Run: `source .venv/bin/activate && python -c "from parsing.section_segmenter import SectionSegmenter; s = SectionSegmenter(); result = s.segment('1. Introduction\nSome text about introduction methods.\n2. Methods\nMethod details.\n3. Results\nResults here.'); print(list(result.keys()))"`

Expected: `['introduction', 'method', 'results']` — the word "introduction" in body text should NOT create a false match.

- [ ] **Step 3: Commit**

```bash
git add parsing/section_segmenter.py
git commit -m "Fix section segmenter to anchor patterns to line starts

Previous regex matched section keywords anywhere in body text,
causing false segmentation. Now requires patterns to appear at
the start of a line with optional numbering. Also added more
section patterns: methodology, related work, background,
evaluation, experiments."
```

---

### Task 4: Fix NLI Label Order Bug

**Files:**
- Modify: `nli/nli_engine.py`

- [ ] **Step 1: Read label order from model config**

Replace the entire `nli/nli_engine.py` with:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from config.settings import NLI_MODEL


class NLIEngine:
    def __init__(self):
        print("Loading NLI model...")

        self.tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)

        # Read label order from model config instead of hardcoding
        self.labels = [
            self.model.config.id2label[i]
            for i in range(len(self.model.config.id2label))
        ]

    def predict(self, c1, c2):
        inputs = self.tokenizer(c1, c2, return_tensors="pt", truncation=True)

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1)[0]

        idx = torch.argmax(probs).item()
        return self.labels[idx], probs[idx].item()
```

- [ ] **Step 2: Verify label order is read correctly**

Run: `source .venv/bin/activate && python -c "from nli.nli_engine import NLIEngine; e = NLIEngine(); print('Labels:', e.labels); r, c = e.predict('The earth is round', 'The earth is flat'); print(f'{r} ({c:.2f})')"`

Expected: Labels printed from model config (should be `['entailment', 'neutral', 'contradiction']`), and the pair should be classified as contradiction.

- [ ] **Step 3: Commit**

```bash
git add nli/nli_engine.py
git commit -m "Fix NLI label order by reading from model config

Previously hardcoded label order as [contradiction, neutral,
entailment] which could silently misclassify if the model uses
a different order. Now reads id2label from model.config at
runtime, eliminating the risk of label mismatch."
```

---

### Task 5: Upgrade Embeddings to SPECTER2

**Files:**
- Modify: `embedding/specter_embedder.py`

- [ ] **Step 1: Swap to SPECTER2**

Replace the entire `embedding/specter_embedder.py` with:

```python
from sentence_transformers import SentenceTransformer
from config.settings import SPECTER_MODEL


class SpecterEmbedder:
    def __init__(self):
        print(f"Loading {SPECTER_MODEL}...")
        self.model = SentenceTransformer(SPECTER_MODEL)

    def encode(self, claims):
        texts = [c["text"] if isinstance(c, dict) else c for c in claims]
        return self.model.encode(texts, batch_size=8, show_progress_bar=True)
```

Note: The `encode` method now handles both plain strings and claim dicts (with metadata). This is needed for the citation tracing feature in later tasks.

- [ ] **Step 2: Verify it loads and encodes**

Run: `source .venv/bin/activate && python -c "from embedding.specter_embedder import SpecterEmbedder; e = SpecterEmbedder(); emb = e.encode(['Neural networks learn representations']); print('Shape:', emb.shape)"`

Expected: Shape `(1, 768)` — SPECTER2 outputs 768-d vectors.

- [ ] **Step 3: Commit**

```bash
git add embedding/specter_embedder.py
git commit -m "Upgrade embeddings from SPECTER to SPECTER2

Pre-proposal specifies SPECTER2 (Singh et al., 2023), trained on
6M citation triplets across 23 fields. Also added support for
claim dicts with metadata alongside plain strings."
```

---

### Task 6: Rewrite Claim Extraction for Ollama + Few-Shot Prompts

**Files:**
- Modify: `claims/prompts.py`
- Modify: `claims/claim_extractor.py`

- [ ] **Step 1: Expand prompts.py with SciFact few-shot examples**

Replace the entire `claims/prompts.py` with:

```python
CLAIM_EXTRACTION_PROMPT = """You are a scientific claim extractor. Given a sentence from a research paper, extract all atomic, independently verifiable scientific claims.

Rules:
- Each claim must be a complete, standalone sentence
- Each claim must be independently verifiable (no "it", "this method", etc. without the referent)
- Decompose compound claims into individual atomic claims
- Preserve the original meaning — do not add interpretation
- Do NOT return metadata (author names, citations, URLs, dates)
- Do NOT return fragments or incomplete sentences
- If the sentence contains no verifiable scientific claim, return NONE

Examples from SciFact:

Sentence: "BERT outperforms GPT-2 on all GLUE benchmark tasks while using fewer parameters."
Claims:
- BERT outperforms GPT-2 on all GLUE benchmark tasks.
- BERT uses fewer parameters than GPT-2.

Sentence: "Our experiments show that retrieval-augmented generation improves factual accuracy but increases latency by 40%."
Claims:
- Retrieval-augmented generation improves factual accuracy.
- Retrieval-augmented generation increases latency by 40%.

Sentence: "We thank the anonymous reviewers for their feedback."
Claims:
NONE

Sentence: "The proposed method achieves state-of-the-art results on SQuAD, reducing error rate by 15% compared to the previous best model, while maintaining similar inference speed."
Claims:
- The proposed method achieves state-of-the-art results on SQuAD.
- The proposed method reduces error rate by 15% compared to the previous best model on SQuAD.
- The proposed method maintains similar inference speed to the previous best model.

Now extract claims from:

Sentence: "{sentence}"
Claims:
"""
```

- [ ] **Step 2: Rewrite claim_extractor.py for Ollama**

Replace the entire `claims/claim_extractor.py` with:

```python
import requests
from config.settings import OLLAMA_MODEL, OLLAMA_URL, MIN_CLAIM_LENGTH
from claims.prompts import CLAIM_EXTRACTION_PROMPT


class ClaimExtractor:
    def __init__(self):
        print(f"Using Ollama ({OLLAMA_MODEL}) for claim extraction")

    def extract_from_sentence(self, sentence, paper_title="", section=""):
        prompt = CLAIM_EXTRACTION_PROMPT.format(sentence=sentence)

        response = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 256}
        })

        if response.status_code != 200:
            print(f"Ollama error: {response.status_code}")
            return []

        text = response.json().get("response", "")
        claims = self._parse_output(text)

        # Attach source metadata for citation tracing
        return [
            {
                "text": c,
                "paper_title": paper_title,
                "section": section,
                "source_sentence": sentence,
            }
            for c in claims
        ]

    def _parse_output(self, text):
        if "NONE" in text.strip().upper():
            return []

        lines = text.strip().split("\n")
        claims = []

        for line in lines:
            line = line.strip().lstrip("- ").lstrip("* ").strip()

            if not line:
                continue

            low = line.lower()

            # Skip metadata / garbage
            if any(x in low for x in [
                "acm", "workshop", "arxiv", "doi",
                "copyright", "permission", "manuscript",
                "author", "email", "@", "http", "www",
                "sentence:", "claims:", "none"
            ]):
                continue

            if len(line) < MIN_CLAIM_LENGTH:
                continue

            if not line.endswith("."):
                line += "."

            claims.append(line)

        return claims
```

- [ ] **Step 3: Verify Ollama is running and extraction works**

Run: `ollama list` to check Mistral is available. If not: `ollama pull mistral`

Then test:
Run: `source .venv/bin/activate && python -c "from claims.claim_extractor import ClaimExtractor; e = ClaimExtractor(); claims = e.extract_from_sentence('BERT outperforms GPT-2 on all GLUE tasks while using fewer parameters.', paper_title='Test Paper', section='Introduction'); print(claims)"`

Expected: List of claim dicts with `text`, `paper_title`, `section`, `source_sentence` keys.

- [ ] **Step 4: Commit**

```bash
git add claims/claim_extractor.py claims/prompts.py
git commit -m "Rewrite claim extraction for Mistral via Ollama

Replace FLAN-T5-small with Mistral 7B via Ollama for much better
claim decomposition. Add few-shot SciFact examples to the prompt
as specified in the pre-proposal. Claims now carry source metadata
(paper_title, section, source_sentence) for citation tracing.
Wire in the prompts.py file that was previously defined but never
imported."
```

---

### Task 7: Rewrite Gap Generation for Ollama + Citations

**Files:**
- Implement: `gap_generation/prompts.py`
- Modify: `gap_generation/generator.py`

- [ ] **Step 1: Implement gap generation prompts**

Replace the entire `gap_generation/prompts.py` with:

```python
GAP_GENERATION_PROMPT = """You are a research gap analyst. Given a cluster of related scientific claims from multiple papers, identify ONE specific research gap — something that is missing, underexplored, or contradicted.

Rules:
- The gap must be specific and actionable (not generic like "more research is needed")
- Ground the gap in the provided claims — explain what the claims show and what is missing
- Reference the source papers by name
- Format: Start with "Research Gap:" followed by the gap statement

Claims:
{claims_text}

Research Gap:
"""

CONTRADICTION_GAP_PROMPT = """You are a research gap analyst. The following pairs of claims from different papers contradict each other. Identify the specific research gap this contradiction reveals.

Rules:
- Explain the contradiction clearly
- Identify what experiment, study, or analysis would resolve the disagreement
- Reference the source papers by name
- Format: Start with "Research Gap:" followed by the gap statement

Contradictions:
{contradictions_text}

Research Gap:
"""
```

- [ ] **Step 2: Rewrite generator.py for Ollama + citations**

Replace the entire `gap_generation/generator.py` with:

```python
import requests
from config.settings import (
    OLLAMA_MODEL, OLLAMA_URL,
    MIN_CLAIM_LENGTH_FOR_GAP, MAX_CLAIMS_PER_GAP
)
from gap_generation.prompts import GAP_GENERATION_PROMPT, CONTRADICTION_GAP_PROMPT


class GapGenerator:
    def __init__(self):
        print(f"Using Ollama ({OLLAMA_MODEL}) for gap generation")

    def generate_gap(self, cluster_claims, contradictions=None):
        if contradictions:
            return self._generate_contradiction_gap(contradictions)
        return self._generate_cluster_gap(cluster_claims)

    def _generate_cluster_gap(self, cluster_claims):
        clean = [c for c in cluster_claims if len(c["text"]) > MIN_CLAIM_LENGTH_FOR_GAP]
        clean = clean[:MAX_CLAIMS_PER_GAP]

        if not clean:
            return {
                "gap": "Insufficient claim data for gap generation.",
                "supporting_claims": [],
                "source_papers": [],
                "type": "open_question",
            }

        claims_text = "\n".join(
            f'- "{c["text"]}" (from: {c["paper_title"]}, section: {c["section"]})'
            for c in clean
        )

        prompt = GAP_GENERATION_PROMPT.format(claims_text=claims_text)
        gap_text = self._call_ollama(prompt)

        source_papers = list({c["paper_title"] for c in clean})

        return {
            "gap": gap_text,
            "supporting_claims": [c["text"] for c in clean],
            "source_papers": source_papers,
            "type": "open_question",
        }

    def _generate_contradiction_gap(self, contradictions):
        lines = []
        all_claims = []
        for c1, c2, conf in contradictions:
            lines.append(
                f'- "{c1["text"]}" (from: {c1["paper_title"]})\n'
                f'  vs. "{c2["text"]}" (from: {c2["paper_title"]})\n'
                f'  [confidence: {conf:.2f}]'
            )
            all_claims.extend([c1, c2])

        prompt = CONTRADICTION_GAP_PROMPT.format(contradictions_text="\n".join(lines))
        gap_text = self._call_ollama(prompt)

        source_papers = list({c["paper_title"] for c in all_claims})

        return {
            "gap": gap_text,
            "supporting_claims": [c["text"] for c in all_claims],
            "source_papers": source_papers,
            "type": "contradiction",
        }

    def _call_ollama(self, prompt):
        response = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 300}
        })

        if response.status_code != 200:
            return "Error: Could not generate gap."

        text = response.json().get("response", "").strip()

        # Clean up: remove "Research Gap:" prefix if present
        if text.lower().startswith("research gap:"):
            text = text[len("research gap:"):].strip()

        return f"Research Gap: {text}" if text else "Research Gap: Unable to synthesize gap from provided claims."
```

- [ ] **Step 3: Verify gap generation works**

Run: `source .venv/bin/activate && python -c "
from gap_generation.generator import GapGenerator
g = GapGenerator()
claims = [
    {'text': 'RAG improves factual accuracy in language models.', 'paper_title': 'Paper A', 'section': 'Results', 'source_sentence': 'test'},
    {'text': 'RAG increases inference latency by 40 percent.', 'paper_title': 'Paper B', 'section': 'Discussion', 'source_sentence': 'test'},
]
result = g.generate_gap(claims)
print(result)
"`

Expected: A dict with `gap`, `supporting_claims`, `source_papers`, `type` keys.

- [ ] **Step 4: Commit**

```bash
git add gap_generation/generator.py gap_generation/prompts.py
git commit -m "Rewrite gap generation for Mistral via Ollama with citations

Replace FLAN-T5-base and rule-based fallback with Mistral 7B.
Gaps now include citation pointers (source papers, supporting claims)
as required by the pre-proposal. Two prompt modes: cluster-based
gaps and contradiction-based gaps. Output is structured dict with
gap text, sources, and gap type."
```

---

### Task 8: Implement Semantic Scholar Client

**Files:**
- Implement: `ingestion/semantic_scholar_client.py`

- [ ] **Step 1: Implement the client**

Replace the entire `ingestion/semantic_scholar_client.py` with:

```python
import requests
import os
import time
from config.settings import SEMANTIC_SCHOLAR_API_URL, PAPER_SAVE_DIR


class SemanticScholarClient:
    """Supplements ArXiv papers with metadata and citation graph discovery."""

    def enrich_and_expand(self, papers, max_extra_papers=10):
        """
        For each paper, fetch metadata from Semantic Scholar.
        Use citation graph to discover related papers.
        Returns enriched paper list + newly discovered papers.
        """
        enriched = []
        seen_titles = {p["title"].lower() for p in papers}
        discovered = []

        for paper in papers:
            metadata = self._fetch_metadata(paper["title"])
            if metadata:
                paper["citation_count"] = metadata.get("citationCount", 0)
                paper["year"] = metadata.get("year")
                paper["venue"] = metadata.get("venue", "")
                paper["semantic_scholar_id"] = metadata.get("paperId", "")

                # Discover related papers from references
                refs = self._fetch_references(metadata["paperId"])
                for ref in refs:
                    ref_title = ref.get("title", "").lower()
                    if ref_title and ref_title not in seen_titles:
                        seen_titles.add(ref_title)
                        discovered.append(ref)

            enriched.append(paper)
            time.sleep(0.5)  # Rate limiting

        # Download top cited discovered papers
        extra = self._download_discovered(discovered, max_extra_papers)

        return enriched, extra

    def _fetch_metadata(self, title):
        """Search Semantic Scholar by title."""
        try:
            response = requests.get(
                f"{SEMANTIC_SCHOLAR_API_URL}/paper/search",
                params={"query": title, "limit": 1,
                        "fields": "paperId,title,citationCount,year,venue"},
                timeout=10,
            )
            if response.status_code == 200:
                results = response.json().get("data", [])
                return results[0] if results else None
        except requests.RequestException:
            pass
        return None

    def _fetch_references(self, paper_id):
        """Get references for a paper."""
        try:
            response = requests.get(
                f"{SEMANTIC_SCHOLAR_API_URL}/paper/{paper_id}/references",
                params={"fields": "title,citationCount,year,externalIds", "limit": 20},
                timeout=10,
            )
            if response.status_code == 200:
                refs = response.json().get("data", [])
                return [r["citedPaper"] for r in refs if r.get("citedPaper")]
        except requests.RequestException:
            pass
        return []

    def _download_discovered(self, discovered, max_papers):
        """Download PDFs for top-cited discovered papers that have ArXiv IDs."""
        # Sort by citation count, take top N
        with_citations = [d for d in discovered if d.get("citationCount")]
        with_citations.sort(key=lambda x: x.get("citationCount", 0), reverse=True)

        extra_papers = []
        for paper in with_citations[:max_papers]:
            arxiv_id = (paper.get("externalIds") or {}).get("ArXiv")
            if not arxiv_id:
                continue

            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
            file_name = f"{arxiv_id}.pdf"
            file_path = os.path.join(PAPER_SAVE_DIR, file_name)

            if not os.path.exists(file_path):
                try:
                    resp = requests.get(pdf_url, stream=True, timeout=30)
                    if resp.status_code == 200:
                        os.makedirs(PAPER_SAVE_DIR, exist_ok=True)
                        with open(file_path, "wb") as f:
                            for chunk in resp.iter_content(chunk_size=1024):
                                if chunk:
                                    f.write(chunk)
                    else:
                        continue
                except requests.RequestException:
                    continue

            extra_papers.append({
                "title": paper.get("title", "Unknown"),
                "pdf_path": file_path,
                "citation_count": paper.get("citationCount", 0),
                "year": paper.get("year"),
            })
            time.sleep(1)  # Rate limiting for downloads

        return extra_papers
```

- [ ] **Step 2: Verify it works**

Run: `source .venv/bin/activate && python -c "
from ingestion.semantic_scholar_client import SemanticScholarClient
s = SemanticScholarClient()
meta = s._fetch_metadata('Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks')
print(meta)
"`

Expected: Dict with `paperId`, `title`, `citationCount`, `year`, `venue`.

- [ ] **Step 3: Commit**

```bash
git add ingestion/semantic_scholar_client.py
git commit -m "Implement Semantic Scholar client for metadata and citation graph

Pre-proposal requires Semantic Scholar to supplement ArXiv with
structured metadata and citation graphs. Client fetches citation
counts, venue, year for each paper, then discovers related papers
via the citation graph. Top-cited discoveries with ArXiv IDs are
auto-downloaded."
```

---

### Task 9: Implement Limitation Retriever

**Files:**
- Create: `claims/limitation_retriever.py`

- [ ] **Step 1: Implement limitation retriever**

Create `claims/limitation_retriever.py`:

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from config.settings import LIMITATION_SEED_QUERIES, LIMITATION_SIMILARITY_THRESHOLD


class LimitationRetriever:
    """
    Retrieves limitation-adjacent claims using seed query similarity,
    then re-clusters them to identify recurring limitation themes.
    """

    def __init__(self, embedder, clusterer):
        self.embedder = embedder
        self.clusterer = clusterer

    def retrieve(self, all_claims, all_embeddings):
        """
        Find claims similar to limitation seed queries.
        Returns: (limitation_claims, limitation_clusters)
        """
        # Embed seed queries
        seed_embeddings = self.embedder.encode(LIMITATION_SEED_QUERIES)

        # Compute similarity of every claim against every seed query
        similarities = cosine_similarity(all_embeddings, seed_embeddings)

        # A claim is limitation-adjacent if it's similar to ANY seed query
        max_similarities = np.max(similarities, axis=1)

        # Select claims above threshold
        limitation_indices = np.where(max_similarities >= LIMITATION_SIMILARITY_THRESHOLD)[0]

        if len(limitation_indices) == 0:
            return [], {}

        limitation_claims = [all_claims[i] for i in limitation_indices]
        limitation_embeddings = all_embeddings[limitation_indices]

        # Re-cluster limitation claims
        if len(limitation_claims) < 2:
            return limitation_claims, {0: limitation_claims}

        labels = self.clusterer.cluster(limitation_embeddings)
        clusters = self.clusterer.group_clusters(limitation_claims, labels)

        # If all noise, put them in one group
        if not clusters:
            clusters = {0: limitation_claims}

        return limitation_claims, clusters
```

- [ ] **Step 2: Verify it works**

Run: `source .venv/bin/activate && python -c "
from embedding.specter_embedder import SpecterEmbedder
from clustering.hdbscan_cluster import ClaimClusterer
from claims.limitation_retriever import LimitationRetriever
import numpy as np

embedder = SpecterEmbedder()
clusterer = ClaimClusterer()
retriever = LimitationRetriever(embedder, clusterer)

claims = [
    {'text': 'Our method does not handle multi-hop reasoning.', 'paper_title': 'A', 'section': 'Discussion', 'source_sentence': 'test'},
    {'text': 'Neural networks achieve 95% accuracy on ImageNet.', 'paper_title': 'B', 'section': 'Results', 'source_sentence': 'test'},
    {'text': 'A limitation of this work is the small dataset size.', 'paper_title': 'C', 'section': 'Discussion', 'source_sentence': 'test'},
]
embs = embedder.encode(claims)
lim_claims, lim_clusters = retriever.retrieve(claims, embs)
print(f'Found {len(lim_claims)} limitation claims')
print(f'Clusters: {list(lim_clusters.keys())}')
"`

Expected: At least 2 limitation claims found (the "does not handle" and "limitation of" ones).

- [ ] **Step 3: Commit**

```bash
git add claims/limitation_retriever.py
git commit -m "Implement limitation retrieval via seed query similarity

Pre-proposal specifies semantic retrieval using seed queries like
'unresolved problem' and 'our approach does not handle' to find
limitation-adjacent claims. These are re-clustered with HDBSCAN
to identify recurring limitation themes for gap generation."
```

---

### Task 10: Update Clustering to Use Config

**Files:**
- Modify: `clustering/hdbscan_cluster.py`

- [ ] **Step 1: Use config values and handle claim dicts**

Replace the entire `clustering/hdbscan_cluster.py` with:

```python
import hdbscan
from config.settings import HDBSCAN_MIN_CLUSTER_SIZE, HDBSCAN_MIN_SAMPLES, HDBSCAN_EPSILON


class ClaimClusterer:
    def __init__(self):
        print("Initializing HDBSCAN...")

        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
            min_samples=HDBSCAN_MIN_SAMPLES,
            cluster_selection_epsilon=HDBSCAN_EPSILON,
        )

    def cluster(self, embeddings):
        return self.clusterer.fit_predict(embeddings)

    def group_clusters(self, claims, labels):
        clusters = {}

        for claim, label in zip(claims, labels):
            if label == -1:
                continue

            clusters.setdefault(label, []).append(claim)

        return clusters
```

- [ ] **Step 2: Commit**

```bash
git add clustering/hdbscan_cluster.py
git commit -m "Update clustering to use centralized config

Replace hardcoded HDBSCAN parameters with values from
config/settings.py for easy tuning."
```

---

### Task 11: Update ArXiv Client to Use Config

**Files:**
- Modify: `ingestion/arxiv_client.py`

- [ ] **Step 1: Use config for save_dir**

Replace the entire `ingestion/arxiv_client.py` with:

```python
import requests
import xml.etree.ElementTree as ET
import os
from config.settings import PAPER_SAVE_DIR


class ArxivClient:
    BASE_URL = "http://export.arxiv.org/api/query"

    def search_and_download(self, query, max_results=50, save_dir=None):
        save_dir = save_dir or PAPER_SAVE_DIR
        os.makedirs(save_dir, exist_ok=True)

        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
        }

        response = requests.get(self.BASE_URL, params=params, timeout=30)

        if response.status_code != 200:
            raise Exception(f"Failed to fetch data from arXiv (status {response.status_code})")

        return self._parse_and_download(response.text, save_dir)

    def _parse_and_download(self, xml_data, save_dir):
        root = ET.fromstring(xml_data)
        papers = []

        for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
            title = entry.find("{http://www.w3.org/2005/Atom}title").text.strip()
            pdf_url = None

            for link in entry.findall("{http://www.w3.org/2005/Atom}link"):
                if link.attrib.get("title") == "pdf":
                    pdf_url = link.attrib["href"]

            if pdf_url:
                file_path = self._download_pdf(pdf_url, save_dir)
                papers.append({
                    "title": title,
                    "pdf_path": file_path,
                })

        return papers

    def _download_pdf(self, url, save_dir):
        file_name = url.split("/")[-1] + ".pdf"
        file_path = os.path.join(save_dir, file_name)

        if os.path.exists(file_path):
            return file_path

        response = requests.get(url, stream=True, timeout=60)

        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

        return file_path
```

- [ ] **Step 2: Commit**

```bash
git add ingestion/arxiv_client.py
git commit -m "Update ArXiv client to use config and add timeouts

Use PAPER_SAVE_DIR from config, default max_results to 50 as the
pre-proposal specifies 50-100 papers. Add request timeouts to
prevent hangs. Add sortBy=relevance for better results."
```

---

### Task 12: Refactor main.py to Full Pipeline

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Rewrite main.py with full pipeline + citation tracing**

Replace the entire `main.py` with:

```python
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
```

- [ ] **Step 2: Commit**

```bash
git add main.py
git commit -m "Refactor main.py to full pipeline with all stages

Now runs: ArXiv + Semantic Scholar → parsing → claim extraction
with metadata → SPECTER2 embeddings → clustering → NLI contradiction
detection → limitation retrieval → gap generation with citations.
Extracted run_pipeline() as reusable function for app.py.
Removed hardcoded query limit of 3 papers, sentence cap of 10,
and manual ACM filtering."
```

---

### Task 13: Refactor app.py to Full Pipeline

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Rewrite app.py with full pipeline + NLI**

Replace the entire `app.py` with:

```python
import streamlit as st
from main import run_pipeline
from config.settings import ARXIV_MAX_RESULTS

st.set_page_config(page_title="Research Gap Finder", layout="wide")

st.title("Research Gap Discovery System")
st.markdown("Analyze research papers to find contradictions, limitations, and open questions.")

col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input("Enter research topic:", "retrieval augmented generation")
with col2:
    max_papers = st.number_input("Max papers:", min_value=3, max_value=100, value=ARXIV_MAX_RESULTS)

run = st.button("Run Analysis")

if run:
    with st.spinner("Running pipeline... (this may take several minutes)"):
        results = run_pipeline(query, max_results=max_papers)

    if not results:
        st.error("Pipeline failed — no claims extracted.")
        st.stop()

    # --- Summary ---
    st.write("## Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Papers", results["papers_analyzed"])
    c2.metric("Claims", results["total_claims"])
    c3.metric("Clusters", results["clusters"])
    c4.metric("Contradictions", results["contradictions"])

    # --- Research Gaps ---
    st.write("## Research Gaps")

    for i, gap in enumerate(results["gaps"]):
        gap_type = gap.get("type", "open_question")
        icon = {"contradiction": "!!", "limitation": ">>", "open_question": "??"}
        label = f"[{icon.get(gap_type, '??')}] Gap {i + 1}: {gap_type.replace('_', ' ').title()}"

        with st.expander(label, expanded=(i < 3)):
            st.success(gap["gap"])

            if gap.get("source_papers"):
                st.write("**Source Papers:**")
                for p in gap["source_papers"]:
                    st.write(f"- {p}")

            if gap.get("supporting_claims"):
                st.write("**Supporting Claims:**")
                for c in gap["supporting_claims"]:
                    st.write(f"- {c}")
```

- [ ] **Step 2: Commit**

```bash
git add app.py
git commit -m "Refactor app.py to use shared pipeline with full features

Now calls run_pipeline() from main.py so both CLI and web UI run
identical pipelines including NLI contradiction detection and
limitation retrieval. Added summary metrics, configurable paper
count, and structured gap display with source citations."
```

---

### Task 14: Implement Evaluation Module

**Files:**
- Implement: `evaluation/metrics.py`
- Implement: `evaluation/evaluator.py`

- [ ] **Step 1: Implement metrics.py**

Replace the entire `evaluation/metrics.py` with:

```python
from sklearn.metrics import precision_score, recall_score, f1_score
from bert_score import score as bert_score
import numpy as np


def compute_claim_f1(predicted_claims, gold_claims, threshold=0.8):
    """
    Compute precision, recall, F1 for claim extraction using BERTScore
    for semantic matching (accounts for valid paraphrasing).
    """
    if not predicted_claims or not gold_claims:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # BERTScore: compare each predicted claim against all gold claims
    P, R, F1 = bert_score(
        predicted_claims, gold_claims,
        lang="en", rescale_with_baseline=True, verbose=False,
    )

    # A predicted claim "matches" if its best BERTScore F1 >= threshold
    pred_matched = (F1.max(dim=1).values >= threshold).float()
    precision = pred_matched.mean().item()

    # For recall: check each gold claim against all predicted claims
    P2, R2, F12 = bert_score(
        gold_claims, predicted_claims,
        lang="en", rescale_with_baseline=True, verbose=False,
    )
    gold_matched = (F12.max(dim=1).values >= threshold).float()
    recall = gold_matched.mean().item()

    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def compute_nli_f1(predictions, gold_labels):
    """
    Compute F1 for NLI contradiction detection.
    predictions: list of predicted labels ("contradiction", "entailment", "neutral")
    gold_labels: list of gold labels
    """
    label_map = {"contradiction": 0, "entailment": 1, "neutral": 2}

    pred_ids = [label_map.get(p, 2) for p in predictions]
    gold_ids = [label_map.get(g, 2) for g in gold_labels]

    return {
        "f1_macro": f1_score(gold_ids, pred_ids, average="macro"),
        "f1_contradiction": f1_score(gold_ids, pred_ids, average="binary", pos_label=0),
        "precision_contradiction": precision_score(gold_ids, pred_ids, average="binary", pos_label=0),
        "recall_contradiction": recall_score(gold_ids, pred_ids, average="binary", pos_label=0),
    }
```

- [ ] **Step 2: Implement evaluator.py**

Replace the entire `evaluation/evaluator.py` with:

```python
import json
from datasets import load_dataset
from claims.claim_extractor import ClaimExtractor
from nli.nli_engine import NLIEngine
from evaluation.metrics import compute_claim_f1, compute_nli_f1


class SciFactEvaluator:
    """Evaluate claim extraction and NLI against SciFact dataset."""

    def __init__(self):
        self.extractor = ClaimExtractor()
        self.nli = NLIEngine()

    def load_scifact(self):
        """Load SciFact dataset from HuggingFace."""
        dataset = load_dataset("allenai/scifact", "claims")
        return dataset

    def evaluate_claim_extraction(self, num_samples=100):
        """
        Run claim extractor on SciFact abstracts and compare
        against ground-truth claims.
        """
        print(f"Evaluating claim extraction on {num_samples} samples...")
        dataset = self.load_scifact()

        predicted = []
        gold = []

        for i, example in enumerate(dataset["train"]):
            if i >= num_samples:
                break

            claim = example["claim"]
            gold.append(claim)

            # Extract claims from the claim text itself
            # (SciFact claims are already atomic, so good extraction = high similarity)
            extracted = self.extractor.extract_from_sentence(claim)
            if extracted:
                predicted.append(extracted[0]["text"])
            else:
                predicted.append("")

        results = compute_claim_f1(predicted, gold)
        print(f"Claim Extraction - P: {results['precision']:.3f}, "
              f"R: {results['recall']:.3f}, F1: {results['f1']:.3f}")
        return results

    def evaluate_nli(self, num_samples=200):
        """
        Run NLI on SciFact claim-abstract pairs and measure
        contradiction detection F1.
        """
        print(f"Evaluating NLI on {num_samples} samples...")
        dataset = self.load_scifact()

        predictions = []
        gold_labels = []

        label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}

        for i, example in enumerate(dataset["train"]):
            if i >= num_samples:
                break

            claim = example["claim"]
            evidence = example.get("evidence", "")
            label = example.get("label")

            if not evidence or label is None:
                continue

            gold_label = label_map.get(label, "neutral")
            gold_labels.append(gold_label)

            pred_label, _ = self.nli.predict(claim, evidence)
            predictions.append(pred_label)

        results = compute_nli_f1(predictions, gold_labels)
        print(f"NLI - F1 (macro): {results['f1_macro']:.3f}, "
              f"F1 (contradiction): {results['f1_contradiction']:.3f}")
        return results

    def run_full_evaluation(self):
        """Run all evaluations and save results."""
        print("\n=== SciFact Evaluation ===\n")

        claim_results = self.evaluate_claim_extraction()
        nli_results = self.evaluate_nli()

        results = {
            "claim_extraction": claim_results,
            "nli_detection": nli_results,
        }

        with open("data/evaluation_results.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to data/evaluation_results.json")
        return results


if __name__ == "__main__":
    evaluator = SciFactEvaluator()
    evaluator.run_full_evaluation()
```

- [ ] **Step 3: Verify imports work**

Run: `source .venv/bin/activate && python -c "from evaluation.metrics import compute_claim_f1, compute_nli_f1; from evaluation.evaluator import SciFactEvaluator; print('Evaluation module OK')"`

Expected: "Evaluation module OK"

- [ ] **Step 4: Commit**

```bash
git add evaluation/evaluator.py evaluation/metrics.py
git commit -m "Implement SciFact evaluation module

Add evaluator.py and metrics.py as specified in the pre-proposal
evaluation plan. Measures claim extraction quality (P/R/F1 with
BERTScore for semantic matching) and NLI contradiction detection
(F1 on SciFact REFUTE pairs). Results saved to JSON."
```

---

### Task 15: Add __init__.py Files

**Files:**
- Create: `config/__init__.py`
- Create: `ingestion/__init__.py`
- Create: `parsing/__init__.py`
- Create: `claims/__init__.py`
- Create: `embedding/__init__.py`
- Create: `clustering/__init__.py`
- Create: `nli/__init__.py`
- Create: `gap_generation/__init__.py`
- Create: `evaluation/__init__.py`

- [ ] **Step 1: Check which __init__.py files exist**

Run: `find /Users/hirakdesai/Developer/USC/NLP-project -name "__init__.py" -type f`

- [ ] **Step 2: Create any missing __init__.py files**

For each package directory that lacks an `__init__.py`, create an empty one. This ensures Python treats them as proper packages.

- [ ] **Step 3: Commit (only if new files were created)**

```bash
git add */__init__.py
git commit -m "Add missing __init__.py files for package imports"
```

---

### Task 16: End-to-End Smoke Test

- [ ] **Step 1: Ensure Ollama is running with Mistral**

Run: `ollama list | grep mistral`

If not found: `ollama pull mistral`

- [ ] **Step 2: Run CLI pipeline with small test**

Run: `source .venv/bin/activate && python -c "
from main import run_pipeline
results = run_pipeline('retrieval augmented generation', max_results=3)
if results:
    print(f'SUCCESS: {len(results[\"gaps\"])} gaps from {results[\"papers_analyzed\"]} papers')
else:
    print('FAILED: No results')
"`

Expected: Pipeline completes, gaps are generated with citation pointers.

- [ ] **Step 3: Verify Streamlit starts**

Run: `source .venv/bin/activate && streamlit run app.py --server.headless true` (kill after confirming it starts at localhost:8501)

- [ ] **Step 4: Final commit if any fixes were needed**

```bash
git add -A
git commit -m "Fix issues found during end-to-end smoke test"
```

---

## Spec Coverage Check

| Spec Requirement | Task |
|-----------------|------|
| Mistral 7B via Ollama for claims | Task 6 |
| SPECTER2 embeddings | Task 5 |
| Few-shot SciFact prompts | Task 6 |
| Fix NLI label order | Task 4 |
| Fix PDF whitespace | Task 2 |
| Fix section segmenter | Task 3 |
| Centralized config | Task 1 |
| Limitation retrieval | Task 9 |
| Semantic Scholar client | Task 8 |
| Citation tracing | Tasks 6, 7, 12 |
| SciFact evaluation | Task 14 |
| Unify main.py and app.py | Tasks 12, 13 |
| Remove sentence cap | Task 12 |
| Remove rule-based fallback | Task 7 |
| Update clustering to use config | Task 10 |
| Update ArXiv client to use config | Task 11 |
| Update requirements.txt | Task 1 |
