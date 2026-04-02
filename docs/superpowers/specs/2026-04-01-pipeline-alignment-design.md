# Design Spec: Pipeline Alignment with Pre-Proposal

**Date:** 2026-04-01
**Goal:** Align the codebase with the pre-proposal for masters project submission. Fix bugs, upgrade models, implement missing features.

---

## 1. Model Upgrades

### Claim Extraction
- **Current:** `google/flan-t5-small` with inline zero-shot prompt
- **Target:** Mistral 7B via Ollama with few-shot SciFact prompts
- **File:** `claims/claim_extractor.py`
- **Prompt source:** Expand `claims/prompts.py` with SciFact examples, import it into extractor
- **Ollama integration:** HTTP calls to `localhost:11434/api/generate`

### Embeddings
- **Current:** `allenai/specter` (v1)
- **Target:** `allenai/specter2`
- **File:** `embedding/specter_embedder.py`
- **Change:** Swap model name, verify output dimension still 768-d

### NLI (Keep Model, Fix Bug)
- **Model:** `MoritzLaurer/deberta-v3-base-zeroshot-v1` (unchanged)
- **File:** `nli/nli_engine.py`
- **Fix:** Read label order from `model.config.id2label` at runtime instead of hardcoding `["contradiction", "neutral", "entailment"]`

### Gap Generation
- **Current:** `google/flan-t5-base` with 60-token limit and rule-based fallback
- **Target:** Mistral 7B via Ollama (reuse same Ollama instance as claim extraction)
- **File:** `gap_generation/generator.py`
- **Change:** Remove rule-based fallback entirely. LLM generates full gap statements with citation pointers.

---

## 2. Pipeline Bug Fixes

### PDF Whitespace Destruction
- **File:** `parsing/pdf_parser.py:25`
- **Current:** `re.sub(r'\s+', ' ', text)` destroys all structure
- **Fix:** Normalize runs of spaces/tabs within lines, but preserve `\n\n` as paragraph breaks

### Section Segmenter Fragility
- **File:** `parsing/section_segmenter.py`
- **Current:** First regex match in entire text, can hit body text
- **Fix:** Anchor patterns to line starts: `r'^\s*\d*\.?\s*introduction\s*$'` (case-insensitive, multiline)

### Unused Better Prompt
- **File:** `claims/prompts.py` (exists, never imported)
- **Fix:** Import into `claim_extractor.py`, expand with SciFact few-shot examples for Mistral

### app.py Skips NLI
- **Current:** `app.py` runs claims → clusters → gaps, skipping contradiction detection
- **Fix:** Both `main.py` and `app.py` run the full pipeline including NLI and limitation retrieval

### Hardcoded Paper Count
- **Current:** `max_results=3` in both entry points
- **Fix:** Default to 50 in `config/settings.py`, configurable via Streamlit slider in `app.py`

### Arbitrary Sentence Cap
- **Current:** `[:10]` limits to first 10 sentences per section
- **Fix:** Remove the cap. Process all sentences.

### Gap Generator Fallback
- **Current:** If FLAN-T5 output is too short, returns hardcoded strings based on keyword matching
- **Fix:** Remove fallback entirely. Mistral 7B produces adequate output without it.

---

## 3. Missing Features

### 3a. Limitation Retrieval
- **New file:** `claims/limitation_retriever.py`
- **Proposal quote:** "Semantic retrieval over the embedding space using seed queries ('unresolved problem', 'our approach does not handle') retrieves limitation-adjacent claims, which are re-clustered to identify recurring themes."
- **Implementation:**
  - Define seed queries in `config/settings.py`: `["unresolved problem", "our approach does not handle", "limitation of", "future work", "remains challenging", "open question"]`
  - Embed seed queries with SPECTER2
  - Cosine similarity against all claim embeddings
  - Threshold (configurable, default 0.5) to select limitation-adjacent claims
  - Re-cluster selected claims with HDBSCAN
  - Return limitation clusters to gap generator

### 3b. Semantic Scholar Client
- **File:** `ingestion/semantic_scholar_client.py` (currently empty stub)
- **Proposal quote:** "Semantic Scholar's public API will supplement the corpus with structured metadata and citation graphs."
- **Implementation:**
  - Use Semantic Scholar public API (no key needed for <100 req/sec)
  - For each ArXiv paper, fetch: citation count, abstract, references, influential citations
  - Use citation graph to discover related papers not in ArXiv results
  - Return same contract as ArxivClient: `[{"title": str, "pdf_path": str}]`
  - Add metadata fields: `citation_count`, `year`, `venue`

### 3c. Citation Tracing / Source Provenance
- **Proposal quote:** Final report has "citation pointers" so gaps are "verifiable rather than hallucinated."
- **Implementation:**
  - Every claim carries metadata: `{"text": str, "paper_title": str, "section": str, "source_sentence": str}`
  - Thread this through embedding, clustering, NLI, and gap generation
  - Gap generator prompt includes source papers for each claim in the cluster
  - Final output format per gap: `{"gap": str, "supporting_claims": [...], "source_papers": [...], "type": "contradiction" | "limitation" | "open_question"}`

### 3d. Evaluation Module
- **Files:** `evaluation/evaluator.py`, `evaluation/metrics.py` (currently empty stubs)
- **Proposal evaluation plan:**

**Extraction Accuracy:**
- Download SciFact dataset (1.4K claims + abstracts)
- Run claim extractor on SciFact abstracts
- Compute precision, recall, F1 at claim level
- Use BERTScore for semantic similarity matching (accounts for paraphrasing)

**Contradiction Detection:**
- Use SciFact REFUTE pairs as ground truth
- Run NLI engine on these pairs
- Compute F1 score

**Research Gap Quality:**
- Scaffold a Likert scale evaluation form (1-5: novelty, groundedness, actionability)
- This is human evaluation — we provide the framework, humans fill it in

**Baseline Comparison:**
- Naive baseline: prompt Mistral with "find research gaps in these papers" + raw text
- Compare against structured pipeline output
- Metric: human evaluation on same Likert scale

---

## 4. Configuration System

- **File:** `config/settings.py` (currently empty stub)
- **All thresholds centralized:**
  ```python
  ARXIV_MAX_RESULTS = 50
  OLLAMA_MODEL = "mistral"
  OLLAMA_URL = "http://localhost:11434/api/generate"
  SPECTER_MODEL = "allenai/specter2"
  NLI_MODEL = "MoritzLaurer/deberta-v3-base-zeroshot-v1"
  COSINE_SIM_MIN = 0.7
  COSINE_SIM_MAX = 0.9
  NLI_CONFIDENCE_THRESHOLD = 0.7
  HDBSCAN_MIN_CLUSTER_SIZE = 2
  HDBSCAN_MIN_SAMPLES = 1
  HDBSCAN_EPSILON = 0.8
  MIN_SENTENCE_LENGTH = 40
  MIN_CLAIM_LENGTH = 30
  LIMITATION_SIMILARITY_THRESHOLD = 0.5
  SEED_QUERIES = ["unresolved problem", "our approach does not handle", ...]
  ```

---

## 5. Unified Pipeline

Both `main.py` and `app.py` call a shared pipeline. The flow:

```
Input: topic string, max_papers (default 50)
  1. ArxivClient.search_and_download(topic, max_papers)
  2. SemanticScholarClient.enrich_and_expand(papers)
  3. For each paper:
     a. PDFParser.extract_text(pdf_path)
     b. SectionSegmenter.segment(text)
     c. SentenceSplitter.split(section_text)
     d. ClaimExtractor.extract(sentences) → claims with metadata
  4. SpecterEmbedder.encode(all_claims)
  5. ClaimClusterer.cluster(embeddings) → topic clusters
  6. NLIEngine: pairwise within clusters → contradictions
  7. LimitationRetriever: seed query search → limitation clusters
  8. GapGenerator: per cluster, with citation pointers → grounded report
Output: {clusters, contradictions, limitations, gaps_with_citations}
```

`main.py`: prints to stdout
`app.py`: renders in Streamlit with expandable sections

---

## 6. Files Changed / Created

| File | Action |
|------|--------|
| `config/settings.py` | Implement (all config) |
| `claims/claim_extractor.py` | Rewrite for Ollama + few-shot |
| `claims/prompts.py` | Expand with SciFact few-shot examples |
| `claims/limitation_retriever.py` | **New** |
| `embedding/specter_embedder.py` | Swap to SPECTER2 |
| `nli/nli_engine.py` | Fix label order bug |
| `gap_generation/generator.py` | Rewrite for Ollama + citations |
| `gap_generation/prompts.py` | Implement (gap generation prompts) |
| `ingestion/semantic_scholar_client.py` | Implement |
| `parsing/pdf_parser.py` | Fix whitespace handling |
| `parsing/section_segmenter.py` | Fix regex anchoring |
| `evaluation/evaluator.py` | Implement (SciFact eval) |
| `evaluation/metrics.py` | Implement (F1, BERTScore) |
| `main.py` | Refactor to use shared pipeline + config |
| `app.py` | Refactor to use shared pipeline + config, add NLI |
| `requirements.txt` | Add ollama, bert-score, datasets |

**Not touched:** `tests/`, `api.py`, `config/logger.py`, `embedding/vector_store.py`, `nli/pair_selector.py` — these remain stubs; they're not in the proposal scope.
