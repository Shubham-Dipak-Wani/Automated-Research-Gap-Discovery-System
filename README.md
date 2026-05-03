# Automated Research Gap Discovery System

A course project for NLP that automatically finds what is missing in academic literature. You give it a research topic, it pulls papers from ArXiv, reads them, groups similar ideas together, checks for contradictions, and tells you what has not been studied yet.

This is v2. The original version used FLAN-T5 for claim extraction and gap generation, which produced mediocre output most of the time. This version replaces both with Claude API calls, adds Semantic Scholar citation graph integration, a LimitationRetriever, and a SciFact evaluation pipeline. The Streamlit UI also got a full overhaul with Plotly charts and a tabbed gap explorer.

---

## What Changed From v1

| Component | v1 | v2 (this version) |
|---|---|---|
| Claim extraction | FLAN-T5-small (local) | Claude Haiku 4.5 (API, batched, 65-example prompt) |
| Gap generation | FLAN-T5-base + rule-based fallback | Claude Opus 4.7 with adaptive thinking |
| Embeddings | SPECTER v1 | SPECTER2 (allenai/specter2_base) |
| Paper sources | ArXiv only | ArXiv + Semantic Scholar citation graph |
| NLI label reading | Hardcoded list | Read from model.config.id2label |
| Limitation detection | Not present | LimitationRetriever with seed query similarity |
| Evaluation | Empty folder | SciFact evaluation with BERTScore |
| Settings | Hardcoded throughout | Centralized in config/settings.py |
| UI | Basic Streamlit text | Plotly charts, tabbed gaps, paper explorer, live progress |

---

## Pipeline Overview

```
ArXiv API
    +
Semantic Scholar  →  PDF Parser (PyMuPDF)  →  Section Segmenter  →  spaCy sentence split
                                                                              ↓
                                              Claude Haiku 4.5  (batched, 65-example prompt)
                                                                              ↓
                                                     SPECTER2 embeddings  (768-dim)
                                                                              ↓
                                                          HDBSCAN clustering
                                                          /                   \
                                          DeBERTa NLI                LimitationRetriever
                                        (contradiction detection)   (8 seed query cosine sim)
                                                          \                   /
                                              Claude Opus 4.7  (gap synthesis, adaptive thinking)
                                                                              ↓
                                                     Streamlit dashboard  +  JSON export
```

---

## Project Structure

```
NLP-project/
│
├── config/
│   ├── settings.py              # all constants: models, thresholds, paths
│   └── logger.py
│
├── ingestion/
│   ├── arxiv_client.py          # ArXiv API with 5-retry backoff on 429
│   └── semantic_scholar_client.py  # metadata + citation graph expansion
│
├── parsing/
│   ├── pdf_parser.py            # PyMuPDF extraction, references cutoff, whitespace clean
│   └── section_segmenter.py    # regex section detection (abstract, intro, methods, etc.)
│
├── claims/
│   ├── claim_extractor.py       # Claude Haiku 4.5 with 65-example cached system prompt
│   ├── limitation_retriever.py  # seed-query cosine similarity to find limitation claims
│   └── prompts.py               # single-sentence fallback prompt (legacy)
│
├── embedding/
│   ├── specter_embedder.py      # SPECTER2 via sentence-transformers
│   └── vector_store.py          # stubbed
│
├── clustering/
│   ├── hdbscan_cluster.py       # HDBSCAN with config-driven params
│   └── utils.py
│
├── nli/
│   ├── nli_engine.py            # DeBERTa-v3, reads labels from model.config.id2label
│   └── pair_selector.py
│
├── gap_generation/
│   ├── generator.py             # Claude Opus 4.7, adaptive thinking, cached system prompt
│   └── prompts.py               # GAP_GENERATION_PROMPT, CONTRADICTION_GAP_PROMPT
│
├── evaluation/
│   ├── evaluator.py             # SciFact claim extraction + NLI evaluation
│   └── metrics.py               # BERTScore claim F1, NLI contradiction F1 via sklearn
│
├── data/
│   └── raw_papers/              # downloaded PDFs (auto-created on first run)
│
├── main.py                      # CLI pipeline, saves to data/results.json
├── app.py                       # full Streamlit UI with Plotly charts
├── api.py                       # stubbed REST API
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/JaanakiDave11/NLP-project.git
cd NLP-project

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=your_key_here
```

The key is loaded automatically via `python-dotenv` in `config/settings.py`. No other configuration is required to run the main pipeline.

---

## Running It

**Web UI (recommended)**
```bash
streamlit run app.py
```
Open `http://localhost:8501`. Enter a research topic in the search bar, choose how many papers to analyze, and click Run Analysis. A live progress bar in the sidebar shows each stage as it completes.

**CLI**
```bash
python main.py
```
Runs the pipeline on `"retrieval augmented generation"` with up to 50 papers (set in `config/settings.py`) and saves results to `data/results.json`.

**Evaluation against SciFact**
```bash
python evaluation/evaluator.py
```
Saves results to `data/evaluation_results.json`.

---

## How Each Stage Works

### 1. Paper Collection

`ArxivClient.search_and_download()` sends a GET to the ArXiv API with the search term. The response is XML. It parses out the title and PDF URL for each result, downloads the PDF to `data/raw_papers/`, and skips already-downloaded files. If ArXiv returns a 429, it backs off and retries up to 5 times.

`SemanticScholarClient.enrich_and_expand()` then does two things. It calls the Semantic Scholar API by paper title to get citation count, year, and venue metadata. It also fetches each paper's reference list and downloads any referenced papers that have ArXiv IDs, up to a configured maximum. Starting with 5 papers from ArXiv can result in analyzing 10 or more after citation graph expansion.

### 2. PDF Parsing

`PDFParser.extract_text()` uses PyMuPDF (`fitz`) to extract raw text page by page. The `_clean_text()` method normalizes whitespace and splits off everything after the word "References" so citations do not pollute claim extraction.

`SentenceSplitter` uses spaCy (`en_core_web_sm`) to split the cleaned text into proper sentences. It handles abbreviations like `et al.`, `Fig.`, and `Dr.` correctly. Anything under 20 characters is discarded. Sentences under `MIN_SENTENCE_LENGTH` (40 chars, from settings) are also dropped before being sent to extraction.

### 3. Section Segmentation

`SectionSegmenter.segment()` uses regex to find known section headers (abstract, introduction, related work, method, results, discussion, conclusion, limitations) and slices the text between them. This means the claim dict knows which part of the paper a sentence came from, which GapGenerator includes in its prompt.

### 4. Claim Extraction (Claude Haiku 4.5)

This is the biggest change from v1. `ClaimExtractor.extract_all()` groups sentences into batches of 10 (`CLAIM_BATCH_SIZE` from settings) and makes one API call per batch.

The system prompt is 65 worked examples long, covering every edge case: compound claims that need splitting, metadata sentences that return NONE, sentences with pronouns, ablation results, limitation statements, funding acknowledgments, table captions, and section navigation text. This prompt is passed with `cache_control={"type": "ephemeral"}` at the top level so Anthropic caches it server-side across calls, which significantly reduces latency and cost when processing many papers.

The response uses `SENTENCE N:` markers so `_parse_batch_output()` can map each extracted claim back to the correct sentence, paper title, and section.

Each extracted claim becomes a dict:
```python
{
    "text": "RAG improves factual accuracy by 18% compared to standard generation.",
    "paper_title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP",
    "section": "abstract",
    "source_sentence": "Our experiments show RAG improves factual accuracy by 18%..."
}
```

### 5. SPECTER2 Embeddings

`SpecterEmbedder.encode()` loads `allenai/specter2_base` via sentence-transformers and encodes all claims into 768-dimensional vectors with batch size 8. The model handles both dict claims and raw strings so it works the same way for the main pipeline and the `LimitationRetriever`. SPECTER2 was trained on scientific citation pairs, so it understands that "RAG" and "retrieval-augmented generation" refer to the same thing.

### 6. HDBSCAN Clustering

`ClaimClusterer.cluster()` runs HDBSCAN on the embedding matrix. Parameters come from settings: `min_cluster_size=2`, `min_samples=1`, `cluster_selection_epsilon=0.8`. Claims with label `-1` (noise) are dropped by `group_clusters()`. If no clusters form, all claims fall into a single fallback group so the pipeline continues rather than crashing.

### 7. Contradiction Detection (DeBERTa NLI)

For each cluster, `NLIEngine.predict()` is called on pairs of claims that pass a cosine similarity pre-filter. Only pairs scoring between `COSINE_SIM_MIN` (0.7) and `COSINE_SIM_MAX` (0.9) are tested — below 0.7 means the claims are probably unrelated, above 0.9 means they are near-duplicates. The DeBERTa model reads its label order from `model.config.id2label` instead of a hardcoded list, so swapping to a different NLI checkpoint works without code changes.

Pairs where the predicted label is `"contradiction"` and confidence exceeds `NLI_CONFIDENCE_THRESHOLD` (0.7) are collected for the contradiction gap.

### 8. Limitation Retrieval

`LimitationRetriever.retrieve()` embeds 8 seed queries from settings:

```python
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
```

It computes cosine similarity between every claim embedding and every seed embedding. Claims scoring above `LIMITATION_SIMILARITY_THRESHOLD` (0.5) against any seed are flagged as limitation-adjacent. These are re-clustered with HDBSCAN so recurring limitation themes group together, then passed to GapGenerator separately to produce limitation-type gaps.

### 9. Gap Generation (Claude Opus 4.7)

`GapGenerator.generate_gap()` handles three gap types. For topic cluster gaps and limitation gaps, `_generate_cluster_gap()` filters to claims over 50 characters, caps at 5 claims, and builds a prompt with paper names and section labels included. For contradiction gaps, `_generate_contradiction_gap()` formats the conflicting claim pairs with their confidence scores.

All three paths call `_call_claude()`, which uses `claude-opus-4-7` with `thinking={"type": "adaptive"}`. Adaptive thinking lets the model reason internally before writing the final gap statement, which produces more grounded and specific output. The system prompt is cached server-side via a `cache_control` block in the system messages list.

The code extracts only the `text` block from the response (skipping thinking blocks), then validates the output starts with `"Research Gap:"` and adds the prefix if Claude forgot it.

### 10. Evaluation (SciFact)

`SciFactEvaluator` runs two evaluations. Claim extraction F1 compares extracted claims against SciFact ground truth using BERTScore with `rescale_with_baseline=True`, so valid paraphrases of the same fact count as correct rather than penalizing different wording. NLI F1 runs the engine on SciFact claim-abstract pairs and reports macro F1 plus contradiction-specific precision, recall, and F1 via sklearn.

---

## Key Design Decisions

| Decision | Reason |
|---|---|
| 10 sentences per Claude call | Reduces API calls by 10x. Cached system prompt makes per-batch overhead near zero. |
| 65-example system prompt | Large example set handles edge cases reliably: compound claims, metadata, pronouns, table captions. |
| Adaptive thinking for Opus | Gap synthesis requires multi-step reasoning. Thinking mode produces more specific, grounded gaps. |
| Seed similarity for limitations | "Remains challenging" and "future work" catch implicit limitations the word "limitation" misses. |
| Labels from model.config.id2label | Decouples code from a specific NLI checkpoint's label ordering. |
| SPECTER2 over general embeddings | Trained on scientific citation pairs. Understands that "RAG" and "retrieval-augmented generation" are the same thing. |
| HDBSCAN over K-means | Does not require specifying K upfront. Labels noise as -1 and discards it cleanly. |
| Cosine similarity over Euclidean | Measures direction (meaning), not magnitude (sentence length). Standard for text embedding comparison. |
| BERTScore for claim F1 | Exact string match punishes valid paraphrases. Semantic similarity scoring is fairer. |
| References cutoff in PDF parser | Removes citations before extraction so bibliography entries do not pollute the claim list. |

---

## Configuration

All constants in `config/settings.py`:

```python
# Models
CLAUDE_CLAIM_MODEL = "claude-haiku-4-5"
CLAUDE_GAP_MODEL   = "claude-opus-4-7"
SPECTER_MODEL      = "allenai/specter2_base"
NLI_MODEL          = "MoritzLaurer/deberta-v3-base-zeroshot-v1"
SPACY_MODEL        = "en_core_web_sm"

# Ingestion
ARXIV_MAX_RESULTS = 50

# Claim extraction
MIN_SENTENCE_LENGTH = 40   # chars — shorter sentences are dropped before extraction
MIN_CLAIM_LENGTH    = 30   # chars — shorter extracted claims are discarded
CLAIM_BATCH_SIZE    = 10   # sentences per Claude Haiku API call

# Clustering
HDBSCAN_MIN_CLUSTER_SIZE = 2
HDBSCAN_MIN_SAMPLES      = 1
HDBSCAN_EPSILON          = 0.8

# NLI
COSINE_SIM_MIN           = 0.7   # lower bound of pre-filter window
COSINE_SIM_MAX           = 0.9   # upper bound — pairs above this are near-duplicates
NLI_CONFIDENCE_THRESHOLD = 0.7

# Limitation retrieval
LIMITATION_SIMILARITY_THRESHOLD = 0.5

# Gap generation
MIN_CLAIM_LENGTH_FOR_GAP = 50   # shorter claims dropped before sending to Opus
MAX_CLAIMS_PER_GAP       = 5    # max claims per Claude Opus prompt
```

---

## Output Format

Results are saved as JSON and exported via the Streamlit UI:

```json
{
  "query": "retrieval augmented generation",
  "papers_analyzed": 12,
  "total_claims": 847,
  "clusters": 43,
  "contradictions": 3,
  "gaps": [
    {
      "gap": "Research Gap: While Lewis et al. demonstrate RAG improves factual accuracy...",
      "supporting_claims": ["RAG improves factual accuracy.", "Dense retrieval outperforms BM25."],
      "source_papers": ["Lewis 2020", "Karpukhin 2020"],
      "type": "open_question"
    },
    {
      "gap": "Research Gap: Papers A and B directly contradict each other on...",
      "supporting_claims": ["...", "..."],
      "source_papers": ["..."],
      "type": "contradiction"
    },
    {
      "gap": "Research Gap: The described approach does not address...",
      "supporting_claims": ["..."],
      "source_papers": ["..."],
      "type": "limitation"
    }
  ]
}
```

---

## Known Limitations

- `vector_store.py` and `pair_selector.py` are stubbed. Embeddings are recomputed from scratch every run — no caching.
- `api.py` is empty. The pipeline is only accessible via CLI or Streamlit.
- Section segmenter uses regex. Non-standard headers like "Our Approach" or "Technical Details" are not detected.
- Semantic Scholar rate limits at 100 requests per minute without an API key. The 0.5s sleep between papers usually keeps the pipeline under this, but very large runs can still hit it.
- `bert-score`, `datasets`, and `peft` are only needed for `evaluation/evaluator.py`. You can skip installing them if you are not running the evaluation.
- The NLI index mapping between cluster embeddings and cluster claims can go out of sync if HDBSCAN skips a cluster ID (e.g., assigns labels 0, 1, 3 skipping 2). This is a known edge case that does not affect normal runs but could produce wrong NLI pairs on unusual inputs.

---

## Authors

Built as part of an NLP course project. Updated by Shubham with Claude API migration, Semantic Scholar citation graph integration, LimitationRetriever, SciFact evaluation pipeline, and full Streamlit UI overhaul.
