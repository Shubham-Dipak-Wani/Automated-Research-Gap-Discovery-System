# Automated Research Gap Discovery System

An end-to-end NLP pipeline that identifies research gaps by analyzing academic papers from ArXiv. The system extracts scientific claims, detects contradictions between them, and synthesizes unexplored research directions — helping researchers quickly find where the literature falls short.

Built as a course project at the **University of Southern California (USC)**.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Pipeline Stages](#pipeline-stages)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Models Used](#models-used)
- [Contributors](#contributors)

---

## Overview

Keeping up with the flood of academic publications is a growing challenge for researchers. This system automates the process of:

1. **Fetching** recent papers from ArXiv on a given topic
2. **Extracting** atomic scientific claims from each paper
3. **Clustering** semantically similar claims using dense embeddings
4. **Detecting contradictions** between claims via Natural Language Inference
5. **Generating** concrete research gap statements from each cluster

The result is a structured summary of what the literature agrees on, where it disagrees, and what remains unexplored.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Interface                              │
│              Streamlit Web App  ·  CLI Pipeline                      │
└────────────────────────────┬────────────────────────────────────────┘
                             │
         ┌───────────────────▼───────────────────┐
         │          Paper Ingestion               │
         │   ArXiv API → PDF Download             │
         └───────────────────┬───────────────────┘
                             │
         ┌───────────────────▼───────────────────┐
         │          Text Extraction               │
         │   PyMuPDF → Section Segmentation       │
         │   → spaCy Sentence Splitting           │
         └───────────────────┬───────────────────┘
                             │
         ┌───────────────────▼───────────────────┐
         │         Claim Extraction               │
         │   FLAN-T5-small + Prompt Engineering   │
         └───────────────────┬───────────────────┘
                             │
         ┌───────────────────▼───────────────────┐
         │         Embedding & Clustering         │
         │   SPECTER (768-d) → HDBSCAN            │
         └───────────────────┬───────────────────┘
                             │
         ┌───────────────────▼───────────────────┐
         │      Contradiction Detection           │
         │   Cosine Similarity Filter →            │
         │   DeBERTa-v3 NLI Classification        │
         └───────────────────┬───────────────────┘
                             │
         ┌───────────────────▼───────────────────┐
         │        Research Gap Generation         │
         │   FLAN-T5-base + Rule-based Fallback   │
         └───────────────────┬───────────────────┘
                             │
                        ▼ Results
              Clusters · Contradictions · Gaps
```

---

## Pipeline Stages

### 1. Paper Ingestion (`ingestion/`)

The **ArxivClient** queries the [ArXiv API](https://arxiv.org/help/api) with a keyword search, parses the XML response, and downloads PDFs to `data/raw_papers/`. Already-downloaded papers are skipped to avoid redundant fetches.

### 2. Text Extraction & Segmentation (`parsing/`)

- **PDFParser** uses [PyMuPDF](https://pymupdf.readthedocs.io/) to extract raw text from each PDF. The text is cleaned (whitespace normalization) and truncated at the "References" section.
- **SectionSegmenter** identifies standard paper sections (Abstract, Introduction, Methods, Results, Discussion, Conclusion) by regex matching, then slices the text accordingly.
- **SentenceSplitter** uses [spaCy](https://spacy.io/) (`en_core_web_sm`) to tokenize each section into individual sentences, filtering out fragments shorter than 20 characters.

### 3. Claim Extraction (`claims/`)

The **ClaimExtractor** feeds each sentence into **FLAN-T5-small** with a prompt asking for atomic scientific claims. The output passes through a multi-stage filter that removes:

- Metadata artifacts (DOIs, author names, URLs, copyright notices)
- Short fragments (< 30 characters)
- Quoted text and page references

Each claim is normalized to end with a period.

### 4. Embedding (`embedding/`)

The **SpecterEmbedder** encodes every claim into a 768-dimensional vector using [SPECTER](https://github.com/allenai/specter) — a model specifically trained on scientific paper embeddings. Batch processing (size 8) keeps encoding efficient.

### 5. Clustering (`clustering/`)

The **ClaimClusterer** groups semantically related claims using [HDBSCAN](https://hdbscan.readthedocs.io/) with relaxed parameters (`min_cluster_size=2`, `cluster_selection_epsilon=0.8`) to form meaningful clusters even from small claim sets. Noise points (label `-1`) are filtered out.

### 6. Contradiction Detection (`nli/`)

Within each cluster, the system identifies potential contradictions:

1. **Cosine similarity filter** — only pairs with similarity between 0.7 and 0.9 are considered (related but not identical)
2. **NLI classification** — [DeBERTa-v3](https://huggingface.co/MoritzLaurer/deberta-v3-base-zeroshot-v1) classifies each pair as *entailment*, *neutral*, or *contradiction*
3. Contradictions with confidence > 0.7 are surfaced

### 7. Research Gap Generation (`gap_generation/`)

The **GapGenerator** takes each cluster's claims and prompts **FLAN-T5-base** to identify a missing research problem. A hybrid strategy is used:

- **ML-first**: the model attempts to generate a novel gap statement
- **Rule-based fallback**: if the output is too short or doesn't match the expected format, domain-specific heuristics generate a gap (covering retrieval/RAG, LLM cognition, and misinformation domains)

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Paper Retrieval** | ArXiv API, Requests |
| **PDF Parsing** | PyMuPDF (fitz) |
| **NLP Processing** | spaCy (`en_core_web_sm`) |
| **Claim Extraction** | FLAN-T5-small (Hugging Face Transformers) |
| **Embeddings** | SPECTER (sentence-transformers) |
| **Clustering** | HDBSCAN |
| **Contradiction Detection** | DeBERTa-v3-base-zeroshot (NLI) |
| **Gap Generation** | FLAN-T5-base (Hugging Face Transformers) |
| **Deep Learning** | PyTorch |
| **Web Interface** | Streamlit |

---

## Project Structure

```
NLP-project/
│
├── ingestion/                     # Paper retrieval
│   ├── arxiv_client.py            #   ArXiv API search & PDF download
│   └── semantic_scholar_client.py #   Semantic Scholar client (planned)
│
├── parsing/                       # Text extraction
│   ├── pdf_parser.py              #   PDF → text, sentence splitting
│   └── section_segmenter.py       #   Section identification (intro, methods, etc.)
│
├── claims/                        # Claim extraction
│   ├── claim_extractor.py         #   FLAN-T5 based extraction + filtering
│   └── prompts.py                 #   Prompt templates
│
├── embedding/                     # Vectorization
│   ├── specter_embedder.py        #   SPECTER scientific embeddings
│   └── vector_store.py            #   Vector persistence (planned)
│
├── clustering/                    # Claim grouping
│   ├── hdbscan_cluster.py         #   HDBSCAN clustering
│   └── utils.py                   #   Clustering utilities
│
├── nli/                           # Contradiction detection
│   ├── nli_engine.py              #   DeBERTa-v3 NLI classifier
│   └── pair_selector.py           #   Pair selection logic (planned)
│
├── gap_generation/                # Gap synthesis
│   ├── generator.py               #   FLAN-T5-base generator + rule fallback
│   └── prompts.py                 #   Generation prompt templates
│
├── evaluation/                    # Metrics & evaluation (planned)
│   ├── evaluator.py
│   └── metrics.py
│
├── config/                        # Configuration (planned)
│   ├── settings.py
│   └── logger.py
│
├── tests/                         # Test suite
│   ├── test_parsing.py
│   ├── test_ingestion.py
│   └── test_claims.py
│
├── data/
│   └── raw_papers/                # Downloaded PDFs (git-ignored)
│
├── main.py                        # CLI entry point — runs full pipeline
├── app.py                         # Streamlit web interface
├── api.py                         # REST API endpoint (planned)
├── requirements.txt               # Python dependencies
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/jaanakidave/NLP-project.git
cd NLP-project

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download the spaCy language model
python -m spacy download en_core_web_sm
```

> **Note:** On first run, the pipeline will automatically download the required Hugging Face models (~2 GB total): FLAN-T5-small, FLAN-T5-base, SPECTER, and DeBERTa-v3. These are cached locally after the first download.

---

## Usage

### CLI Pipeline

Run the full pipeline from the command line:

```bash
python main.py
```

This will:
1. Download 3 papers on "retrieval augmented generation" from ArXiv
2. Extract and cluster claims
3. Detect contradictions
4. Generate research gaps
5. Print all results to the terminal

### Streamlit Web App

Launch the interactive web interface:

```bash
streamlit run app.py
```

The app lets you:
- Enter any research topic
- View extracted claims, clusters, and generated research gaps
- Explore results interactively with expandable sections

---

## Models Used

| Model | Purpose | Source |
|---|---|---|
| **google/flan-t5-small** | Claim extraction from sentences | [Hugging Face](https://huggingface.co/google/flan-t5-small) |
| **google/flan-t5-base** | Research gap generation | [Hugging Face](https://huggingface.co/google/flan-t5-base) |
| **allenai/specter** | Scientific claim embeddings (768-d) | [Hugging Face](https://huggingface.co/allenai/specter) |
| **MoritzLaurer/deberta-v3-base-zeroshot-v1** | NLI contradiction detection | [Hugging Face](https://huggingface.co/MoritzLaurer/deberta-v3-base-zeroshot-v1) |
| **en_core_web_sm** | Sentence tokenization | [spaCy](https://spacy.io/models/en#en_core_web_sm) |

---

## Contributors

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/jaanakidave">
        <img src="https://github.com/jaanakidave.png" width="100px;" alt="Jaanaki Dave"/><br />
        <sub><b>Jaanaki Dave</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/hirak214">
        <img src="https://github.com/hirak214.png" width="100px;" alt="Hirak Desai"/><br />
        <sub><b>Hirak Desai</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/ShreeyaSHalwasia">
        <img src="https://github.com/ShreeyaSHalwasia.png" width="100px;" alt="Shreeya S Halwasia"/><br />
        <sub><b>Shreeya S Halwasia</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/ryaverma">
        <img src="https://github.com/ryaverma.png" width="100px;" alt="Rya Verma"/><br />
        <sub><b>Rya Verma</b></sub>
      </a>
    </td>
  </tr>
</table>

---

## License

This project was developed for academic purposes at the University of Southern California.
