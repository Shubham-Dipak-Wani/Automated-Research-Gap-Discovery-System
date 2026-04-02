# Automated Research Gap Discovery System

An end-to-end NLP pipeline that identifies research gaps by analyzing academic papers from ArXiv. The system extracts scientific claims, clusters them semantically, detects contradictions via NLI, retrieves recurring limitations, and synthesizes grounded research gap statements with citation pointers.

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
- [Evaluation](#evaluation)
- [Models Used](#models-used)
- [Configuration](#configuration)
- [Contributors](#contributors)

---

## Overview

Keeping up with the flood of academic publications is a growing challenge for researchers. This system automates the process of:

1. **Fetching** papers from ArXiv and Semantic Scholar on a given topic
2. **Extracting** atomic scientific claims from each paper using Mistral 7B
3. **Clustering** semantically similar claims using SPECTER2 embeddings + HDBSCAN
4. **Detecting contradictions** between claims via DeBERTa-v3 NLI
5. **Retrieving limitations** via seed query similarity search
6. **Generating** grounded research gap statements with citation pointers

The result is a structured report of what the literature agrees on, where it disagrees, what limitations recur, and what remains unexplored — with every gap traced back to its source papers.

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
         │   ArXiv API + Semantic Scholar          │
         │   → PDF Download + Citation Graph       │
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
         │   Mistral 7B (Ollama) + Few-Shot       │
         │   SciFact Prompts + Source Metadata     │
         └───────────────────┬───────────────────┘
                             │
         ┌───────────────────▼───────────────────┐
         │         Embedding & Clustering         │
         │   SPECTER2 (768-d) → HDBSCAN           │
         └───────────────────┬───────────────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
   ┌──────────▼──────────┐     ┌────────────▼───────────┐
   │  Contradiction       │     │  Limitation Retrieval   │
   │  Detection (NLI)     │     │  Seed Query Similarity  │
   │  DeBERTa-v3          │     │  → Re-clustering        │
   └──────────┬──────────┘     └────────────┬───────────┘
              │                             │
              └──────────────┬──────────────┘
                             │
         ┌───────────────────▼───────────────────┐
         │        Research Gap Generation         │
         │   Mistral 7B (Ollama) + Citations      │
         └───────────────────┬───────────────────┘
                             │
                        ▼ Results
          Clusters · Contradictions · Limitations
           · Gaps with Citation Pointers
```

---

## Pipeline Stages

### 1. Paper Ingestion (`ingestion/`)

- **ArxivClient** queries the ArXiv API with a keyword search, downloads PDFs to `data/raw_papers/`. Already-downloaded papers are cached.
- **SemanticScholarClient** enriches papers with metadata (citation counts, venue, year) and discovers related papers via citation graphs.

### 2. Text Extraction & Segmentation (`parsing/`)

- **PDFParser** extracts text using PyMuPDF, preserving paragraph structure and truncating at the References section.
- **SectionSegmenter** identifies paper sections (Abstract, Introduction, Methods, Results, etc.) using anchored regex patterns.
- **SentenceSplitter** tokenizes sections into sentences using spaCy.

### 3. Claim Extraction (`claims/`)

**ClaimExtractor** sends each sentence to **Mistral 7B** (via Ollama) with few-shot SciFact prompts to decompose it into atomic, independently verifiable claims. Each claim carries source metadata (paper title, section, source sentence) for citation tracing.

### 4. Embedding & Clustering (`embedding/`, `clustering/`)

- **SpecterEmbedder** encodes claims into 768-dimensional vectors using **SPECTER2**, a scientific embedding model trained on 6M citation triplets.
- **ClaimClusterer** groups related claims using HDBSCAN.

### 5. Contradiction Detection (`nli/`)

Within each cluster:
1. **Cosine similarity filter** selects pairs with similarity between 0.7 and 0.9
2. **DeBERTa-v3 NLI** classifies pairs as entailment, neutral, or contradiction
3. High-confidence contradictions are surfaced

### 6. Limitation Retrieval (`claims/limitation_retriever.py`)

Seed queries like "unresolved problem" and "our approach does not handle" are embedded with SPECTER2 and compared against all claim embeddings. Limitation-adjacent claims are re-clustered to identify recurring limitation themes.

### 7. Research Gap Generation (`gap_generation/`)

**GapGenerator** uses Mistral 7B to synthesize research gaps from each cluster, including:
- **Cluster gaps** — open questions from topic clusters
- **Contradiction gaps** — disagreements between papers
- **Limitation gaps** — recurring limitations across papers

Every gap includes citation pointers to source papers and supporting claims.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Paper Retrieval** | ArXiv API, Semantic Scholar API |
| **PDF Parsing** | PyMuPDF (fitz) |
| **NLP Processing** | spaCy (`en_core_web_sm`) |
| **Claim Extraction** | Mistral 7B via Ollama |
| **Embeddings** | SPECTER2 (sentence-transformers) |
| **Clustering** | HDBSCAN |
| **Contradiction Detection** | DeBERTa-v3-base-zeroshot (NLI) |
| **Gap Generation** | Mistral 7B via Ollama |
| **Deep Learning** | PyTorch |
| **Evaluation** | BERTScore, SciFact dataset |
| **Web Interface** | Streamlit |

---

## Project Structure

```
NLP-project/
│
├── ingestion/                     # Paper retrieval
│   ├── arxiv_client.py            #   ArXiv API search & PDF download
│   └── semantic_scholar_client.py #   Metadata enrichment & citation graph
│
├── parsing/                       # Text extraction
│   ├── pdf_parser.py              #   PDF → text, sentence splitting
│   └── section_segmenter.py       #   Section identification
│
├── claims/                        # Claim extraction & analysis
│   ├── claim_extractor.py         #   Mistral-based extraction + filtering
│   ├── prompts.py                 #   Few-shot SciFact prompt templates
│   └── limitation_retriever.py    #   Seed query limitation retrieval
│
├── embedding/                     # Vectorization
│   └── specter_embedder.py        #   SPECTER2 scientific embeddings
│
├── clustering/                    # Claim grouping
│   └── hdbscan_cluster.py         #   HDBSCAN clustering
│
├── nli/                           # Contradiction detection
│   └── nli_engine.py              #   DeBERTa-v3 NLI classifier
│
├── gap_generation/                # Gap synthesis
│   ├── generator.py               #   Mistral-based gap generation
│   └── prompts.py                 #   Gap generation prompt templates
│
├── evaluation/                    # Metrics & evaluation
│   ├── evaluator.py               #   SciFact evaluation runner
│   └── metrics.py                 #   F1, BERTScore computation
│
├── config/                        # Configuration
│   └── settings.py                #   All thresholds, model names, URLs
│
├── data/
│   └── raw_papers/                # Downloaded PDFs (git-ignored)
│
├── main.py                        # CLI entry point — runs full pipeline
├── app.py                         # Streamlit web interface
└── requirements.txt               # Python dependencies
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com/) (for running Mistral 7B locally)

### Step 1: Install Ollama and Pull Mistral

```bash
# macOS
brew install ollama

# Or download from https://ollama.com/ for other platforms

# Start the Ollama server
ollama serve

# In a new terminal, pull the Mistral model (~4.4 GB)
ollama pull mistral
```

> **Tip:** To store models on an external drive, set the `OLLAMA_MODELS` environment variable before starting the server:
> ```bash
> OLLAMA_MODELS="/path/to/your/drive/ollama/models" ollama serve
> ```

### Step 2: Set Up Python Environment

```bash
# Clone the repository
git clone https://github.com/jaanakidave/NLP-project.git
cd NLP-project

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download the spaCy language model
python -m spacy download en_core_web_sm
```

### Step 3: Verify Setup

```bash
# Check Ollama is running with Mistral
ollama list  # Should show mistral:latest

# Quick test
python -c "from config.settings import OLLAMA_URL; import requests; print(requests.get('http://localhost:11434/api/tags', proxies={'http': None}).json())"
```

> **Note:** On first run, SPECTER2 and DeBERTa-v3 models will be downloaded automatically from Hugging Face (~1.5 GB total). These are cached locally after the first download.

---

## Usage

### CLI Pipeline

```bash
python main.py
```

This will:
1. Download 50 papers on "retrieval augmented generation" from ArXiv
2. Enrich with Semantic Scholar metadata and discover related papers
3. Extract claims with source provenance
4. Embed, cluster, and detect contradictions
5. Retrieve limitation-adjacent claims
6. Generate research gaps with citation pointers
7. Save results to `data/results.json`

### Streamlit Web App

```bash
streamlit run app.py
```

The app lets you:
- Enter any research topic
- Configure the number of papers to analyze (3-100)
- View summary metrics (papers, claims, clusters, contradictions)
- Explore research gaps with source papers and supporting claims

---

## Evaluation

Run the SciFact evaluation suite:

```bash
python -m evaluation.evaluator
```

This evaluates:
- **Claim extraction quality** — Precision, Recall, F1 against SciFact ground-truth claims using BERTScore for semantic matching
- **NLI contradiction detection** — F1 score on SciFact REFUTE pairs

Results are saved to `data/evaluation_results.json`.

---

## Models Used

| Model | Purpose | Source |
|---|---|---|
| **Mistral 7B** (via Ollama) | Claim extraction & gap generation | [Ollama](https://ollama.com/library/mistral) |
| **allenai/specter2_base** | Scientific claim embeddings (768-d) | [Hugging Face](https://huggingface.co/allenai/specter2_base) |
| **MoritzLaurer/deberta-v3-base-zeroshot-v1** | NLI contradiction detection | [Hugging Face](https://huggingface.co/MoritzLaurer/deberta-v3-base-zeroshot-v1) |
| **en_core_web_sm** | Sentence tokenization | [spaCy](https://spacy.io/models/en#en_core_web_sm) |

---

## Configuration

All thresholds and model settings are centralized in `config/settings.py`:

| Setting | Default | Purpose |
|---|---|---|
| `ARXIV_MAX_RESULTS` | 50 | Papers to download from ArXiv |
| `OLLAMA_MODEL` | `"mistral"` | LLM for claim extraction & gap generation |
| `SPECTER_MODEL` | `"allenai/specter2_base"` | Embedding model |
| `COSINE_SIM_MIN` / `MAX` | 0.7 / 0.9 | Similarity window for NLI pair selection |
| `NLI_CONFIDENCE_THRESHOLD` | 0.7 | Minimum confidence for contradictions |
| `HDBSCAN_MIN_CLUSTER_SIZE` | 2 | Minimum claims per cluster |
| `LIMITATION_SIMILARITY_THRESHOLD` | 0.5 | Threshold for limitation retrieval |

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
