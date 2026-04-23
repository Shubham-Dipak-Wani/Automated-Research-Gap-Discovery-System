import os
from dotenv import load_dotenv

load_dotenv()

# --- Ingestion ---
ARXIV_MAX_RESULTS = 50
SEMANTIC_SCHOLAR_API_URL = "https://api.semanticscholar.org/graph/v1"
PAPER_SAVE_DIR = "data/raw_papers"

# --- Claude API Models ---
# Haiku: high-volume claim extraction (fast, cost-effective, good at extraction)
CLAUDE_CLAIM_MODEL = "claude-haiku-4-5"
# Opus: high-quality research gap synthesis (complex reasoning, low volume)
CLAUDE_GAP_MODEL = "claude-opus-4-7"

# --- Local Models (embeddings + NLI) ---
SPECTER_MODEL = "allenai/specter2_base"
NLI_MODEL = "MoritzLaurer/deberta-v3-base-zeroshot-v1"
SPACY_MODEL = "en_core_web_sm"

# --- Claim Extraction ---
MIN_SENTENCE_LENGTH = 40
MIN_CLAIM_LENGTH = 30
CLAIM_BATCH_SIZE = 10  # sentences per Claude API call

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
