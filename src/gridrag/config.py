"""Central configuration: paths, models and retrieval settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
INDEX_DIR = ROOT / "data" / "index"

CHUNKS_FILE = PROCESSED_DIR / "chunks.jsonl"
EMBEDDINGS_FILE = INDEX_DIR / "embeddings.npy"
INDEX_CHUNKS_FILE = INDEX_DIR / "chunks.jsonl"

# Local models served by Ollama (no data leaves the machine)
EMBED_MODEL = "bge-m3"        # multilingual embedding model
LLM_MODEL = "qwen2.5:7b"      # instruction model; another LLM is this one setting
LLM_CONTEXT = 8192            # tokens the LLM can read (question + retrieved chunks + answer)
LLM_MAX_ANSWER = 800          # tokens per answer: without a limit a repetition loop never ends
LLM_MAX_REPORT = 1500         # tokens per category of the report: a JSON list is longer than an answer
LLM_REPEAT_PENALTY = 1.1      # above 1 discourages repeating text, also numbers copied from the excerpts

# Chunking
MAX_CHUNK_CHARS = 1800        # long articles are split into paragraphs up to this size

# Retrieval
TOP_K = 4                     # chunks passed to the LLM
RRF_K = 60                    # constant of reciprocal rank fusion (hybrid search)
