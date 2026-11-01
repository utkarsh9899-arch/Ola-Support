"""Central configuration for the Ola support agent.

Import this module FIRST (before crewai, chromadb, sentence_transformers) so the
offline / no-telemetry environment variables are in place before those libraries
read them.
"""
import os
from pathlib import Path

# --- Zero-network guarantees -------------------------------------------------
# CrewAI telemetry would otherwise make an outbound call on crew.kickoff().
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
# CrewAI's separate trace collection (opt-in, but forced off explicitly).
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
# ChromaDB anonymized telemetry.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
# LangChain / LangSmith tracing stays off.
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ.setdefault("LANGSMITH_TRACING", "false")
# Tokenizers fork warning noise.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# --- LLM mode ----------------------------------------------------------------
# MOCK_LLM=1 (default) -> deterministic mock LLMs, zero API keys, zero network.
MOCK_LLM: bool = os.environ.get("MOCK_LLM", "1").strip().lower() not in ("0", "false", "no")

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
KB_DIR = ROOT / "kb"
CHROMA_DIR = ROOT / "chroma_db"
LOG_DIR = ROOT / "logs"
TRANSCRIPT_DIR = ROOT / "transcripts"

# --- Embeddings / retrieval --------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_FIXED = "ola_kb_fixed"
COLLECTION_SENTENCE = "ola_kb_sentence"
TOP_K = 3

# Filled in after empirical calibration (Task 4) and chunking evaluation (Task 5).
SIMILARITY_THRESHOLD: float | None = None
DEPLOYED_COLLECTION: str | None = None

# --- Runtime governance (Task 15) --------------------------------------------
MAX_TOKENS_PER_REQUEST = 1500
COST_PER_1K_TOKENS_USD = 0.002  # notional, for budget reporting only


def enable_hf_offline() -> None:
    """Call after the embedding model has been downloaded once."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
