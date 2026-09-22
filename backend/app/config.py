"""All settings live here, read from environment / .env. Nothing is hard-coded elsewhere."""
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.1-flash-lite")
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://rag:rag@localhost:5433/ragdb")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))      # characters per chunk
TOP_K = int(os.getenv("TOP_K", "4"))                  # chunks retrieved per question
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.40"))  # below this = "not relevant"
MAX_RETRIES = 1                                       # regenerate at most once if validation fails

PRICE_INPUT_PER_M = float(os.getenv("PRICE_INPUT_PER_M", "0.25"))
PRICE_OUTPUT_PER_M = float(os.getenv("PRICE_OUTPUT_PER_M", "1.50"))
