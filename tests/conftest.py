"""Shared pytest setup.

The pipeline modules (rag/hybrid/rerank) live in notebooks/ and import each other
by bare name, so put that dir on sys.path. Also give rag a dummy API key so the
module imports without a real .env — every test mocks the actual LLM call, so no
network is ever touched.
"""
import os
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "notebooks"))

os.environ.setdefault("OPENROUTER_API_KEY", "test-dummy-key")
