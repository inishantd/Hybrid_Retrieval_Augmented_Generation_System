import os
import sys
from pathlib import Path

# Ensure src.config can be imported even without a real .env present
os.environ.setdefault("GROQ_API_KEY", "test-key-for-unit-tests")

# Make "src" importable when pytest is run from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))