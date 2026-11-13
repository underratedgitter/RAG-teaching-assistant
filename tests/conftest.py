"""Make the shared test helpers importable however pytest imports the tests.

The helper module lives next to the tests and is imported by name. The
default import mode puts this directory on sys.path; the importlib mode does
not, so put it there ourselves.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
