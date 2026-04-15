"""
pytest configuration for InsureAI test suite.
Ensures the project root is on sys.path for all tests.
"""

import sys
from pathlib import Path

# Add project root to path so all imports resolve correctly
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (skipped in quick runs)"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests requiring live API access"
    )
