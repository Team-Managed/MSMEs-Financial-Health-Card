import sys
from pathlib import Path

# Add project root to sys.path so "from backend.app.xxx import ..." works
sys.path.insert(0, str(Path(__file__).parent.parent))
