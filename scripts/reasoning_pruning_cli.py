"""Local CLI wrapper for reasoning-pruning workflows.

This script is the repository-local execution entry point for the CLI layer in
`src/reasoning_pruning/ui_or_cli.py`. It keeps local uv usage simple before the
package is installed, while the actual workflow logic remains in the reusable
package used by tests and future Hugging Face Jobs.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reasoning_pruning.ui_or_cli import main


if __name__ == "__main__":
    raise SystemExit(main())
