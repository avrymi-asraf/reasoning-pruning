"""Local CLI wrapper for reasoning-pruning workflows.

This script is the repository-local entry point for the package CLI. It adds the
`src` tree to `sys.path` for uv/source-tree execution and then delegates all
workflow logic to `reasoning_pruning.cli`. It is intended for local smoke runs,
dry runs, and explicit dataset publishing commands.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reasoning_pruning.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
