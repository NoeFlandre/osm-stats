"""Thin Python wrapper that invokes the bash reproducer.

Lets users call ``python -m scripts.reproducibility.reproduce``
without having to know the path to ``reproduce.sh``.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SH = ROOT / "data" / "reproducibility" / "reproduce.sh"


def main() -> int:
    if not SH.exists():
        print(f"error: reproducer not found at {SH}")
        return 1
    print(f"running: bash {SH}")
    return subprocess.call(["bash", str(SH)])


if __name__ == "__main__":
    sys.exit(main())
