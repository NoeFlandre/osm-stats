"""Entry point shim. See :mod:`src.cli.cli` for the actual implementation."""
from src.cli.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
