"""Write the env/agri per-cluster breakdown to output/env_agri_breakdown.md.

Run with:
    .venv/bin/python -m scripts.write_env_agri_breakdown
"""
from pathlib import Path

from src.core.features.breakdown import write_breakdown_artifact

OUTPUT = Path("output/env_agri_breakdown.md")


def main() -> None:
    path = write_breakdown_artifact(OUTPUT)
    print(f"wrote: {path}")


if __name__ == "__main__":
    main()
