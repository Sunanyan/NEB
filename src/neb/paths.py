from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SRC = ROOT / "src"
DOCS = ROOT / "docs"
TESTS = ROOT / "tests"
NOTEBOOKS = ROOT / "notebooks"
EXPERIMENTS = ROOT / "experiments"

REFERENCES = DOCS / "references"
REPORTS = DOCS / "reports"
ASSETS = DOCS / "assets"

def experiment_path(reaction, environment = "gas_phase", *,create = False,):

    path = EXPERIMENTS / reaction / environment

    if create:
        path.mkdir(parents=True, exist_ok=True)

    return path