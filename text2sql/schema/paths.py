from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
SCHEMA_DIR = BASE_DIR / "schema"
SCHEMA_ARTIFACTS_DIR = SCHEMA_DIR / "artifacts"

V1_SCHEMA_DIR = SCHEMA_ARTIFACTS_DIR / "v1"
V1_SCHEMA_SUMMARY_PATH = V1_SCHEMA_DIR / "schema_summary.json"

V2_SCHEMA_DIR = SCHEMA_ARTIFACTS_DIR / "v2"
V2_SUMMARIES_DIR = V2_SCHEMA_DIR / "summaries"
V2_ENTITIES_DIR = V2_SCHEMA_DIR / "entities"


def v2_summary_path(mode: str) -> Path:
    return V2_SUMMARIES_DIR / f"schema_summary_{mode}.json"


def v2_entities_path(mode: str) -> Path:
    return V2_ENTITIES_DIR / f"schema_entities_{mode}.json"
