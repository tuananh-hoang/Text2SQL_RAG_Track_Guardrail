import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TEST_FILE = BASE_DIR / "data" / "test_cases_v1_1_stress.csv"
DEFAULT_OUTPUT = BASE_DIR / "evaluation" / "results" / "validation" / "test_case_validation_report.csv"
BLOCK_SENTINELS = {"NO_SQL", "BLOCK"}


def get_readonly_url() -> str:
    load_dotenv(BASE_DIR / ".env")
    db_url = os.getenv("DB_READONLY_URL")
    if not db_url:
        raise RuntimeError("Missing DB_READONLY_URL in text2sql/.env")
    return db_url


def read_test_cases(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_execute_case(case: dict[str, str], engine: Any) -> dict[str, Any]:
    try:
        with engine.connect() as conn:
            df = pd.read_sql_query(case["gold_sql"].strip().rstrip(";"), conn)
    except Exception as exc:
        return {
            "id": case["id"],
            "category": case["category"],
            "difficulty": case["difficulty"],
            "gold_sql_runs": False,
            "row_count": "",
            "is_empty": "",
            "warning": "",
            "error": str(exc),
            "status": "invalid",
        }

    row_count = len(df)
    warnings = []
    if row_count == 0:
        warnings.append("empty_result")
    if row_count > 100000:
        warnings.append("too_many_rows")

    return {
        "id": case["id"],
        "category": case["category"],
        "difficulty": case["difficulty"],
        "gold_sql_runs": True,
        "row_count": row_count,
        "is_empty": row_count == 0,
        "warning": ";".join(warnings),
        "error": "",
        "status": "valid",
    }


def validate_block_case(case: dict[str, str]) -> dict[str, Any]:
    is_valid = case["gold_sql"].strip().upper() in BLOCK_SENTINELS
    return {
        "id": case["id"],
        "category": case["category"],
        "difficulty": case["difficulty"],
        "gold_sql_runs": False,
        "row_count": "",
        "is_empty": "",
        "warning": "",
        "error": "" if is_valid else "Block cases must use gold_sql NO_SQL or BLOCK",
        "status": "valid" if is_valid else "invalid",
    }


def write_report(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "category",
        "difficulty",
        "gold_sql_runs",
        "row_count",
        "is_empty",
        "warning",
        "error",
        "status",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate stress-test gold SQL against PostgreSQL.")
    parser.add_argument("--test-file", default=str(DEFAULT_TEST_FILE), help="CSV test file to validate.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Validation report CSV path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    test_file = Path(args.test_file)
    output_path = Path(args.output)

    if not test_file.is_absolute():
        test_file = BASE_DIR / test_file
    if not output_path.is_absolute():
        output_path = BASE_DIR / output_path

    cases = read_test_cases(test_file)
    engine = create_engine(
        get_readonly_url(),
        connect_args={"options": "-c statement_timeout=30000"},
    )

    rows = []
    try:
        for case in cases:
            if case["expected_behavior"] == "execute":
                rows.append(validate_execute_case(case, engine))
            else:
                rows.append(validate_block_case(case))
    finally:
        engine.dispose()

    write_report(rows, output_path)
    valid_count = sum(row["status"] == "valid" for row in rows)
    invalid_count = len(rows) - valid_count
    warning_count = sum(bool(row["warning"]) for row in rows)

    print("=== Test Case Validation Summary ===")
    print(f"Total cases : {len(rows)}")
    print(f"Valid       : {valid_count}")
    print(f"Invalid     : {invalid_count}")
    print(f"Warnings    : {warning_count}")
    print(f"Report      : {output_path}")
    return 1 if invalid_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
