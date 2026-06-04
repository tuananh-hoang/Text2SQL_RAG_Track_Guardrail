# Text-to-SQL Evaluation

## Folder Layout

```text
evaluation/
|-- evaluator.py                  # CLI entrypoint for benchmark runs
|-- replay_results.py             # Replay generated SQL without calling LLM
|-- validate_test_cases.py         # Gold SQL validation
|-- specs/
|   `-- test_spec_v1_1.json
|-- results/
|   |-- v1/                        # V1 baseline outputs
|   |-- v1_1/                      # V1.1 stress/debug outputs
|   |-- mentor/                    # Mentor hard-case runs and gold-SQL analysis
|   |-- validation/                # Gold SQL validation reports
|   `-- v2/                        # Reserved for V2 outputs
`-- viewers/
    |-- csv_review_dashboard.html  # Generic CSV review dashboard
    `-- testcase_viewer.html       # Test-case focused viewer
```

## Commands

Validate stress-test gold SQL:

```powershell
python evaluation\validate_test_cases.py `
  --test-file data\test_cases_v1_1_stress.csv `
  --output evaluation\results\validation\test_case_validation_report.csv
```

Run V1.1 stress evaluation:

```powershell
python evaluation\evaluator.py `
  --test-file data\test_cases_v1_1_stress.csv `
  --output evaluation\results\v1_1\results_v1_1_stress.csv `
  --error-analysis evaluation\results\v1_1\error_analysis_v1_1.csv
```

Replay V1.1 generated SQL:

```powershell
python evaluation\replay_results.py `
  --input-results evaluation\results\v1_1\results_v1_1_general_prompt.csv `
  --output evaluation\results\v1_1\results_v1_1_general_prompt_replay.csv `
  --error-analysis evaluation\results\v1_1\error_analysis_v1_1_general_prompt_replay.csv
```

## Review UI

Open this file in a browser:

```text
text2sql/evaluation/viewers/csv_review_dashboard.html
```

Then load CSV files from the versioned result folders, for example:

```text
text2sql/evaluation/results/v1_1/
text2sql/evaluation/results/mentor/
text2sql/evaluation/results/validation/
```
