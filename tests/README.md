# Tests

Run from the repository root:

```powershell
.venv\Scripts\python.exe -m pytest -q --junitxml=outputs/reports/pytest_results.xml
.venv\Scripts\python.exe -m src.data.audit
```

The suite covers loading/cleaning, feature policy and preprocessing, deterministic
regression groups, holdout/CV alignment and separation, configuration, and unchanged
classification behavior. Existing preprocessing tests fit estimators as smoke
checks; the new grouped-preparation tests do not. The audit reads the recorded
JUnit result; rerun tests after code changes before regenerating the report.
