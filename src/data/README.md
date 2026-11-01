# Data

`load.py` loads H2 and applies stateless cleaning. `groups.py` creates stable
SHA-256 groups from normalized allowed flag-only regression inputs, excluding
ADR and all outcome/post-booking fields. `split.py` uses GroupShuffleSplit for
regression, preserves the classification stratified split, and materializes
GroupKFold assignments.

Use `preparation.prepare_regression_data("config.yaml")` as the shared entry point.
It returns aligned X/y/groups for the holdout and frozen training-only baseline
and tuning CV tuples. Keep their row order and reuse them across all future models.
Configuration paths are resolved relative to config.yaml; invalid and duplicate
settings raise errors. Low-level make_split retains its four-item return interface.

Run `python -m src.data.audit` to regenerate the team audit report. The default
group key deliberately ignores raw IDs and learned AgentTop. It is conservative
across agent variants, but rare-category encoding can still merge different keys;
the report measures that distinction. No estimator is trained by the audit.
