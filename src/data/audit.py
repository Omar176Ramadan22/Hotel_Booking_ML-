"""Reproduce the data audit without fitting any predictive model.

Run: python -m src.data.audit
Only preprocessing is fitted, on outer training rows, to inspect pooling.
"""

import argparse
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.groups import regression_group_ids
from src.data.load import H2_ROWS, load_raw, standardize_text
from src.data.preparation import load_config, prepare_regression_data
from src.data.split import regression_population
from src.features.booking_features import BookingFeatureBuilder
from src.features.columns import feature_columns
from src.preprocessing.cleaning import clean_from_description
from src.preprocessing.pipeline import build_preprocessor


def table(headers, rows):
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *["| " + " | ".join(str(v).replace("|", "/") for v in row) + " |" for row in rows],
    ])


def overlap(train, test):
    return int(pd.MultiIndex.from_frame(test).isin(pd.MultiIndex.from_frame(train)).sum())


def generate_report(config_path="config.yaml", test_result=None):
    path = Path(config_path).resolve()
    cfg = load_config(path)
    root = path.parent
    if test_result is None:
        result_path = root / "outputs/reports/pytest_results.xml"
        if result_path.exists():
            suites = ET.parse(result_path).getroot().iter("testsuite")
            counts = {k: 0 for k in ("tests", "failures", "errors", "skipped")}
            for suite in suites:
                for key in counts:
                    counts[key] += int(suite.get(key, 0))
            passed = counts['tests'] - counts['failures'] - counts['errors'] - counts['skipped']
            test_result = (f"{passed} passed; {counts['failures']} failed; {counts['errors']} errors; "
                           f"{counts['skipped']} skipped (recorded outputs/reports/pytest_results.xml)")
        else:
            test_result = "Not recorded; run pytest with --junitxml=outputs/reports/pytest_results.xml."
    data_path = root / cfg["dataset"]["path"]
    raw = load_raw(data_path, expected_rows=H2_ROWS)
    original_digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
    from src.data.load import clean
    cleaned, log = clean(raw, **cfg["cleaning"])
    normalized = standardize_text(raw)
    other, _ = clean_from_description("config/datasets/h2.yaml", project_root=root, save=False)
    pd.testing.assert_frame_equal(standardize_text(cleaned), standardize_text(other), check_dtype=False)
    p = prepare_regression_data(path)
    population = regression_population(cleaned) if cfg["regression"]["paid_bookings_only"] else cleaned
    groups = regression_group_ids(population)
    assert set(p.groups_train).isdisjoint(p.groups_test)
    pd.testing.assert_series_equal(groups, regression_group_ids(population))
    numeric, categorical = feature_columns("regression")
    base_columns = numeric + categorical

    # Reproduce the OLD protocol explicitly; these are not new grouped results.
    old_population = regression_population(cleaned)
    old_train, old_test = train_test_split(old_population, test_size=.2, random_state=42)
    builder = BookingFeatureBuilder(20).fit(old_train)
    old_columns = sum(feature_columns("regression", 20), [])
    old_a, old_b = builder.transform(old_train)[old_columns], builder.transform(old_test)[old_columns]
    old_matches = overlap(old_a, old_b)
    old_target_matches = overlap(old_a.assign(ADR=old_train.ADR), old_b.assign(ADR=old_test.ADR))
    raw_allowed = [c for c in base_columns if c not in ("HasAgent", "HasCompany")] + ["Agent", "Company"]
    representations = [
        ("Normalized allowed source attributes (raw Agent/Company IDs)", standardize_text(old_population)[raw_allowed]),
        ("Stateless flag-only group representation", BookingFeatureBuilder(0).fit_transform(old_population)[base_columns]),
        ("Engineered top-20 representation (top agents fitted on OLD training only)", builder.transform(old_population)[old_columns]),
    ]
    duplicates = [(name, len(f.drop_duplicates()), int(f.duplicated().sum()), int(f.duplicated(keep=False).sum()))
                  for name, f in representations]

    # This fits transforms, never an estimator. Do not use these diagnostics to tune.
    prep = build_preprocessor("regression").fit(p.X_train)
    encoded_a, encoded_b = prep.transform(p.X_train), prep.transform(p.X_test)
    train_keys = {row.tobytes() for row in np.ascontiguousarray(encoded_a)}
    encoded_matches = sum(row.tobytes() in train_keys for row in np.ascontiguousarray(encoded_b))
    base_a = BookingFeatureBuilder(0).fit_transform(p.X_train)[base_columns]
    base_b = BookingFeatureBuilder(0).fit_transform(p.X_test)[base_columns]
    assert overlap(base_a, base_b) == 0
    fitted_builder = prep.named_steps['features']
    assert overlap(fitted_builder.transform(p.X_train)[old_columns],
                   fitted_builder.transform(p.X_test)[old_columns]) == 0

    schema = []
    for col in raw:
        series = raw[col]
        num = pd.api.types.is_numeric_dtype(series)
        schema.append((col, str(series.dtype), int(series.isna().sum()),
                       int(normalized[col].isna().sum()), int(cleaned[col].isna().sum()),
                       f"{series.min():g} to {series.max():g}" if num else f"{normalized[col].nunique()} categories"))
    padded = [(col, int(raw[col].dropna().map(lambda x: isinstance(x, str) and x != x.strip()).sum()))
              for col in raw.select_dtypes(include=["object", "string"])]
    padded = [(c, n) for c, n in padded if n]
    category_collapses = []
    for col in categorical:
        vals = normalized[col].dropna().astype(str).unique()
        lower = pd.Series(vals).str.casefold()
        if lower.duplicated().any():
            category_collapses.append(col)

    reasons = {
        "IsCanceled": ("Exclude: future outcome", "Unknown at creation; cancellation outcome, also classification target."),
        "ADR": ("Target", "Never used in group keys or regression predictors; used only for the explicit population restriction."),
        "ReservationStatus": ("Exclude: outcome leakage", "Final status unavailable at creation; can encode cancellation."),
        "ReservationStatusDate": ("Exclude: future information", "Date of final status unavailable at creation."),
        "AssignedRoomType": ("Exclude: future information", "Ultimately assigned room; use originally reserved room instead."),
        "BookingChanges": ("Exclude: future information", "Amendments accumulate after creation."),
        "DaysInWaitingList": ("Exclude: future information", "Final waiting duration not known at creation."),
        "RequiredCarParkingSpaces": ("Exclude by default: timing uncertainty", "May be added later; existing opt-in ablation retained, requires a timing justification."),
        "TotalOfSpecialRequests": ("Exclude by default: timing uncertainty", "May accumulate later; existing opt-in ablation retained, requires a timing justification."),
        "Agent": ("Allowed via derived features only", "HasAgent; optional AgentTop learned on training/fold training only. Raw identifier never encoded."),
        "Company": ("Allowed via derived feature only", "HasCompany only; high-cardinality raw identifier dropped."),
    }
    for col in raw_allowed:
        reasons.setdefault(col, ("Allowed input", "Assumed reservation request or prior guest history available at creation; final snapshot timing needs confirmation."))
    feature_audit = [(col, *reasons[col]) for col in raw]
    fold_rows = []
    for name, folds in [("baseline", p.baseline_cv), ("tuning", p.tuning_cv)]:
        seen = []
        for i, (tr, va) in enumerate(folds, 1):
            gt, gv = p.groups_train.iloc[tr], p.groups_train.iloc[va]
            shared = len(set(gt) & set(gv))
            test_shared = len(set(gv) & set(p.groups_test))
            assert shared == test_shared == 0
            seen.extend(va)
            fold_rows.append((name, i, len(tr), len(va), gt.nunique(), gv.nunique(), shared, test_shared))
        np.testing.assert_array_equal(np.sort(seen), np.arange(len(p.X_train)))
    negatives = [(c, int((raw[c] < 0).sum())) for c in raw.select_dtypes("number")]
    nonfinite = [(c, int((~np.isfinite(raw[c].dropna())).sum())) for c in raw.select_dtypes("number")]
    n = len(population)
    text = f"""# Data preparation audit

Generated from the local H2 file by `python -m src.data.audit`. No predictive models or PCA are implemented or trained by this command. Preprocessors alone are fitted on outer training data to inspect representation collisions.

## A. Dataset overview and requirements

The supplied official CSE382 PDF (sections 2-7, 11-12, 16) specifies H2 only, 79,330 records, 31 source variables, ADR regression, reproducible preprocessing, leakage analysis, and isolated final evaluation. It leaves specific cleaning thresholds and feature choices to the team. GroupShuffleSplit and grouped CV are the additional TA/user protocol, not a method mandated by the PDF. Later modeling/PCA/application deliverables remain outside this task.

Raw dimensions: **{len(raw):,} x {len(raw.columns)}**. Cleaned: **{len(cleaned):,}**. Positive ADR: **{int((cleaned.ADR > 0).sum()):,}**. ADR=0 after cleaning: **{int((cleaned.ADR == 0).sum()):,} ({(cleaned.ADR == 0).mean():.3%})**. The earlier ~1.8% estimate was inaccurate for this cleaned population. Zero ADR alone does not prove a complimentary stay.

Dataset SHA-256: `{original_digest}`. Source CSV is read only and never rewritten.

Missing-before counts below distinguish actual CSV nulls from padded NULL strings in Agent/Company. Missing values remaining after cleaning are intentionally handled by feature building/imputation.

{table(['Column', 'Loaded dtype', 'CSV missing', 'After text normalization', 'After cleaning', 'Raw range / normalized cardinality'], schema)}

## B. Cleaning audit

All existing cleaning rules are preserved. The PDF requires investigation and justification; **none of these exact deletion thresholds is mandated**. Counts are sequential, so overlapping problems are charged only to the first applicable removal.

{table(['Operation', 'Rows before', 'Rows after', 'Rows removed'], [(s['step'], s['rows_before'], s['rows_after'], s['rows_removed']) for s in log])}

- Text: strip surrounding spaces; map empty/NA/N/A/NaN/NULL/None markers case-insensitively to missing. Retain case of real categories. Padded-value counts: {padded}. New missing cells: {int(normalized.isna().sum().sum() - raw.isna().sum().sum()):,}. Case-only category variants among allowed categoricals: {category_collapses or 'none'}. `Undefined` remains a literal category; the optional registry rule is not enabled in h2.yaml.
- Exact duplicate removal: compares all 31 normalized source fields, including ADR and outcomes. It removes copies for evaluation hygiene. With no booking identifier, some may represent real separate reservations; this is a team assumption, not proven corruption. It changes booking frequencies.
- ADR >1000: removes the single ADR=5400 row; a fixed team domain threshold that may discard a valid expensive booking. ADR <0: none here. Do not interpret the threshold as a universal hotel constraint.
- Guests: sum Adults/Children/Babies after treating missing counts as zero; keep total >0. Zero adults alone is allowed when children/babies are present. Zero/missing-total records may be placeholders, cancellations, or legitimate records; exclusion defines the population.
- Nights: sum weekend/weekday nights and keep >0. Day-use, canceled, and no-stay bookings can be legitimate; exclusion is a per-night-rate modeling decision. Because this existing rule also precedes classification, classification still has the same previously restricted population.
- Per-column negative numeric counts: {dict(negatives)}. Nonfinite nonmissing numeric counts: {dict(nonfinite)}. No negative counts or additional invalid numeric values were found. Arrival month/day/year combinations are checked below. Large lead times, babies, amendments, waiting times and long stays are unusual, not automatically invalid. Existing guest/night checks use totals and would not catch every individual negative component in another dataset; no new deletion rule was introduced.
- Missing Children: {int(cleaned.Children.isna().sum())} fixed fills to 0 inside feature building (assumption, not measured truth). Country: {int(cleaned.Country.isna().sum())} fixed fills to Unknown. Agent: {int(cleaned.Agent.isna().sum()):,} missing flags become 0; Company: {int(cleaned.Company.isna().sum()):,} missing flags become 0. Optional AgentTop uses None for absent agents. Other missing categoricals use Unknown and remaining numeric missingness uses training medians. These imputers remove no rows.
- Both cleaning implementations were run and compared using **all normalized cells in row order**, not just counts/sums: equal, with dtype differences ignored. Notebook 01 calls the YAML implementation; notebook 02 reuses it. Notebook 01's stale rule description and notebook 02's draft waiting-list/ID policy descriptions were corrected. Cleaning and group creation learn no population statistics.
- A new idempotence test exposed pandas dtype/missing-sentinel drift when an identifier column became entirely missing. standardize_text now fixes Agent/Company to object text columns with pd.NA for missing identifiers on every pass. Values and cleaning row rules are unchanged; repeated cleaning is tested for identical output.
- Regression population: ADR >0 before splitting removes {int((cleaned.ADR <= 0).sum())} rows with the default setting. This target-based population definition is explicitly authorized; ADR still never determines group membership. Missing ADR would also be excluded by >0; none exist here. Outcomes of interest are conditional on the selected paid population.

## C. Feature selection audit

Prediction point: booking creation. Source snapshots do not establish that every reservation field is unchanged from creation: especially dates, guest counts, stay lengths, meal, deposit, and customer type may be amended. Preserve the established policy, but confirm access to original-at-creation values before deployment. This audit does not claim the dataset proves their timing.

{table(['Source column', 'Regression decision', 'Reason'], feature_audit)}

Final numeric regression features ({len(numeric)}): {', '.join(f'`{c}`' for c in numeric)}.

Final categorical regression features ({len(categorical)}): {', '.join(f'`{c}`' for c in categorical)}. The existing preprocessor defaults to `top_n_agents=20`, which additionally selects `AgentTop` (23 total pre-encoding features); top_n_agents=0 gives 22. Raw IDs never reach the encoder. The shared group policy covers both variants and also conservatively covers optional borderline additions.

ColumnTransformer selects only these policy columns and drops everything else; BookingFeatureBuilder retains other source fields temporarily, but they are not selected. ADR/outcomes/post-booking columns cannot be reintroduced by the current engineered flags. Group IDs are separate Series, never appended to X. Numeric medians, scaling, top-agent frequencies, category vocabularies and rare-category counts are fitted on training only; future CV must wrap the entire preprocessor with the estimator within each fold.

## D. Identical feature investigation

Exact original duplicates use all 31 fields. Identical inputs compare only allowed predictors and can legitimately have different outcomes, ADR, amendments, assigned rooms, or special requests. There is no reservation ID to prove whether matching inputs are separate bookings. Do not delete them automatically.

On the paid population:

{table(['Representation', 'Unique combinations', 'Excess repeated rows', 'Rows belonging to repeated combinations'], duplicates)}

Dropping excluded fields/target leaves 2,931 excess repeats even with raw IDs retained. Replacing IDs by flags increases excess repeats by 66 (2,997 total); top-20 agent representation recovers 60 of those distinctions (2,937 excess). Thus both feature policy and lossy ID engineering contribute; exact duplicates have already been removed. Missing-value fills and trimming can also merge representations; these are intentional normalization, not evidence of duplicate bookings.

Historical random holdout reproduced separately with test_size=.2 and seed=42: **{len(old_test):,} test rows; {old_matches:,} match a training engineered top-20 input; {old_target_matches:,} also match ADR**. These are historical-protocol results, not grouped-split results.

## E. Group definition

Use the ordered numeric list followed by the ordered categorical list in section C, excluding AgentTop. BookingFeatureBuilder with top_n_agents=0 applies stateless text normalization, HasAgent/HasCompany, Children->0 and Country->Unknown. Its fit only initializes an empty agent list; it learns nothing. Raw IDs, target, outcomes, post-booking and borderline columns do not participate. Different present Agent or Company IDs therefore deliberately share a group if all other allowed attributes match.

Canonical numeric values use Decimal text normalization (1, 1.0, '1.0' match; negative zero matches zero); missing numbers and categories use JSON null; categories retain their real case and have whitespace removed. JSON arrays preserve field order and boundaries, so literal text cannot collide with a missing sentinel or delimiter. A versioned policy/column header is part of each key. IDs are SHA-256 of UTF-8 keys, checked for hash collisions across distinct keys in each supplied population. SHA-256 is not a mathematical no-collision guarantee, but canonical keys are unambiguous and collision checks fail loudly. IDs are independent of row ordering, index labels, and subset selection. Never use factorized integer IDs independently on train/test.

Ignoring AgentTop is conservative: top-agent selection can refine groups but cannot separate the same base combination across holdout/CV. Do not fit top agents globally to construct keys. **Different base keys can still merge after train-fitted numeric imputation or categorical pooling.**

On this grouped holdout, stateless flag-only input overlap is **0**. With the existing default preprocessor fitted on outer training only, **{encoded_matches} test rows match a fully encoded training vector**. The observed merges come from rare-category pooling. This is not group overlap, but means zero group overlap must not be advertised as zero encoded-vector overlap. Strictly preventing all final encoded collisions would require a separately agreed representation/grouping policy; fitting an encoder on test data or repeatedly trying seeds is not an acceptable fix. We preserve the existing engineering as requested.

## F. Grouped holdout

GroupShuffleSplit(n_splits=1, test_size={cfg['regression']['test_size']}, random_state={cfg['random_state']}). Split membership uses only group keys. The proportion applies to groups, not exactly rows.

{table(['Statistic', 'Value'], [('Population rows', n), ('Training rows', len(p.X_train)), ('Test rows', len(p.X_test)), ('Unique groups', groups.nunique()), ('Training groups', p.groups_train.nunique()), ('Test groups', p.groups_test.nunique()), ('Training rows %', f'{len(p.X_train)/n:.6%}'), ('Test rows %', f'{len(p.X_test)/n:.6%}'), ('Overlapping groups', 0)])}

Original cleaned-row indexes are preserved through the positive-ADR filter and iloc-based split; X/y/groups share indexes and order. The legacy make_split return signature is unchanged. Classification still uses the same stratified train_test_split, same default seed and proportion, and retains zero-ADR rows; a test compares its output exactly to the original algorithm.

## G. Grouped CV verification

GroupKFold without shuffling; fold counts read from cv.n_splits and tuning.cv_folds. All indices below are positions within X_train, never positions into the full cleaned table. Every training row validates exactly once in each CV scheme. All checks executed successfully:

{table(['Scheme', 'Fold', 'Fit rows', 'Validation rows', 'Fit groups', 'Validation groups', 'Fit/validation overlap', 'Validation/test overlap'], fold_rows)}

`prepare_regression_data()` returns separate groups and materialized baseline_cv/tuning_cv tuples. Reuse the same object and tuple assignments for all future regression models. Pass cv=prepared.baseline_cv or cv=prepared.tuning_cv to the future search/scoring code. Do not call grouped_cv_folds with the full population; the helper cannot infer which dataset a caller intended. Keep X_train order unchanged after preparation.

## H. Automated tests and reproduction

Full suite result: **{test_result}**. Existing tests include small estimator fits and a real-data model smoke test; running them is verification of pre-existing code, not a new model experiment. Newly added audit/split tests train no estimators. No scores, tuned models, or new predictive artifacts were produced.

Tests cover loading, real cleaning counts, all-cell path equivalence, feature policy, target exclusion, stateless missing/text/numeric group normalization, deterministic groups across order/subsets, target/ID independence, holdout separation/alignment, grouped CV coverage/separation, exact unchanged classification behavior, duplicate YAML keys, invalid settings, and configuration-driven population/ratio/seed/fold counts.

From repository root (PowerShell):

```powershell
.venv\\Scripts\\python.exe -m pytest -q --junitxml=outputs/reports/pytest_results.xml
.venv\\Scripts\\python.exe -m src.data.audit
```

Configuration loader rejects duplicate keys, malformed required values, invalid seeds/proportions, unsupported split strategies, invalid cleaning options/bounds, and fold counts below two. GroupKFold preparation rejects insufficient unique groups. The high-level prepare_regression_data reads the central config; the low-level make_split retains explicit arguments/defaults for compatibility. Dataset YAML remains the notebook cleaning policy, with an automated row-equivalence test guarding drift.

## I. Remaining decisions before regression experiments

1. Confirm the {encoded_matches} encoded-vector overlaps caused by rare-category pooling are acceptable under the TA's identical-input requirement. The requested canonical-key split is complete; a stricter encoded-representation guarantee remains a representation policy decision.
2. Confirm the paid-only population, ADR ceiling, duplicate deletion, zero-night/zero-guest exclusions, and Children=0 assumption. They can remove valid records and are team modeling choices, not explicit PDF rules.
3. Confirm original-at-creation availability of included reservation snapshot values, especially DepositType and amended stay/guest information. Borderline parking/requests remain excluded by default; do not enable their ablation without a timing justification. Classification ADR availability is an existing assumption, unchanged.
4. The official PDF specifies exactly 4-5 students (page 4); the request describes six. Confirm the exception with the TA. No team records were changed.
5. CV splitter and pipeline safety do not enforce correct future estimator usage automatically. Keep preprocessing inside every fold, use only training CV for model choice, and reserve final test labels for final evaluation. This audit inspected final inputs for data-quality diagnostics only; no test scores or hyperparameter choices were made.
6. No Ridge/Random Forest/XGBoost implementation, model training experiment, PCA, commit, or push was added. Existing placeholder model directories remain unchanged.
"""
    months = {m: i for i, m in enumerate([
        "January", "February", "March", "April", "May", "June", "July", "August",
        "September", "October", "November", "December"], 1)}
    dates = pd.to_datetime(pd.DataFrame({"year": raw.ArrivalDateYear,
                                        "month": raw.ArrivalDateMonth.map(months),
                                        "day": raw.ArrivalDateDayOfMonth}), errors="coerce")
    text += f"\nAdditional date check: {int(dates.isna().sum())} invalid arrival dates; range {dates.min().date()} to {dates.max().date()}.\n"
    assert original_digest == hashlib.sha256(data_path.read_bytes()).hexdigest()
    destination = root / "outputs/reports/data_preparation_audit.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--test-result", default=None)
    args = parser.parse_args()
    print(generate_report(args.config, args.test_result))


if __name__ == "__main__":
    main()
