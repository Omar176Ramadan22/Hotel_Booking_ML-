# Data preparation audit

Generated from the local H2 file by `python -m src.data.audit`. No predictive models or PCA are implemented or trained by this command. Preprocessors alone are fitted on outer training data to inspect representation collisions.

## A. Dataset overview and requirements

The supplied official CSE382 PDF (sections 2-7, 11-12, 16) specifies H2 only, 79,330 records, 31 source variables, ADR regression, reproducible preprocessing, leakage analysis, and isolated final evaluation. It leaves specific cleaning thresholds and feature choices to the team. GroupShuffleSplit and grouped CV are the additional TA/user protocol, not a method mandated by the PDF. Later modeling/PCA/application deliverables remain outside this task.

Raw dimensions: **79,330 x 31**. Cleaned: **53,042**. Positive ADR: **52,320**. ADR=0 after cleaning: **722 (1.361%)**. The earlier ~1.8% estimate was inaccurate for this cleaned population. Zero ADR alone does not prove a complimentary stay.

Dataset SHA-256: `6796f5b9b2b9e7281662b29730fc2880f8e614107d0d14d09be2ede1a6e605a4`. Source CSV is read only and never rewritten.

Missing-before counts below distinguish actual CSV nulls from padded NULL strings in Agent/Company. Missing values remaining after cleaning are intentionally handled by feature building/imputation.

| Column | Loaded dtype | CSV missing | After text normalization | After cleaning | Raw range / normalized cardinality |
| --- | --- | --- | --- | --- | --- |
| IsCanceled | int64 | 0 | 0 | 0 | 0 to 1 |
| LeadTime | int64 | 0 | 0 | 0 | 0 to 629 |
| ArrivalDateYear | int64 | 0 | 0 | 0 | 2015 to 2017 |
| ArrivalDateMonth | str | 0 | 0 | 0 | 12 categories |
| ArrivalDateWeekNumber | int64 | 0 | 0 | 0 | 1 to 53 |
| ArrivalDateDayOfMonth | int64 | 0 | 0 | 0 | 1 to 31 |
| StaysInWeekendNights | int64 | 0 | 0 | 0 | 0 to 16 |
| StaysInWeekNights | int64 | 0 | 0 | 0 | 0 to 41 |
| Adults | int64 | 0 | 0 | 0 | 0 to 4 |
| Children | float64 | 4 | 4 | 4 | 0 to 3 |
| Babies | int64 | 0 | 0 | 0 | 0 to 10 |
| Meal | str | 0 | 0 | 0 | 4 categories |
| Country | str | 24 | 24 | 3 | 166 categories |
| MarketSegment | str | 0 | 0 | 0 | 8 categories |
| DistributionChannel | str | 0 | 0 | 0 | 5 categories |
| IsRepeatedGuest | int64 | 0 | 0 | 0 | 0 to 1 |
| PreviousCancellations | int64 | 0 | 0 | 0 | 0 to 21 |
| PreviousBookingsNotCanceled | int64 | 0 | 0 | 0 | 0 to 72 |
| ReservedRoomType | str | 0 | 0 | 0 | 8 categories |
| AssignedRoomType | str | 0 | 0 | 0 | 9 categories |
| BookingChanges | int64 | 0 | 0 | 0 | 0 to 21 |
| DepositType | str | 0 | 0 | 0 | 3 categories |
| Agent | str | 0 | 8131 | 5471 | 223 categories |
| Company | str | 0 | 75641 | 50300 | 207 categories |
| DaysInWaitingList | int64 | 0 | 0 | 0 | 0 to 391 |
| CustomerType | str | 0 | 0 | 0 | 4 categories |
| ADR | float64 | 0 | 0 | 0 | 0 to 5400 |
| RequiredCarParkingSpaces | int64 | 0 | 0 | 0 | 0 to 3 |
| TotalOfSpecialRequests | int64 | 0 | 0 | 0 | 0 to 5 |
| ReservationStatus | str | 0 | 0 | 0 | 3 categories |
| ReservationStatusDate | str | 0 | 0 | 0 | 864 categories |

## B. Cleaning audit

All existing cleaning rules are preserved. The PDF requires investigation and justification; **none of these exact deletion thresholds is mandated**. Counts are sequential, so overlapping problems are charged only to the first applicable removal.

| Operation | Rows before | Rows after | Rows removed |
| --- | --- | --- | --- |
| strip whitespace + missing markers -> NaN | 79330 | 79330 | 0 |
| drop exact duplicate rows | 79330 | 53428 | 25902 |
| drop ADR > 1000 | 53428 | 53427 | 1 |
| drop ADR < 0 | 53427 | 53427 | 0 |
| drop bookings with zero guests | 53427 | 53273 | 154 |
| drop bookings with zero nights | 53273 | 53042 | 231 |

- Text: strip surrounding spaces; map empty/NA/N/A/NaN/NULL/None markers case-insensitively to missing. Retain case of real categories. Padded-value counts: [('Meal', 79330), ('ReservedRoomType', 79330), ('AssignedRoomType', 79330), ('DepositType', 79330), ('Agent', 79330), ('Company', 79330)]. New missing cells: 83,772. Case-only category variants among allowed categoricals: none. `Undefined` remains a literal category; the optional registry rule is not enabled in h2.yaml.
- Exact duplicate removal: compares all 31 normalized source fields, including ADR and outcomes. It removes copies for evaluation hygiene. With no booking identifier, some may represent real separate reservations; this is a team assumption, not proven corruption. It changes booking frequencies.
- ADR >1000: removes the single ADR=5400 row; a fixed team domain threshold that may discard a valid expensive booking. ADR <0: none here. Do not interpret the threshold as a universal hotel constraint.
- Guests: sum Adults/Children/Babies after treating missing counts as zero; keep total >0. Zero adults alone is allowed when children/babies are present. Zero/missing-total records may be placeholders, cancellations, or legitimate records; exclusion defines the population.
- Nights: sum weekend/weekday nights and keep >0. Day-use, canceled, and no-stay bookings can be legitimate; exclusion is a per-night-rate modeling decision. Because this existing rule also precedes classification, classification still has the same previously restricted population.
- Per-column negative numeric counts: {'IsCanceled': 0, 'LeadTime': 0, 'ArrivalDateYear': 0, 'ArrivalDateWeekNumber': 0, 'ArrivalDateDayOfMonth': 0, 'StaysInWeekendNights': 0, 'StaysInWeekNights': 0, 'Adults': 0, 'Children': 0, 'Babies': 0, 'IsRepeatedGuest': 0, 'PreviousCancellations': 0, 'PreviousBookingsNotCanceled': 0, 'BookingChanges': 0, 'DaysInWaitingList': 0, 'ADR': 0, 'RequiredCarParkingSpaces': 0, 'TotalOfSpecialRequests': 0}. Nonfinite nonmissing numeric counts: {'IsCanceled': 0, 'LeadTime': 0, 'ArrivalDateYear': 0, 'ArrivalDateWeekNumber': 0, 'ArrivalDateDayOfMonth': 0, 'StaysInWeekendNights': 0, 'StaysInWeekNights': 0, 'Adults': 0, 'Children': 0, 'Babies': 0, 'IsRepeatedGuest': 0, 'PreviousCancellations': 0, 'PreviousBookingsNotCanceled': 0, 'BookingChanges': 0, 'DaysInWaitingList': 0, 'ADR': 0, 'RequiredCarParkingSpaces': 0, 'TotalOfSpecialRequests': 0}. No negative counts or additional invalid numeric values were found. Arrival month/day/year combinations are checked below. Large lead times, babies, amendments, waiting times and long stays are unusual, not automatically invalid. Existing guest/night checks use totals and would not catch every individual negative component in another dataset; no new deletion rule was introduced.
- Missing Children: 4 fixed fills to 0 inside feature building (assumption, not measured truth). Country: 3 fixed fills to Unknown. Agent: 5,471 missing flags become 0; Company: 50,300 missing flags become 0. Optional AgentTop uses None for absent agents. Other missing categoricals use Unknown and remaining numeric missingness uses training medians. These imputers remove no rows.
- Both cleaning implementations were run and compared using **all normalized cells in row order**, not just counts/sums: equal, with dtype differences ignored. Notebook 01 calls the YAML implementation; notebook 02 reuses it. Notebook 01's stale rule description and notebook 02's draft waiting-list/ID policy descriptions were corrected. Cleaning and group creation learn no population statistics.
- A new idempotence test exposed pandas dtype/missing-sentinel drift when an identifier column became entirely missing. standardize_text now fixes Agent/Company to object text columns with pd.NA for missing identifiers on every pass. Values and cleaning row rules are unchanged; repeated cleaning is tested for identical output.
- Regression population: ADR >0 before splitting removes 722 rows with the default setting. This target-based population definition is explicitly authorized; ADR still never determines group membership. Missing ADR would also be excluded by >0; none exist here. Outcomes of interest are conditional on the selected paid population.

## C. Feature selection audit

Prediction point: booking creation. Source snapshots do not establish that every reservation field is unchanged from creation: especially dates, guest counts, stay lengths, meal, deposit, and customer type may be amended. Preserve the established policy, but confirm access to original-at-creation values before deployment. This audit does not claim the dataset proves their timing.

| Source column | Regression decision | Reason |
| --- | --- | --- |
| IsCanceled | Exclude: future outcome | Unknown at creation; cancellation outcome, also classification target. |
| LeadTime | Allowed input | Assumed reservation request or prior guest history available at creation; final snapshot timing needs confirmation. |
| ArrivalDateYear | Allowed input | Assumed reservation request or prior guest history available at creation; final snapshot timing needs confirmation. |
| ArrivalDateMonth | Allowed input | Assumed reservation request or prior guest history available at creation; final snapshot timing needs confirmation. |
| ArrivalDateWeekNumber | Allowed input | Assumed reservation request or prior guest history available at creation; final snapshot timing needs confirmation. |
| ArrivalDateDayOfMonth | Allowed input | Assumed reservation request or prior guest history available at creation; final snapshot timing needs confirmation. |
| StaysInWeekendNights | Allowed input | Assumed reservation request or prior guest history available at creation; final snapshot timing needs confirmation. |
| StaysInWeekNights | Allowed input | Assumed reservation request or prior guest history available at creation; final snapshot timing needs confirmation. |
| Adults | Allowed input | Assumed reservation request or prior guest history available at creation; final snapshot timing needs confirmation. |
| Children | Allowed input | Assumed reservation request or prior guest history available at creation; final snapshot timing needs confirmation. |
| Babies | Allowed input | Assumed reservation request or prior guest history available at creation; final snapshot timing needs confirmation. |
| Meal | Allowed input | Assumed reservation request or prior guest history available at creation; final snapshot timing needs confirmation. |
| Country | Allowed input | Assumed reservation request or prior guest history available at creation; final snapshot timing needs confirmation. |
| MarketSegment | Allowed input | Assumed reservation request or prior guest history available at creation; final snapshot timing needs confirmation. |
| DistributionChannel | Allowed input | Assumed reservation request or prior guest history available at creation; final snapshot timing needs confirmation. |
| IsRepeatedGuest | Allowed input | Assumed reservation request or prior guest history available at creation; final snapshot timing needs confirmation. |
| PreviousCancellations | Allowed input | Assumed reservation request or prior guest history available at creation; final snapshot timing needs confirmation. |
| PreviousBookingsNotCanceled | Allowed input | Assumed reservation request or prior guest history available at creation; final snapshot timing needs confirmation. |
| ReservedRoomType | Allowed input | Assumed reservation request or prior guest history available at creation; final snapshot timing needs confirmation. |
| AssignedRoomType | Exclude: future information | Ultimately assigned room; use originally reserved room instead. |
| BookingChanges | Exclude: future information | Amendments accumulate after creation. |
| DepositType | Allowed input | Assumed reservation request or prior guest history available at creation; final snapshot timing needs confirmation. |
| Agent | Allowed via derived features only | HasAgent; optional AgentTop learned on training/fold training only. Raw identifier never encoded. |
| Company | Allowed via derived feature only | HasCompany only; high-cardinality raw identifier dropped. |
| DaysInWaitingList | Exclude: future information | Final waiting duration not known at creation. |
| CustomerType | Allowed input | Assumed reservation request or prior guest history available at creation; final snapshot timing needs confirmation. |
| ADR | Target | Never used in group keys or regression predictors; used only for the explicit population restriction. |
| RequiredCarParkingSpaces | Exclude by default: timing uncertainty | May be added later; existing opt-in ablation retained, requires a timing justification. |
| TotalOfSpecialRequests | Exclude by default: timing uncertainty | May accumulate later; existing opt-in ablation retained, requires a timing justification. |
| ReservationStatus | Exclude: outcome leakage | Final status unavailable at creation; can encode cancellation. |
| ReservationStatusDate | Exclude: future information | Date of final status unavailable at creation. |

Final numeric regression features (14): `LeadTime`, `ArrivalDateYear`, `ArrivalDateWeekNumber`, `ArrivalDateDayOfMonth`, `StaysInWeekendNights`, `StaysInWeekNights`, `Adults`, `Children`, `Babies`, `IsRepeatedGuest`, `PreviousCancellations`, `PreviousBookingsNotCanceled`, `HasAgent`, `HasCompany`.

Final categorical regression features (8): `ArrivalDateMonth`, `Meal`, `Country`, `MarketSegment`, `DistributionChannel`, `ReservedRoomType`, `DepositType`, `CustomerType`. The existing preprocessor defaults to `top_n_agents=20`, which additionally selects `AgentTop` (23 total pre-encoding features); top_n_agents=0 gives 22. Raw IDs never reach the encoder. The shared group policy covers both variants and also conservatively covers optional borderline additions.

ColumnTransformer selects only these policy columns and drops everything else; BookingFeatureBuilder retains other source fields temporarily, but they are not selected. ADR/outcomes/post-booking columns cannot be reintroduced by the current engineered flags. Group IDs are separate Series, never appended to X. Numeric medians, scaling, top-agent frequencies, category vocabularies and rare-category counts are fitted on training only; future CV must wrap the entire preprocessor with the estimator within each fold.

## D. Identical feature investigation

Exact original duplicates use all 31 fields. Identical inputs compare only allowed predictors and can legitimately have different outcomes, ADR, amendments, assigned rooms, or special requests. There is no reservation ID to prove whether matching inputs are separate bookings. Do not delete them automatically.

On the paid population:

| Representation | Unique combinations | Excess repeated rows | Rows belonging to repeated combinations |
| --- | --- | --- | --- |
| Normalized allowed source attributes (raw Agent/Company IDs) | 49389 | 2931 | 5188 |
| Stateless flag-only group representation | 49323 | 2997 | 5317 |
| Engineered top-20 representation (top agents fitted on OLD training only) | 49383 | 2937 | 5200 |

Dropping excluded fields/target leaves 2,931 excess repeats even with raw IDs retained. Replacing IDs by flags increases excess repeats by 66 (2,997 total); top-20 agent representation recovers 60 of those distinctions (2,937 excess). Thus both feature policy and lossy ID engineering contribute; exact duplicates have already been removed. Missing-value fills and trimming can also merge representations; these are intentional normalization, not evidence of duplicate bookings.

Historical random holdout reproduced separately with test_size=.2 and seed=42: **10,464 test rows; 927 match a training engineered top-20 input; 673 also match ADR**. These are historical-protocol results, not grouped-split results.

## E. Group definition

Use the ordered numeric list followed by the ordered categorical list in section C, excluding AgentTop. BookingFeatureBuilder with top_n_agents=0 applies stateless text normalization, HasAgent/HasCompany, Children->0 and Country->Unknown. Its fit only initializes an empty agent list; it learns nothing. Raw IDs, target, outcomes, post-booking and borderline columns do not participate. Different present Agent or Company IDs therefore deliberately share a group if all other allowed attributes match.

Canonical numeric values use Decimal text normalization (1, 1.0, '1.0' match; negative zero matches zero); missing numbers and categories use JSON null; categories retain their real case and have whitespace removed. JSON arrays preserve field order and boundaries, so literal text cannot collide with a missing sentinel or delimiter. A versioned policy/column header is part of each key. IDs are SHA-256 of UTF-8 keys, checked for hash collisions across distinct keys in each supplied population. SHA-256 is not a mathematical no-collision guarantee, but canonical keys are unambiguous and collision checks fail loudly. IDs are independent of row ordering, index labels, and subset selection. Never use factorized integer IDs independently on train/test.

Ignoring AgentTop is conservative: top-agent selection can refine groups but cannot separate the same base combination across holdout/CV. Do not fit top agents globally to construct keys. **Different base keys can still merge after train-fitted numeric imputation or categorical pooling.**

On this grouped holdout, stateless flag-only input overlap is **0**. With the existing default preprocessor fitted on outer training only, **23 test rows match a fully encoded training vector**. The observed merges come from rare-category pooling. This is not group overlap, but means zero group overlap must not be advertised as zero encoded-vector overlap. Strictly preventing all final encoded collisions would require a separately agreed representation/grouping policy; fitting an encoder on test data or repeatedly trying seeds is not an acceptable fix. We preserve the existing engineering as requested.

## F. Grouped holdout

GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42). Split membership uses only group keys. The proportion applies to groups, not exactly rows.

| Statistic | Value |
| --- | --- |
| Population rows | 52320 |
| Training rows | 41867 |
| Test rows | 10453 |
| Unique groups | 49323 |
| Training groups | 39458 |
| Test groups | 9865 |
| Training rows % | 80.021024% |
| Test rows % | 19.978976% |
| Overlapping groups | 0 |

Original cleaned-row indexes are preserved through the positive-ADR filter and iloc-based split; X/y/groups share indexes and order. The legacy make_split return signature is unchanged. Classification still uses the same stratified train_test_split, same default seed and proportion, and retains zero-ADR rows; a test compares its output exactly to the original algorithm.

## G. Grouped CV verification

GroupKFold without shuffling; fold counts read from cv.n_splits and tuning.cv_folds. All indices below are positions within X_train, never positions into the full cleaned table. Every training row validates exactly once in each CV scheme. All checks executed successfully:

| Scheme | Fold | Fit rows | Validation rows | Fit groups | Validation groups | Fit/validation overlap | Validation/test overlap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 1 | 33493 | 8374 | 31566 | 7892 | 0 | 0 |
| baseline | 2 | 33493 | 8374 | 31566 | 7892 | 0 | 0 |
| baseline | 3 | 33494 | 8373 | 31567 | 7891 | 0 | 0 |
| baseline | 4 | 33494 | 8373 | 31567 | 7891 | 0 | 0 |
| baseline | 5 | 33494 | 8373 | 31566 | 7892 | 0 | 0 |
| tuning | 1 | 27911 | 13956 | 26305 | 13153 | 0 | 0 |
| tuning | 2 | 27911 | 13956 | 26305 | 13153 | 0 | 0 |
| tuning | 3 | 27912 | 13955 | 26306 | 13152 | 0 | 0 |

`prepare_regression_data()` returns separate groups and materialized baseline_cv/tuning_cv tuples. Reuse the same object and tuple assignments for all future regression models. Pass cv=prepared.baseline_cv or cv=prepared.tuning_cv to the future search/scoring code. Do not call grouped_cv_folds with the full population; the helper cannot infer which dataset a caller intended. Keep X_train order unchanged after preparation.

## H. Automated tests and reproduction

Full suite result: **61 passed; 0 failed; 0 errors; 0 skipped (recorded outputs/reports/pytest_results.xml)**. Existing tests include small estimator fits and a real-data model smoke test; running them is verification of pre-existing code, not a new model experiment. Newly added audit/split tests train no estimators. No scores, tuned models, or new predictive artifacts were produced.

Tests cover loading, real cleaning counts, all-cell path equivalence, feature policy, target exclusion, stateless missing/text/numeric group normalization, deterministic groups across order/subsets, target/ID independence, holdout separation/alignment, grouped CV coverage/separation, exact unchanged classification behavior, duplicate YAML keys, invalid settings, and configuration-driven population/ratio/seed/fold counts.

From repository root (PowerShell):

```powershell
.venv\Scripts\python.exe -m pytest -q --junitxml=outputs/reports/pytest_results.xml
.venv\Scripts\python.exe -m src.data.audit
```

Configuration loader rejects duplicate keys, malformed required values, invalid seeds/proportions, unsupported split strategies, invalid cleaning options/bounds, and fold counts below two. GroupKFold preparation rejects insufficient unique groups. The high-level prepare_regression_data reads the central config; the low-level make_split retains explicit arguments/defaults for compatibility. Dataset YAML remains the notebook cleaning policy, with an automated row-equivalence test guarding drift.

## I. Remaining decisions before regression experiments

1. Confirm the 23 encoded-vector overlaps caused by rare-category pooling are acceptable under the TA's identical-input requirement. The requested canonical-key split is complete; a stricter encoded-representation guarantee remains a representation policy decision.
2. Confirm the paid-only population, ADR ceiling, duplicate deletion, zero-night/zero-guest exclusions, and Children=0 assumption. They can remove valid records and are team modeling choices, not explicit PDF rules.
3. Confirm original-at-creation availability of included reservation snapshot values, especially DepositType and amended stay/guest information. Borderline parking/requests remain excluded by default; do not enable their ablation without a timing justification. Classification ADR availability is an existing assumption, unchanged.
4. The official PDF specifies exactly 4-5 students (page 4); the request describes six. Confirm the exception with the TA. No team records were changed.
5. CV splitter and pipeline safety do not enforce correct future estimator usage automatically. Keep preprocessing inside every fold, use only training CV for model choice, and reserve final test labels for final evaluation. This audit inspected final inputs for data-quality diagnostics only; no test scores or hyperparameter choices were made.
6. No Ridge/Random Forest/XGBoost implementation, model training experiment, PCA, commit, or push was added. Existing placeholder model directories remain unchanged.

Additional date check: 0 invalid arrival dates; range 2015-07-01 to 2017-08-31.
