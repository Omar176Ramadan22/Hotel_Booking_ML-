# Ridge regression: frozen model report

Ridge is finished and frozen. This report describes existing saved experiments; it does not change the pipeline, settings, predictions, or results. No new fitting, tuning, PCA, or test-set scoring was performed for this documentation update.

## Task and data

Predict ADR at booking creation using attributes assumed to be available then. The source is H2 only (79,330 rows, 31 variables). Fixed cleaning leaves 53,042 rows. Regression uses the 52,320 positive-ADR bookings: 41,867 training rows and 10,453 held-out rows. Zero-ADR stays are outside this regression population; their meaning is not verified as complimentary stays.

Groups are built from normalized permitted inputs using agent/company presence flags, without targets, outcomes, or learned top-agent lists. A seeded group split keeps each raw group on one side of the holdout. Training CV uses the saved deterministic grouped folds. Rare-category pooling and imputation can still make different raw groups share an encoded vector: the earlier audit found 23 held-out rows matching training encodings. Raw-group separation is not a guarantee of zero encoded-vector overlap.

## Baseline versus tuned

Values are copied from [the CV CSV](../tables/regression_cv_results.csv) and [the test CSV](../tables/regression_test_results.csv). CV RMSE is the mean of fold RMSEs; test metrics use the original unclipped model outputs.

| Version | Alpha | Numeric degree | CV RMSE | CV MAE | CV R2 | Test RMSE | Test MAE | Test R2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 1.0 | 1 | 20.7796 | 14.6843 | 0.7246 | 21.3064 | 14.8641 | 0.7051 |
| tuned | 0.0001 | 2 | 19.7785 | 13.6341 | 0.7505 | 20.1471 | 13.7340 | 0.7363 |

The tuned model is linear in an expanded basis containing numeric squares and pairwise interactions; it is not linear in the original booking inputs. Numeric processing is median imputation, polynomial expansion, then scaling. Categorical processing is Unknown imputation and one-hot encoding with rare-category pooling. The top 20 agents are learned inside fitting. All learned preprocessing and Ridge belong to one sklearn pipeline.

Most of the improvement came from the degree-2 representation. An earlier read-only audit reconstructed the historical five-fold grid: degree 1 at alpha 0.0001 gave CV RMSE 20.7766, compared with 20.7796 for degree 1 at alpha 1.0. Degree 2 at alpha 0.0001 gave 19.7785. Thus changing alpha alone barely helped the linear representation. These ablation values are an audit reconstruction, not an originally persisted complete search table; no search was rerun for this report.

The frozen recipe is [regression_ridge_best_params.json](../results/regression_ridge_best_params.json). Baseline alpha is 1.0 with degree 1; the tuned recipe is alpha 0.0001 with degree 2.

## Regression leakage audit

The prediction point is booking creation. The following exclusions apply before an estimator sees the data.

| Excluded source column | Reason |
|---|---|
| ADR | Regression target; never an input. |
| IsCanceled | Eventual booking outcome, unavailable at creation. |
| ReservationStatus | Final booking outcome, which can also encode cancellation. |
| ReservationStatusDate | Timing of a later outcome. |
| AssignedRoomType | Room assigned later; it can indirectly reveal the charged rate. Use ReservedRoomType. |
| BookingChanges | Accumulates amendments after creation. |
| DaysInWaitingList | Final waiting-list duration may be known only later. |
| RequiredCarParkingSpaces | Final parking requests can include later additions; excluded by default. |
| TotalOfSpecialRequests | Final request count can include later additions; excluded by default. |
| Agent | Raw identifier replaced by HasAgent and a training-learned AgentTop category. |
| Company | Raw identifier replaced by HasCompany. |

Arrival information, requested stay length, party size, reserved room, meal, market/distribution channel, deposit/customer type, and prior-booking history are treated as booking-time attributes. Their timing is an explicit assumption: the released dataset does not prove that every recorded value equals its value at creation. This should be discussed with the TA if a stricter interpretation is required.

Fixed cleaning learns no medians, frequencies, or scaling statistics. Medians, top-agent membership, rare categories, encoder categories, scaling parameters, and Ridge coefficients are learned inside training folds. Final fitting uses only the outer training rows. Prediction reads the saved pipeline and matching bounds, not the dataset. The refit command uses the frozen JSON recipe and original split to rebuild both baseline/tuned artifacts and their bounds without a search or evaluation.

## Diagnostics

Existing [actual versus predicted](../figures/ridge_cv_actual_vs_predicted.png) and [residual](../figures/ridge_cv_residuals.png) plots use training-only out-of-fold predictions. They are post-selection diagnostics, not independent model-selection evidence. The [diagnostic JSON](../results/regression_ridge_diagnostics.json) reports pooled training-CV RMSE 19.7809; the small difference from mean fold RMSE 19.7785 comes from pooling versus averaging. It records four negative out-of-fold predictions.

## Limitations and evaluation history

1. The holdout was inspected while the search was changing. Polynomial alpha 0.001 produced test RMSE 20.1679; the alpha range was extended because the best value hit a training-CV boundary, and alpha 0.0001 later produced 20.1471. This is a previously inspected holdout, not a blind test. The reported test number must not be described as an unbiased final estimate.
2. Historical tuned Ridge used the same five folds for selection and reporting. Its CV score can be optimistic. The current 3-fold search / 5-fold comparison plan for future unfrozen models does not retroactively change Ridge's protocol; a post-selection comparison on the same training pool is not nested CV.
3. Degree-2 terms can extrapolate badly. The prediction interface rejects numeric inputs and complete arrival dates outside saved training bounds, including years after 2017. It checks valid arrival dates and H2 week consistency. Accepted categorical values are normalized at the interface and mapped back to trained spellings; unseen categories are rejected with valid options.
4. Ridge remains linear in its expanded basis. The recorded tuned XGBoost is stronger here (CV RMSE 14.4006, test RMSE 14.6056), consistent with trees modeling interactions and nonlinear effects more flexibly. This is a comparison of recorded runs, not a new deployment selection; protocols and holdout history must be disclosed.
5. The UI clips a finite negative prediction to zero with a warning; the saved evaluation CSVs retain raw outputs. Positive-ADR training does not guarantee positive Ridge predictions.
6. Random grouped splits over 2015-2017 bookings do not establish future-year performance. Deployment after the observed period needs a separate validation plan.

## Team proposals: not decisions

All owners below are **TBD**. The team must approve these proposals; Ridge remains frozen.

- **Model-selection rule:** select the lowest tuned CV RMSE; break ties using lower CV MAE, then shorter training time measured under comparable conditions. Never select using test results. Owner: TBD.
- **Comparison table:** the regression owner is proposed to maintain the common baseline/tuned table, with protocol and representation labels. Owner: TBD. A third regression model is still needed for the complete project requirement; only Ridge and XGBoost currently appear in the recorded regression CSVs.
- **Current post-2017 behavior:** reject. Retaining it avoids unsupported polynomial extrapolation but blocks present-day bookings and other dates beyond the training maximum.
- **Alternative: warn and use 2017 as the model's year.** This could make a demo usable, but it assumes no later price drift. Its future-year prediction error has not been measured. Clipping the year alone would not resolve the full arrival-date bound or week consistency; the team would need an explicit date policy. Not implemented.
- **Alternative: drop ArrivalDateYear.** Earlier training-only five-fold ablations increased Ridge CV RMSE from 19.78 to 21.10 and a 100-tree XGBoost baseline from 15.46 to 17.12. The latter is a baseline ablation, not an ablation of the tuned 430-tree model. Dropping the feature costs historical CV performance and still does not prove future generalization. Not implemented.

## PCA: deferred

No PCA experiment is included or claimed complete. The later comparison will use separate new source/output files and preserve all existing code, settings, and results. Compare Ridge and a tree model with/without PCA on the frozen training CV folds only. Fit PCA and all other learned preprocessing inside each fold's pipeline. Report an explained-variance curve, justify retained components, use a prepared and appropriately scaled numerical representation, and perform no test-set scoring for the PCA comparison. Owner: TBD.

LDA and the classification original/PCA/LDA comparison remain separate team requirements.

## Optional coefficient discussion

The following are copied directly from the existing tuned joblib; no fitting or scoring was run. They are coefficients of the transformed basis, not independent effects of raw booking fields.

| Sign | Transformed term | Coefficient |
|---|---|---:|
| Positive | `ArrivalDateYear Children` | 5375.8961 |
| Positive | `ArrivalDateDayOfMonth` | 5020.7644 |
| Positive | `ArrivalDateYear` | 4052.0433 |
| Positive | `LeadTime` | 3691.5881 |
| Positive | `ArrivalDateYear Adults` | 3683.6781 |
| Negative | `Children` | -5380.1392 |
| Negative | `ArrivalDateYear ArrivalDateDayOfMonth` | -5019.8270 |
| Negative | `ArrivalDateYear^2` | -4045.2843 |
| Negative | `LeadTime ArrivalDateYear` | -3704.1721 |
| Negative | `Adults` | -3696.6451 |

A positive coefficient raises that term's contribution as its standardized value increases; a negative coefficient lowers it. Numeric polynomial terms are scaled, while categorical indicators use 0/1 encoding. ArrivalDateYear is close to a large constant, so products involving it are highly correlated with the corresponding main effects. The large opposite-signed coefficients above can cancel. Therefore, do not infer that adding a child or adult alone lowers ADR from the sign of its main-effect coefficient; its interactions change at the same time. These coefficients are descriptive, not causal or a stable feature-importance ranking.

## Reproduction and revision

Repository: https://github.com/Omar176Ramadan22/Hotel_Booking_ML-

Frozen implementation commit: `4e9581b27aaabd47b93fed257dd5759a6ac06a60`. This documentation is added in a following commit without changing implementation or saved results. See the root README for environment setup, refit/predict commands, and submission ZIP revision recording.
