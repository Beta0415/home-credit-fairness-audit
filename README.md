# Fairness Audit Project

This is a fairness audit project. It examines two loan-default models on Home Credit applications, where `TARGET = 1` means the applicant had a repayment difficulty and `TARGET = 0` means they did not.

The models were not built by the author of this audit. They come from [zakyikow/loan-default-prediction](https://github.com/zakyikow/loan-default-prediction). The data come from the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) competition.

The audit uses the XGBoost and Keras MLP specifications from that work. Their hyperparameters were already chosen there. This project does not reselect them. It profiles the applications, checks data quality, fits those two specifications, and then audits the saved models on the held-out test set. `data.csv` is the competition application table saved locally (307,511 rows, 122 fields, about 166 MB).

## Notebooks

### Data profiling — `data_profiling.ipynb`

A first look at the raw application file before any rows are dropped and before the not-employed codes are treated as missing. Tables are written to `output/profiling/`.

1. **Shape, Schema & Row Identity.** Loads the file, checks that `SK_ID_CURR` identifies each application once, and records the column types, null counts, and number of distinct values.
2. **Target Balance.** Counts the 24,825 defaults and 282,686 non-defaults (8.07% default).
3. **Missingness.** Counts null cells only. The employment code `365243` and organization code `XNA` are still treated as stored values here. Also flags fields whose most common value covers at least 99.5% of rows.
4. **Categorical Value.** Lists the 16 non-numeric fields, how many categories each takes, and which category is most common.
5. **Numeric Range & Sentinel Value.** Summarizes the numeric fields and counts placeholder or out-of-range values, including the 55,374 not-employed employment codes.
6. **Potential Group Used in Fairness Audit.** Builds gender, age band, income band, education, and family status, and reports the observed default rate in each group.
7. **Missingness & External Credit Score By Gender.** Compares null rates of seven often-missing fields, and the distribution of the three external credit scores, by gender.
8. **Saved Profiling Output.** Writes the tables above and counts the six rows later dropped for modeling: four with gender `XNA` and two with family status `Unknown`.

### Data quality — `data_quality.ipynb`

Checks missingness, extreme values, and whether gender, age, and income are represented consistently in the training, validation, and test splits. The models are not retrained. Tables are written to `output/data_quality/` and `output/audit/`.

1. **Overall Missingness Profiling.** Counts missing values on all 307,511 applications. Nulls are missing, and so are `DAYS_EMPLOYED = 365243` and `ORGANIZATION_TYPE = "XNA"`.
2. **Missingness Pattern and Mechanism.** Asks whether missingness is spread evenly. The six excluded rows are dropped first, leaving 307,505 applications.
   - **Gender, age band, income, and default.** Shows the missing rate of each field inside each group.
   - **2.2 Association With Gender, Age, Income, and Default.** Tests those gaps with a chi-square test and, where there are enough missing rows, a logistic regression.
   - **2.3 MCAR Test.** Runs Little's MCAR test on a numeric subset of fields.
3. **Outlier and Winsorization Audit.** Describes the caps used when the models were trained and where the extreme values sit.
   - **3.1 Outlier Prevalence.** Counts how many observed values exceed each cap and how far the raw tail extends past it.
   - **3.2 Demographic Outlier Disparities.** Checks whether values above the cap concentrate in a gender, age band, or income band.
   - **3.3 Outliers and Default Outcomes.** Compares the default rate above each cap with the default rate of the other observed values.
   - **3.4 Synthetic Ratio If Winsorisation Applied.** Checks whether an extreme monetary value can still enter the model through a ratio built before the cap.
4. **Train/Validation/Test Representation.** Compares the original 70/15/15 split. The split is stratified on the overall target, with seed 42: 215,253 training, 46,126 validation, and 46,126 test applications.
   - **4.1 Demographic Representation Across Splits.** Compares the gender, age, and income mix of the three sets.
   - **4.2 Default-Rate Representation.** Compares each group's observed default rate across the three sets.

### Loan default prediction — `xgboost_mlp.ipynb`

Fits the two models used in the fairness audit and scores them on the test set. The hyperparameters were preconfigured in [zakyikow/loan-default-prediction](https://github.com/zakyikow/loan-default-prediction). This notebook does not search for them again. It does choose each model's validation threshold, and the epoch at which the MLP stops. Fitted models, the preprocessor, and the test metrics are written to `output/model/`.

1. **Shared Preprocessing.** Drops the six excluded applications, flags the employment sentinel, builds age and four credit ratios, caps extreme counts and monetary values, and makes the stratified 70/15/15 split. Both models use the same transformed columns.
2. **XGBoost Model.** Fits the preconfigured gradient-boosted trees, with the default class up-weighted by the training imbalance, and picks the validation threshold that maximizes the gap between the true positive rate and the false positive rate.
3. **MLP Neural Network.** Fits the preconfigured network (128 units, then 64, with dropout 0.5) using the same class-weight ratio and the same kind of validation threshold. Early stopping watches validation ROC-AUC.

The last cell compares the two models on the test set and saves `test_metrics.csv`.

### Fairness audit — `fairness_audit.ipynb`

Scores the saved XGBoost and MLP models on the reconstructed test set and reports group differences in approval, uncertainty, and feature attribution. Sections 1–9 score the saved models. Section 10 retrains both models without the gender column. A grouping selector controls which groups appear in Sections 4–8, the Section 8 comparison, and Section 9.2. Tables are written to `output/fairness/`.

1. **Reconstruct Validation & Test Data.** Rebuilds the modeling table and the original train, validation, and test split, then scores the saved models. The models are not retrained here.
2. **Overall Model Performance.** Chooses one validation threshold per model and reports overall discrimination and classification performance. Default is the positive class in this section.
3. **Audit Population, Protected Group & Data Quality.** Records who was removed before modeling, missingness by gender, and the test-set size and default rate inside each group.

**Grouping Selector.** Chooses which of gender, age band, income band, gender × age band, and gender × income band are shown in Sections 4–8, the Section 8 comparison, and Section 9.2.

4. **Fairness Metric.** Measures demographic parity, equal opportunity, and equalized odds, each as a difference and a ratio. Approval of a non-default applicant is the favorable outcome.
5. **Fairness Metric Statistical Uncertainty.** Adds 95% bootstrap intervals to the Section 4 metrics, using 1,000 resamples of the test set.
6. **Within Model Fairness Metric Hypothesis Testing.** Tests, separately for each model, whether subgroups in a grouping have the same approval rates.
7. **Between Model Fairness Metric Hypothesis Testing.** Tests whether XGBoost and the MLP have the same fairness gaps on the same applicants.
8. **Aleatoric Uncertainty, Epistemic Uncertainty & Label Stability.** Compares predictive uncertainty and decision stability across groups with a bootstrap ensemble of each model.
9. **SHAP Model Interpretability & Feature Attribution.** Explains the original models' predicted default probabilities. SHAP values are in probability units.
   - **9.1 Global Feature Importance.** Ranks feature families by mean absolute SHAP on the full test set.
   - **9.2 Subgroup Feature Importance.** Repeats that ranking inside the subgroups selected above.
   - **9.3 Group Prediction Disparity Decomposition.** Splits the gap in mean predicted default probability between prespecified groups into feature-family contributions. Those contrasts stay fixed.
10. **Exploratory Fairness Mitigation: Fairness Through Unawareness.** Retrains both models without the explicit gender input and compares them with the original models.
    - **10.1 Gender-Blind Model Training.** Fits the same XGBoost and MLP specifications with the gender columns removed.
    - **10.2 Predictive Performance Comparison.** Compares discrimination and classification performance before and after removing gender.
    - **10.3 Fairness Metric Comparison.** Recomputes the Section 4 metrics for the original and gender-blind scores.
    - **10.4 Subgroup and Intersectional Impact.** Compares subgroup approval rates and counts how many individual decisions change.
    - **10.5 Mitigation Effectiveness and Fairness-Performance Trade-offs.** Summarizes the change in the gender gaps and attaches a paired bootstrap interval.

## CSV files

### Source files

- `data.csv` — the Home Credit application table used by every notebook: one row per application, the `TARGET` outcome, and the 121 application fields.
- `feature_description.csv` — the competition's column dictionary: which table a field belongs to and what the field means.

### Tables written by the notebooks

These files are created when the notebooks run and are included in the repository. `output/fairness/cache/` is left out. That folder holds regenerated bootstrap and SHAP arrays, about 58 MB, which the notebooks rebuild when it is missing.

`output/profiling/`

- `schema.csv` — storage type, null count, null rate, and number of distinct values for every column.
- `missingness.csv` — the same null summary, sorted from the most missing field to the least missing field.
- `low_information_fields.csv` — fields with one distinct value, or a most-common value covering at least 99.5% of rows.
- `categorical_summary.csv` — one row per categorical field, with its most common and rarest values.
- `categorical_values.csv` — every category, with its count and its share of applications.
- `numeric_summary.csv` — mean, spread, quartiles, missing rate, and 99th percentile for each numeric field.
- `anomaly_checks.csv` — counts of sentinel and out-of-range values, including the employment code `365243`.
- `missingness_by_gender.csv` — null rates of seven often-missing fields by gender.
- `exclusion_summary.csv` — how many rows have gender `XNA`, family status `Unknown`, or either code.
- `group_gender.csv`, `group_age_band.csv`, `group_income_band.csv`, `group_education.csv`, `group_family_status.csv` — group size, default count, default rate, and share of the file.

`output/data_quality/`

- `missingness_by_field.csv` — fields with at least one missing value, counting the two not-employed codes as missing.
- `missingness_rates_by_gender.csv`, `missingness_rates_by_age_band.csv`, `missingness_rates_by_income.csv`, `missingness_rates_by_default.csv` — missing rate of each field inside each group.
- `missingness_chi_square.csv` — chi-square tests of missingness against gender, age, income, and default.
- `missingness_logit.csv` — logistic regressions for the association between those groups and a missing value.
- `mcar_field_selection.csv` — which numeric fields enter Little's MCAR test, and which are left out.
- `littles_mcar_test.csv` — the MCAR test statistic, degrees of freedom, and p-value.

`output/audit/outlier_winsorization/`

- `outlier_prevalence.csv` — how many observed values exceed each training cap, and how far the maximum sits beyond it.
- `capping_by_gender.csv`, `capping_by_age.csv`, `capping_by_income.csv` — share of observed values above the cap inside each demographic group.
- `monetary_capping_by_split.csv` — how often the monetary caps bind in training, validation, and test.
- `default_rates_by_cap.csv` — default rate above each cap and below it.
- `ratio_sensitivity_to_caps.csv` — how much the four credit ratios change when the monetary inputs are capped.

`output/audit/split_representation/`

- `representation_by_gender.csv`, `representation_by_age.csv`, `representation_by_income.csv` — each group's share of the training, validation, and test sets.
- `default_rates_by_gender.csv`, `default_rates_by_age.csv`, `default_rates_by_income.csv` — each group's default rate in the three sets.
- `overall_default_by_split.csv` — the overall default rate in each set.
- `test_set_support.csv` — an earlier count of test-set group size and defaults. The current notebooks do not rewrite this file.

`output/model/`

- `test_metrics.csv` — test-set performance of the fitted XGBoost and MLP models.

`output/fairness/01_reconstruct/`

- `sample_split.csv` — training, validation, and test sizes after the two exclusion rules.

`output/fairness/02_performance/`

- `overall_metrics.csv` — validation and test performance of both saved models at the chosen thresholds.

`output/fairness/03_population/`

- `exclusion_summary.csv` — rows removed for gender `XNA` or family status `Unknown`.
- `missingness_by_gender.csv` — null rates of key fields by gender in the audited population.
- `population_summary.csv` — test-set size, default rate, and support for every subgroup.
- `intersection_inventory.csv` — which combined groups are large enough to report.

`output/fairness/04_fairness_metrics/`

- `headline_fairness_metrics.csv` — demographic parity, equal opportunity, and equalized odds for each model and grouping.
- `group_level_fairness_metrics.csv` — selection rate, true positive rate, and false positive rate inside each subgroup.

`output/fairness/05_statistical_uncertainty/`

- `overall_fairness_metrics_ci.csv` — bootstrap point estimates and 95% intervals for the headline fairness metrics.
- `group_confidence_intervals.csv` — the same intervals for subgroup rates.
- `wide_confidence_intervals.csv` — intervals wider than the reporting cutoff.
- `wide_interval_metric_counts.csv`, `wide_interval_grouping_counts.csv` — how those wide intervals are spread across metrics and groupings.

`output/fairness/06_within_model_tests/`

- `overall_within_model_tests.csv` — tests of whether a model's subgroups share the same overall approval rates.
- `hypothesis_tests.csv` — the subgroup pairwise tests.
- `hypothesis_test_rejections.csv` — which of those tests reject equal rates.
- `hypothesis_significant_share.csv` — the share of tests that reject, by model and grouping.

`output/fairness/07_between_model_tests/`

- `overall_between_model_tests.csv` — tests of whether XGBoost and the MLP have the same fairness gaps.
- `subgroup_between_model_tests.csv` — the same comparison inside subgroups.
- `subgroup_between_model_h1_rate.csv` — how often the between-model comparison favors one model.

`output/fairness/08_predictive_uncertainty/`

- `subgroup_predictive_uncertainty.csv` — aleatoric uncertainty, epistemic uncertainty, and label stability by subgroup.
- `overall_uncertainty_gaps.csv` — the largest gap in each uncertainty measure across a grouping.
- `aleatoric_uncertainty_gap.csv`, `epistemic_uncertainty_gap.csv`, `label_stability_gap.csv` — those gaps in long form.
- `applicant_uncertainty.csv` — applicant-level uncertainty from the bootstrap ensemble.
- `ensemble_size_stability.csv` — how stable the uncertainty estimates are as more bootstrap models are added.

`output/fairness/09_shap/`

- `shap_sample_coverage.csv` — how many test applicants receive a SHAP explanation.
- `global_shap_feature_importance.csv` — mean absolute SHAP by feature family for each model.
- `global_shap_encoded.csv` — the same importance before one-hot columns are grouped into families.
- `gender_age_income_shap.csv` — SHAP for the gender, age, and income families.
- `subgroup_shap_feature_importance.csv` — feature-family importance inside each subgroup.
- `demographic_feature_attribution.csv` — attribution focused on the demographic features.
- `group_prediction_contrasts.csv` — the prespecified group pairs whose prediction gap is decomposed.
- `group_prediction_disparity_decomposition.csv` — how much each feature family contributes to that gap.

`output/fairness/10_mitigation/`

- `predictive_performance.csv` — test performance of the original and gender-blind models.
- `predictive_performance_change.csv` — gender-blind performance minus original performance.
- `fairness_metric_comparison.csv` — Section 4 metrics before and after removing gender.
- `fairness_tally.csv` — counts of fairness results under the fixed threshold.
- `subgroup_fairness_impact.csv` — subgroup approval and error rates before and after removing gender.
- `individual_decision_changes.csv` — how many applicants stay approved, stay declined, or switch.
- `mitigation_effectiveness.csv` — the change in the gender gaps.
- `paired_bootstrap_intervals.csv` — paired bootstrap intervals for those changes.

`output/fairness/cache/` is not in the repository. Locally it holds cached test scores and group labels, plus the bootstrap and SHAP arrays later sections reuse.

## Environment

Python dependencies are listed in `requirements.txt`. The notebooks expect `data.csv` in the project folder and write their tables and figures under `output/`.
