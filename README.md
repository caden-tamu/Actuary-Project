# P&C Actuarial Pricing Project — freMTPL Frequency GLM

## Overview

A ratemaking/pricing project using the **freMTPL** (French Motor Third-Party Liability) dataset to build and validate a claim frequency model, following the standard actuarial GLM pricing workflow. Built in Python (pandas, statsmodels, seaborn/matplotlib), with a simplified Excel relativity-table deliverable as the final output.

This project is scoped to **pricing/ratemaking, frequency only**:
- No reserving component (no loss triangles, development, or IBNR).
- No severity modeling or pure premium calculation — this is a deliberate scoping decision, made to keep the project finishable, not an oversight. Noted below as a natural extension for future work.

## Data

- **Source:** freMTPL frequency table (`freq`), joined by `PolicyID` with the severity table. The severity table is **not used** — see scope note above.
- **Target:** `ClaimNb` (claim count per policy).
- **Offset:** `log(Exposure)` (policy-years) — used as a GLM offset, not a predictor, so the model predicts a *rate* rather than a raw count.
- **Final model variables:** `BonusMalus`, `DrivAge` (binned), `VehAge` (binned), `Region`, `VehGas`, `VehPower` (linear).
- **Not used in the final model:** `Density` (explored in EDA, not added to the model), `VehBrand` (skipped — see reasoning below).

## Assumptions and Data-Quality Decisions

- No policy-level exposure or claim-count exclusions/caps have been applied. Known freMTPL quirks (e.g., `Exposure > 1` values, high claim counts relative to exposure) were identified during EDA but intentionally left unaddressed, as a documented limitation rather than a cleaning step, to keep the project scope manageable.
- `pd.qcut` was used for binning continuous variables. For skewed variables (`BonusMalus` especially, also `Density`), `duplicates='drop'` caused fewer bins than requested (e.g., 10 requested → 5 actual for `BonusMalus`) due to a large mass of policies at the minimum value. Noted but not resolved with custom bin edges.
- Bin edges for `DrivAge` and `VehAge` were derived from the **training set only** (via `qcut` on `train`, then applied to `test` with `pd.cut` using the same edges) to avoid test-set leakage.
- Train/test split: 80/20 random split, `random_state=42`, no stratification.

## Exploratory Data Analysis

- **Continuous variables** (`DrivAge`, `VehAge`, `Density`, `BonusMalus`): exposure and frequency plotted by decile bucket.
  - `BonusMalus`: strong, roughly monotonic increasing relationship with frequency (collapsed to 5 bins due to skew — see assumptions above).
  - `Density`: monotonic increasing relationship with frequency, full 10 bins. Explored but not carried into the final model.
  - `DrivAge`: non-monotonic — frequency rises through mid-age brackets, peaks around ages 48–57, then eases off in the oldest bracket.
  - `VehAge`: largely monotonic decreasing — highest frequency for the newest vehicles (0–2 years), sharp drop after, roughly flat through 4–12 years, sharp drop again for 12+ years.
- **Discrete/categorical variables** (`VehPower`, `Region`, `VehGas`, `VehBrand`): exposure and frequency plotted by category.
  - `VehBrand` was not added to the model — given `Region`'s result (below), a similarly high-cardinality categorical with mostly insignificant individual levels was judged unlikely to add meaningful value, and was deprioritized to keep scope manageable.
- **Overdispersion check:** claim frequency mean ≈ 0.0532, variance ≈ 0.0577 (ratio ≈ 1.08) — mild overdispersion. Judged acceptable for a standard Poisson GLM (point estimates unaffected; standard errors may be mildly underestimated).
- **Zero-inflation check:** actual proportion of zero-claim policies ≈ 94.98%, vs. Poisson-implied ≈ 94.8% — negligible gap, no meaningful zero-inflation.
- **Interaction check:** `DrivAge × BonusMalus` cross-tab (4x2 heatmap, after `BonusMalus` collapsed from requested 4 bins). Found only a mild interaction (BonusMalus effect slightly stronger for the oldest driver bracket) — judged not material enough to include as an interaction term. Decision made to proceed with an **additive model** rather than testing further variable pairs.

## Frequency GLM — Final Model

Built incrementally in statsmodels (`smf.glm`), Poisson family, log link, with `log(Exposure)` as an offset in every version. Each variable was tested and compared against the prior model version using deviance, AIC, and (once validation was built) Gini, before being kept.

**Build sequence and findings:**

1. `BonusMalus` (linear) + `DrivAge` (linear) — baseline. Both significant; `DrivAge`'s linear term flagged as likely mis-specified given its non-monotonic EDA shape.
2. `DrivAge` re-entered as **binned** (5 quintile bins, edges from train only) — deviance and AIC improved (175,720 → 175,440), directly confirming the rise-then-peak-then-dip pattern in the bin coefficients (peak in the 48–57 bracket).
3. `VehAge` added, tested linear then binned — binned version improved fit again (deviance 174,200 → 173,920; AIC 229,599 → 229,337), confirming a non-linear, largely declining relationship with vehicle age.
4. `Region` added as `C(Region)` (~21 levels). AIC improved modestly (229,337 → 229,235), but only 1 of ~20 region coefficients (`Auvergne`) was significant at p < 0.05. Kept in given the AIC improvement, but flagged as a candidate for grouping or credibility-weighting in a production setting rather than 21 separate levels.
5. `VehGas` added as `C(VehGas)` (binary). Significant (p < 0.001, ~8% higher frequency for `Regular` vs. `Diesel`), modest AIC/deviance improvement for a single parameter. Gini barely moved (0.246 → 0.2468) — statistically real but low practical discriminatory value. Kept due to low complexity cost.
6. `VehPower` tested three ways:
   - **Linear:** significant (p < 0.001), modest AIC improvement.
   - **Full categorical** (`C(VehPower)`, one coefficient per power level): larger AIC improvement, but coefficients for high-power levels (12–15) were statistically insignificant with wide standard errors — a sign of overfitting to sparse, low-exposure categories rather than a real effect.
   - **Decision:** kept **linear**, prioritizing model stability and simplicity over the marginal, likely-overfit AIC gain from full categorical treatment.

**Final model formula:**
```
ClaimNb ~ BonusMalus + C(DrivAgeBin) + C(VehAgeBin) + C(Region) + C(VehGas) + VehPower
```
Offset: `log(Exposure)`. Fit on the 80% training split.

**General finding across variables:** continuous rating variables in this dataset tend to have non-monotonic relationships with frequency; binning outperformed raw linear terms for both `DrivAge` and `VehAge`. By contrast, high-cardinality categorical and fine-grained ordinal variables (`Region`, `VehPower`) showed diminishing or overfitting-prone returns from finer treatment, favoring simpler specifications.

## Validation

- Reusable `liftChart()` and `giniCoefficient()` functions built, both predicting on the **held-out test set** (never train), using a test-derived offset.
- **Bug found and fixed:** the first lift chart version sorted deciles by predicted **claim count** rather than predicted **frequency**, which conflated risk ranking with exposure length and produced an inverted-looking chart. Fixed by dividing predicted count by exposure before binning into deciles.
- **Final lift chart:** actual and predicted frequency rise together across all 10 deciles in consistent order, with close agreement through most of the range and a slight overprediction in the highest-risk decile (expected — more variance at the tails, less supporting data for extreme predictions).
- **Final Gini coefficient: 0.2471** — within the typical 0.2–0.4 range for real-world motor frequency models, including published examples using this same dataset. Represents genuine, validated discriminatory power, not just a training-fit artifact.
- Variable-by-variable Gini tracking during the build (`VehGas`: 0.246 → 0.2468; `VehPower`: 0.2468 → 0.2471) showed that AIC/deviance improvements didn't always translate to meaningful gains in out-of-sample discrimination — an explicit, documented finding, not just a footnote.

## Scope Decisions (Simplification)

Made deliberately partway through the project to keep it finishable and focused, rather than open-ended:

- **Frequency-only model** — no severity modeling, no pure premium/rate relativity combining frequency × severity.
- **`VehBrand` not modeled** — deprioritized based on `Region`'s result (high-cardinality categoricals showing weak individual significance here).
- **Data-quality cleanup not applied** — known issues documented as a limitation rather than fixed in code.
- **`VehPower` kept linear**, not full categorical, to avoid overfitting sparse categories and to keep the model simple and stable.
- **Excel deliverable simplified** to a relativity table (variable, level, relativity = `exp(coefficient)`) with the final lift chart and Lorenz curve embedded, rather than a full interactive rate-calculator workbook.

## Not Yet Started / Future Work

- Severity modeling (Gamma/lognormal GLM on claim amount, conditional on a claim occurring) and pure premium calculation.
- Grouped or credibility-weighted treatment of `Region` (and potentially `VehBrand`) in place of raw high-cardinality categorical terms.
- Formal cleaning/capping of known data-quality issues (`Exposure > 1`, outlier claim counts) and re-validation of the model on cleaned data.
- Excel relativity table and embedded validation charts (in progress).
