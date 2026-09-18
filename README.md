# P&C Actuarial Pricing Project — freMTPL Frequency GLM

## Overview

A ratemaking/pricing project using the **freMTPL** (French Motor Third-Party Liability) dataset to build a claim frequency model, following the standard actuarial GLM pricing workflow. Built in Python (pandas, statsmodels, seaborn/matplotlib), with an Excel deliverable planned as a final output.

This project is scoped to **pricing/ratemaking only** — no reserving component (no loss triangles, no development, no IBNR).

## Data

- **Source:** freMTPL frequency table (`freq`), joined by `PolicyID` with the severity table (not yet used — severity modeling has not started).
- **Key fields used so far:** `ClaimNb` (target — claim count), `Exposure` (policy-years, used as a model offset), `BonusMalus`, `DrivAge`, `VehAge`, `Region`.

## Assumptions and Data-Quality Decisions

- No policy-level exposure or claim-count exclusions/caps have been formally applied yet; known freMTPL quirks (e.g., `Exposure > 1` values, high claim counts relative to exposure) were noted during EDA but not yet cleaned/capped in code.
- `pd.qcut` was used for binning continuous variables. For skewed variables (`BonusMalus` especially, also `Density`), `duplicates='drop'` caused fewer bins than requested (e.g., 10 requested → 5 actual for `BonusMalus`) due to a large mass of policies at the minimum value. This was noted but not resolved with custom bin edges.
- Bin edges for `DrivAge` and `VehAge` were derived from the **training set only** (via `qcut` on `train`, then applied to `test` with `pd.cut` using the same edges) to avoid test-set leakage.
- Train/test split: 80/20 random split, `random_state=42`, no stratification.

## Exploratory Data Analysis (Completed)

- **Continuous variables** (`DrivAge`, `VehAge`, `Density`, `BonusMalus`): exposure and frequency plotted by decile bucket.
  - `BonusMalus`: strong, roughly monotonic increasing relationship with frequency (collapsed to 5 bins due to skew — see assumptions above).
  - `Density`: monotonic increasing relationship with frequency, full 10 bins.
  - `DrivAge`: non-monotonic — frequency rises through mid-age brackets, peaks around ages 48–57, then eases off in the oldest bracket.
  - `VehAge`: largely monotonic decreasing — highest frequency for the newest vehicles (0–2 years), sharp drop after, roughly flat through 4–12 years, sharp drop again for 12+ years.
- **Discrete/categorical variables** (`VehPower`, `Region`, `VehGas`, `VehBrand`): exposure and frequency plotted by category (function written; full review of outputs across all four not yet walked through in detail).
- **Overdispersion check:** claim frequency mean ≈ 0.0532, variance ≈ 0.0577 (ratio ≈ 1.08) — mild overdispersion. Judged acceptable for a standard Poisson GLM (point estimates unaffected; standard errors may be mildly underestimated).
- **Zero-inflation check:** actual proportion of zero-claim policies ≈ 94.98%, vs. Poisson-implied ≈ 94.8% — negligible gap, no meaningful zero-inflation.
- **Interaction check:** `DrivAge × BonusMalus` cross-tab (4x2 heatmap, after `BonusMalus` collapsed from requested 4 bins). Found only a mild interaction (BonusMalus effect slightly stronger for the oldest driver bracket) — judged not material enough to include as an interaction term. Decision made to proceed with an **additive model** and revisit interactions later via residual analysis if needed, rather than testing further variable pairs.

## Frequency GLM (In Progress)

Built incrementally in statsmodels (`smf.glm`), Poisson family, log link, with `log(Exposure)` as an **offset** (not a predictor) in every model version.

**Variable-by-variable build:**

1. `BonusMalus` (linear) + `DrivAge` (linear) — baseline model. Both coefficients positive and significant. `DrivAge`'s linear term flagged as a likely mis-specification given its non-monotonic EDA shape.
2. `DrivAge` re-entered as **binned** (5 quintile bins, edges from train only) in place of linear — deviance and AIC both improved (deviance 175,720 → 175,440; confirmed the rise-then-dip pattern directly in the bin coefficients, peaking in the 48–57 bracket).
3. `VehAge` added, tested linear then binned (4 quintile-style bins). Binned version improved fit again (deviance 174,200 → 173,920; AIC 229,599 → 229,337), confirming a non-linear relationship (highest frequency for newest vehicles, declining with age).
4. `Region` added as `C(Region)` (categorical, ~21 levels). AIC improved slightly (229,337 → 229,235; deviance 173,920 → 173,780), but only 1 of ~20 region coefficients (`Auvergne`) was statistically significant at p < 0.05. Decision: keep `Region` in for now given the AIC improvement, but flag that a grouped or credibility-weighted treatment of region would likely be more robust than 21 separate categorical levels for a production rating plan.

**Current model formula:**
`ClaimNb ~ BonusMalus + C(DrivAgeBin) + C(VehAgeBin) + C(Region)`, offset = `log(Exposure)`, fit on the 80% training split.

**General finding across variables:** continuous rating variables in this dataset tend to have non-monotonic relationships with frequency; binning has consistently outperformed raw linear terms (true for both `DrivAge` and `VehAge`).

## Validation (In Progress)

- Reusable `liftChart()` function written: predicts on the **test set** (using `test`-derived offset, never train), sorts policies into predicted-risk deciles, and compares actual vs. predicted frequency per decile.
- **Bug identified and not yet fixed:** the first version of the lift chart sorted deciles by predicted **claim count** rather than predicted **frequency**, which conflated risk ranking with exposure length and produced an inverted/incorrect-looking chart (actual frequency highest in the "lowest risk" decile). Fix identified: divide predicted count by exposure to get predicted frequency before binning into deciles. Not yet re-run with the fix.
- Gini coefficient / Lorenz curve analysis: not yet started.

## Not Yet Started

- Severity modeling (no claim-amount modeling yet; only frequency).
- Pure premium / rate relativity calculation combining frequency × severity.
- Cleaning/capping of known data-quality issues (`Exposure > 1`, outlier claim counts).
- Full review of categorical variable EDA (`VehPower`, `VehGas`, `VehBrand`).
- Fixing and re-running the lift chart with predicted frequency (not count).
- Gini coefficient calculation.
- Excel deliverable (rate manual-style workbook with relativity tables and a rate calculator).
- Model documentation/writeup of final findings.
