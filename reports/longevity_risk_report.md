# Longevity Risk Report — French Insurance Market

**Project:** Mortality & Longevity Analysis — France  
**Author:** Actuarial Data Analytics  
**Date:** April 2026  
**Model:** Lee-Carter (1992) — French HMD/INSEE data 1968–2022

---

## Executive Summary

This report synthesises the key findings from five analysis notebooks on French mortality trends and longevity risk quantification. The analysis covers the period 1968–2022 using open data from INSEE, HMD, and WHO, calibrated for the French insurance market.

**Bottom line:** Life expectancy at 65 has increased by +4.2 years for men since 1990 and continues to improve at 0.8–1.2% per year. Under Solvency II, a 20% permanent longevity shock increases annuity reserves by 3.5–4.5%.

---

## 1. Mortality Trends (1968–2022)

### 1.1 Observed Improvements

Mortality rates at key ages have fallen sharply since 1990:

| Age | qx (1990) | qx (2022) | Improvement |
|-----|-----------|-----------|-------------|
| 65  | 0.01823   | 0.00891   | **-51.1%**  |
| 70  | 0.03142   | 0.01537   | **-51.1%**  |
| 75  | 0.05801   | 0.02943   | **-49.3%**  |
| 80  | 0.10234   | 0.05621   | **-45.1%**  |

*Source: Notebooks 01 (EDA) and 02 (Mortality Table Construction)*

### 1.2 Life Expectancy at Age 65

- **Men**: +4.2 years since 1990 (from ~16.0 to ~20.2 additional years)
- **Women**: +3.8 years since 1990 (from ~20.0 to ~23.8 additional years)

The gender gap in e₆₅ has narrowed slightly from ~4 years (1990) to ~3.5 years (2022), driven by faster male improvement.

### 1.3 Cohort Effects

Generations 1940–1950 show accelerated mortality improvement relative to the period trend. This is attributed to:
- Post-war improvements in healthcare access
- Anti-smoking campaigns effective for this cohort
- Improved cardiovascular care in the 1990s

---

## 2. Lee-Carter Model Results

### 2.1 Model Fit (1968–2022)

The Lee-Carter model was calibrated on the French HMD mortality surface (ages 40–100, years 1968–2022).

**SVD decomposition quality:**
- First principal component explains >95% of mortality variance
- β_x: peaks at ages 65–80, reflecting highest sensitivity to time trend at retirement ages
- κ_t: clear downward trend of ~-0.8 to -1.2 per year (= mortality improvement)

**ARIMA(0,1,0) fit on κ_t:**
- Drift: ~-0.9 to -1.1 per year
- Volatility (σ): 0.3–0.5 per year
- Implied annual mortality improvement: **0.8–1.2%** at ages 65–80

### 2.2 25-Year Projections (2025–2050)

| Year | e₀ Males | e₀ Females | 95% CI (Males) |
|------|----------|------------|----------------|
| 2025 | 80.2     | 85.8       | [79.1, 81.3]   |
| 2030 | 81.1     | 86.5       | [79.4, 82.8]   |
| 2040 | 82.7     | 87.8       | [80.1, 85.3]   |
| 2050 | 84.1     | 89.0       | [80.8, 87.4]   |

*Source: Notebook 03 (Lee-Carter Model)*

### 2.3 Model Limitations

- The standard Lee-Carter model does not capture cohort effects directly (CBD or APC extension recommended)
- ARIMA(0,1,0) projection may underestimate parameter uncertainty for very long horizons
- COVID-19 introduced a temporary structural break in 2020 (see Section 4)

---

## 3. Solvency II Longevity Stress Testing

### 3.1 Regulatory Framework

Under the Solvency II Delegated Acts (Article 142), the longevity module applies a **permanent 20% reduction** in age-specific mortality rates. This is the VaR 99.5% calibration.

### 3.2 Reserve Impact

| Configuration | Reserve Increase |
|---------------|-----------------|
| Males, age 65, i=3% | **~3.9%** |
| Females, age 65, i=3% | **~4.2%** |
| Blended 50/50, age 65, i=3% | **~4.1%** |
| Males, age 65, i=2% | **~5.1%** |

**Business context:** For a €100M annuity book, the longevity SCR requires **€3.5M–4.5M** of additional own funds.

### 3.3 Sensitivity Analysis

Key drivers of the longevity SCR:
1. **Discount rate** (inverse relationship — lower rates → higher SCR)
2. **Portfolio age** (younger policyholders → higher duration → larger impact)
3. **Shock magnitude** (10% shock ≈ 2% reserve impact; 30% shock ≈ 6%)

*Source: Notebook 04 (Longevity Stress Testing)*

---

## 4. COVID-19 Excess Mortality

### 4.1 France 2020 Impact

COVID-19 caused a **temporary +8.3% excess mortality** in France in 2020:
- Total excess deaths: ~56,000 above the 2010–2019 trend baseline
- Most affected: ages 65–84 (contributing ~65% of total excess)
- France ranked in the middle tier for European COVID mortality

### 4.2 Recovery Trajectory

- 2021: partial rebound (-2.1pp vs 2020 excess), some mortality displacement effect
- 2022: near-normalisation; trend improvement resumed
- Long-term Lee-Carter κ_t trajectory not structurally disrupted

### 4.3 Actuarial Implications

For Solvency II reserving purposes:
- COVID is treated as a **transient shock** under the standard formula
- No permanent adjustment to best-estimate mortality assumptions required
- Monitoring recommended for cohort 1940–1960 (elevated post-COVID mortality risk)

*Source: Notebook 05 (COVID Excess Mortality)*

---

## 5. Key Risk Metrics Summary

| Risk Factor | Metric | Value |
|-------------|--------|-------|
| Improvement trend | Annual qx improvement (ages 65–80) | **0.8–1.2%/yr** |
| Longevity risk | e₆₅ increase since 1990 (males) | **+4.2 years** |
| Longevity risk | e₆₅ increase since 1990 (females) | **+3.8 years** |
| SII stress | Reserve increase (20% shock, age 65) | **3.5–4.5%** |
| SII stress | SCR per €100M annuity book | **~€3.5–4.5M** |
| COVID shock | 2020 excess mortality France | **+8.3%** |
| Model uncertainty | 95% CI width for e₀ by 2050 | **±3.3 years** |

---

## 6. Recommendations

1. **Use cohort-adjusted projection** (CBD or LC+cohort) for portfolios with high exposure to 1940–1955 cohorts.
2. **Stress-test at 30% shock** for internal capital models exceeding the SII standard formula.
3. **Monitor κ_t annually**: if drift accelerates beyond -1.2/yr, revise best estimate assumptions.
4. **COVID normalisation confirmed** by 2022–2023 data; no structural change to trend required.
5. **Gender-specific tables** are material: applying unisex tables underestimates female longevity risk by ~0.3pp.

---

## Data Sources

| Source | Dataset | Period | Licence |
|--------|---------|--------|---------|
| INSEE | Tables de mortalité France | 1994–2022 | Licence Ouverte Etalab v2.0 |
| HMD | France mortality (FRA) | 1968–2022 | Free / registration |
| WHO GHO | Excess mortality France | 2020–2022 | CC BY-NC-SA 3.0 IGO |

All scripts, notebooks and data pipelines are reproducible via `make pipeline`. See `README.md` for setup instructions.
