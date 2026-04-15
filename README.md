# Mortality & Longevity Analysis — French Insurance Market 📈

> **Exploratory data analysis and statistical modeling** of mortality trends and longevity risk in the French insurance market.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-notebooks-orange)](https://jupyter.org)
[![Statsmodels](https://img.shields.io/badge/statsmodels-0.14-green)](https://statsmodels.org)
[![CI](https://github.com/Abdousurf/mortality-longevity-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/Abdousurf/mortality-longevity-analysis/actions)
[![DVC](https://img.shields.io/badge/DVC-versioned-945dd6)](https://dvc.org)
[![Open Data](https://img.shields.io/badge/Open%20Data-INSEE%20%7C%20WHO%20%7C%20HMD-blue)](https://www.insee.fr/fr/statistiques/7635678)

## Overview

This project applies actuarial and statistical methods to analyze **mortality and longevity trends** — a critical risk for life insurers, pension funds, and annuity providers.

Longevity risk = people living *longer than expected* → annuity/pension liabilities increase beyond projections.

This analysis covers:
1. **EDA** — French population mortality data (1968–2022)
2. **Mortality Table Construction** — from raw experience data
3. **Lee-Carter Model** — stochastic mortality projection
4. **Longevity Stress Testing** — Solvency II perspective
5. **Business Insights** — impact on annuity reserves

## Key Findings

- Life expectancy at 65 increased by **+4.2 years** for men and **+3.8 years** for women since 1990
- Lee-Carter projections suggest **0.8–1.2% annual improvement** in mortality rates
- A 1-year longevity shock increases annuity liabilities by approximately **3.5–4.5%**
- Significant **cohort effects** for generations 1940–1950 (post-war healthcare)
- COVID-19 caused a temporary **+8.3% excess mortality** in 2020, partially reversing in 2021

## Methodology

```
Raw Data (INSEE/HMD)
        │
        ▼
  Data Cleaning & QC
  ─ Age/period completeness
  ─ Outlier detection (wars, pandemics)
        │
        ▼
  Mortality Table Construction
  ─ qx (probability of death)
  ─ Graduated tables (Whittaker-Henderson)
  ─ Comparison vs TF00-02 reference table
        │
        ▼
  Lee-Carter Model
  ─ SVD decomposition: ln(μx,t) = αx + βx·κt
  ─ κt projection: ARIMA(0,1,0)
  ─ Confidence intervals via bootstrap
        │
        ▼
  Longevity Risk Quantification
  ─ VaR 99.5% (Solvency II)
  ─ Annuity reserve sensitivity
  ─ Trend scenario analysis
        │
        ▼
  Visualizations & Report
  ─ Lexis diagram
  ─ Period vs cohort life tables
  ─ 25-year mortality fan charts
```

## Open Data Sources

| Source | Dataset | Licence | Script |
|--------|---------|---------|--------|
| **INSEE** (data.gouv.fr) | Tables de mortalité France 1994–2022 | Licence Ouverte Etalab v2.0 | `src/download_hmd_data.py` |
| **WHO GHO** (API REST) | Surmortalité COVID-19 France | CC BY-NC-SA 3.0 IGO | `src/download_hmd_data.py` |
| **HMD** (mortality.org) | Données historiques France 1816–2022 | Free / inscription requise | `src/download_hmd_data.py` |
| **INSEE Population** | Population France par âge 1975–2023 | Licence Ouverte Etalab v2.0 | `src/download_hmd_data.py` |

Toutes les sources sont accessibles gratuitement. La CI utilise un fallback synthétique (loi Makeham-Gompertz calibrée France) si les sources sont inaccessibles.

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python + Pandas | Data manipulation |
| NumPy + SciPy | Statistical computing |
| Statsmodels | ARIMA, regression |
| Matplotlib + Seaborn | Visualization |
| Plotly | Interactive charts |
| Jupyter | Analysis notebooks |
| **Papermill** | **Exécution automatisée des notebooks (CI/CD)** |
| **Pandera** | **Validation actuarielle des données (DataOps)** |
| **DVC** | **Versionnage données + reproductibilité** |
| **GitHub Actions** | **CI/CD : téléchargement → validation → exécution notebooks → rapport HTML** |
| **pre-commit + ruff** | **Qualité code + nbstripout (outputs supprimés avant commit)** |

## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 01 | `eda_mortality_france.ipynb` | Population mortality EDA, Lexis diagrams |
| 02 | `mortality_table_construction.ipynb` | Building qx tables from raw experience data |
| 03 | `lee_carter_model.ipynb` | Lee-Carter calibration and projection |
| 04 | `longevity_stress_testing.ipynb` | Solvency II longevity shock, reserve impact |
| 05 | `covid_excess_mortality.ipynb` | COVID-19 mortality analysis |

## Sample Outputs

### Mortality Improvement (1990–2022)

```
Age   | qx (1990) | qx (2022) | Improvement
65    |  0.01823  |  0.00891  |   -51.1%
70    |  0.03142  |  0.01537  |   -51.1%
75    |  0.05801  |  0.02943  |   -49.3%
80    |  0.10234  |  0.05621  |   -45.1%
```

### Lee-Carter Projection (2025–2050)

```
Year  | e₀(M)  | e₀(F)  | 95% CI (M)
2025  | 80.2   | 85.8   | [79.1, 81.3]
2030  | 81.1   | 86.5   | [79.4, 82.8]
2040  | 82.7   | 87.8   | [80.1, 85.3]
2050  | 84.1   | 89.0   | [80.8, 87.4]
```

## Getting Started

```bash
git clone https://github.com/Abdousurf/mortality-longevity-analysis
cd mortality-longevity-analysis
pip install -r requirements.txt
pre-commit install

# Téléchargement données open data (INSEE + WHO)
make data-download

# Pipeline DataOps complet (download → validation → notebooks → rapport)
make pipeline

# Ou en une commande
make pipeline-ci  # version rapide pour CI (données démo sans téléchargement)

# Lancer Jupyter Lab
make lab
```

## Who Is This For?

- **Actuaries** building mortality assumptions for Solvency II reserves
- **Data analysts** in life insurance, pension funds, annuity providers
- **Quants** modeling longevity-linked products (survivor bonds, longevity swaps)

---

*Written by a consultant with 10+ years in actuarial science and data analytics.*
*[LinkedIn](https://www.linkedin.com/in/abdou-john/)*
