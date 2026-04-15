"""Lee-Carter mortality model notebook for France (1968-2022).

Implements a complete Lee-Carter mortality modeling workflow including
model calibration on French HMD data, stochastic mortality projections,
life expectancy analysis, and Solvency II longevity stress testing.

Execute as a script or open with Jupytext/Jupyter.

Covers:
    - Lee-Carter calibration on French HMD data.
    - kappa_t time index projection (ARIMA random walk).
    - 25-year mortality fan chart.
    - Life expectancy at 65 projection.
    - Comparison: males vs females.
"""

# %% [markdown]
# # Lee-Carter Mortality Model — France (1968–2022)
#
# The Lee-Carter model (1992) is the **industry standard** for stochastic mortality projection.
# It decomposes log-mortality into:
# - **α_x**: age profile (average mortality level)
# - **β_x**: age sensitivity to time trend
# - **κ_t**: time index (driven by improvements in medicine, lifestyle)
#
# Projection: κ_t follows a **random walk with drift** → uncertainty fans out over time

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from pathlib import Path
import sys
sys.path.insert(0, str(Path("..").resolve()))

from src.mortality_tables import LeeCarter, build_life_table, longevity_shock_impact

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120, "font.size": 11})

# %% [markdown]
# ## 1. Load HMD Data
#
# Source: Human Mortality Database (France)
# Register free at mortality.org

# %%
def load_hmd_france(data_dir: Path, gender: str = "Male") -> pd.DataFrame:
    """Load HMD France death rates as an ages-by-years matrix.

    Reads the HMD mortality rates file for the specified gender, filters
    to ages 40-100 and years 1968-2022, and pivots into matrix format.

    Args:
        data_dir: Path to the directory containing raw HMD data files.
        gender: Sex to load, either 'Male' or 'Female'.

    Returns:
        DataFrame with ages as index and years as columns, containing
        central death rates (Mx).
    """
    suffix = "m" if gender == "Male" else "f"
    filepath = data_dir / f"FRA.Mx_{suffix}x1.txt"

    df = pd.read_csv(filepath, sep=r"\s+", skiprows=2, na_values=".")
    df = df[df["Age"] != "110+"].copy()
    df["Age"] = df["Age"].astype(int)
    df = df[df["Age"].between(40, 100)]
    df = df[df["Year"].between(1968, 2022)]

    # Pivot to matrix format: index=Age, columns=Year
    mx_matrix = df.pivot(index="Age", columns="Year", values="Total")
    mx_matrix = mx_matrix.astype(float).fillna(method="ffill", axis=1)
    return mx_matrix


DATA_DIR = Path("../data/raw")

try:
    mx_male = load_hmd_france(DATA_DIR, "Male")
    mx_female = load_hmd_france(DATA_DIR, "Female")
    print(f"Data loaded: ages {mx_male.index.min()}–{mx_male.index.max()}, "
          f"years {mx_male.columns.min()}–{mx_male.columns.max()}")
except FileNotFoundError:
    # Generate synthetic data if HMD not downloaded yet
    print("HMD data not found — generating synthetic mortality surface...")
    ages = np.arange(40, 101)
    years = np.arange(1968, 2023)
    # Gompertz baseline + linear improvement
    A, B = 0.0001, 0.1
    improvement_rate = 0.015  # 1.5% annual improvement
    mx_vals = np.outer(
        A * np.exp(B * (ages - 40)),
        np.exp(-improvement_rate * (years - 1968))
    )
    mx_male = pd.DataFrame(mx_vals, index=ages, columns=years)
    mx_female = pd.DataFrame(mx_vals * 0.65, index=ages, columns=years)  # women live longer

# %% [markdown]
# ## 2. Fit Lee-Carter Model

# %%
lc_male = LeeCarter().fit(mx_male)
lc_female = LeeCarter().fit(mx_female)

print("Lee-Carter Fit — Males")
print(f"  κ_t range: [{lc_male.kappa.min():.1f}, {lc_male.kappa.max():.1f}]")
print(f"  κ_t drift: {lc_male._kappa_drift:.3f} (annual)")
print(f"  κ_t sigma: {lc_male._kappa_sigma:.3f}")

# %% [markdown]
# ## 3. Plot α_x and β_x Parameters

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

ages = lc_male.ages

axes[0].plot(ages, lc_male.alpha, label="Males", color="steelblue")
axes[0].plot(ages, lc_female.alpha, label="Females", color="coral", linestyle="--")
axes[0].set_title("α_x — Average log-mortality by age")
axes[0].set_xlabel("Age")
axes[0].set_ylabel("ln(μ_x)")
axes[0].legend()

axes[1].plot(ages, lc_male.beta, label="Males", color="steelblue")
axes[1].plot(ages, lc_female.beta, label="Females", color="coral", linestyle="--")
axes[1].set_title("β_x — Age sensitivity to time trend")
axes[1].set_xlabel("Age")
axes[1].set_ylabel("β_x")
axes[1].legend()

years = lc_male.years
axes[2].plot(years, lc_male.kappa, label="Males", color="steelblue")
axes[2].plot(years, lc_female.kappa, label="Females", color="coral", linestyle="--")
axes[2].set_title("κ_t — Mortality time index (declining = improving)")
axes[2].set_xlabel("Year")
axes[2].set_ylabel("κ_t")
axes[2].legend()

plt.tight_layout()
plt.savefig("../visualizations/lee_carter_parameters.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. 25-Year Mortality Projections — Fan Chart

# %%
projection_male = lc_male.project_qx(horizon=25, confidence=0.95, n_sim=2000)
projection_female = lc_female.project_qx(horizon=25, confidence=0.95, n_sim=2000)

# Fan chart at age 70
age_focus = 70

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"Projected Mortality Rate q_{age_focus} — France (2023–2047)", fontsize=13)

for ax, proj, gender, color in zip(
    axes,
    [projection_male, projection_female],
    ["Males", "Females"],
    ["steelblue", "coral"]
):
    central = proj["central"].loc[age_focus]
    lower = proj["lower"].loc[age_focus]
    upper = proj["upper"].loc[age_focus]

    ax.fill_between(central.index, lower, upper, alpha=0.25, color=color, label="95% CI")
    ax.plot(central.index, central, color=color, linewidth=2.5, label="Central estimate")

    # Historical last 10 years
    hist_qx = mx_male.loc[age_focus, -10:] / (1 + 0.5 * mx_male.loc[age_focus, -10:])
    ax.plot(hist_qx.index, hist_qx, color="gray", linestyle="--", alpha=0.7, label="Historical")

    ax.set_title(f"{gender}")
    ax.set_xlabel("Year")
    ax.set_ylabel(f"q_{age_focus}")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2%}"))
    ax.legend()

plt.tight_layout()
plt.savefig(f"../visualizations/mortality_fan_chart_age{age_focus}.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Life Expectancy Projection at Age 65

# %%
e65_male = lc_male.life_expectancy(projection_male["central"], start_age=65)
e65_female = lc_female.life_expectancy(projection_female["central"], start_age=65)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(e65_male.index, e65_male.values, "o-", color="steelblue", label="Males e₆₅")
ax.plot(e65_female.index, e65_female.values, "s--", color="coral", label="Females e₆₅")
ax.set_title("Projected Life Expectancy at Age 65 — France")
ax.set_xlabel("Year")
ax.set_ylabel("Life expectancy (years)")
ax.legend()
ax.grid(True, alpha=0.3)

# Annotations
ax.annotate(f"e₆₅(M) 2047 ≈ {e65_male.iloc[-1]:.1f} yrs",
            xy=(e65_male.index[-1], e65_male.iloc[-1]),
            xytext=(-60, 10), textcoords="offset points",
            fontsize=9, color="steelblue")

plt.tight_layout()
plt.savefig("../visualizations/life_expectancy_projection.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. Solvency II Longevity Stress

# %%
qx_2023_male = projection_male["central"].iloc[:, 0].values
result = longevity_shock_impact(
    qx_base=qx_2023_male,
    shock_factor=0.80,       # SII: 20% permanent reduction
    interest_rate=0.03,
    start_age=65
)

print("\n── Solvency II Longevity Stress (Males, age 65) ─────────────────")
print(f"  Annuity ä₆₅ (base):      {result['annuity_base']:.4f}")
print(f"  Annuity ä₆₅ (stressed):  {result['annuity_stressed']:.4f}")
print(f"  Reserve increase:         +{result['reserve_increase_pct']:.1f}%")
print(f"  SCR proxy per € 1 annuity: {result['scr_proxy_per_unit']:.4f}")
print(f"\n  → A €100M annuity book requires ~€{result['reserve_increase_pct']:.0f}M "
      f"additional capital under longevity shock")

# %% [markdown]
# ## 7. Summary Table — Mortality Improvement by Age Cohort

# %%
ages_report = [50, 60, 65, 70, 75, 80, 85]

summary_rows = []
for age in ages_report:
    qx_hist = mx_male.loc[age, 1990] / (1 + 0.5 * mx_male.loc[age, 1990])
    qx_curr = mx_male.loc[age, 2022] / (1 + 0.5 * mx_male.loc[age, 2022])
    qx_proj = projection_male["central"].loc[age].iloc[-1] if age in projection_male["central"].index else np.nan
    improvement_hist = (qx_curr - qx_hist) / qx_hist
    improvement_proj = (qx_proj - qx_curr) / qx_curr if not np.isnan(qx_proj) else np.nan
    summary_rows.append({
        "Age": age,
        "qx 1990": f"{qx_hist:.4f}",
        "qx 2022": f"{qx_curr:.4f}",
        "qx 2047 (proj)": f"{qx_proj:.4f}" if not np.isnan(qx_proj) else "N/A",
        "Improvement 1990–2022": f"{improvement_hist:.1%}",
        "Improvement 2022–2047": f"{improvement_proj:.1%}" if not np.isnan(improvement_proj) else "N/A",
    })

summary = pd.DataFrame(summary_rows).set_index("Age")
print("\n── Mortality Improvement Summary — France Males ─────────────────")
print(summary.to_string())
