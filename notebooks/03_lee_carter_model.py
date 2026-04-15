"""Lee-Carter mortality model notebook for France (1968-2022).

Walks through a complete mortality forecasting workflow: fits the
Lee-Carter model to French death-rate data, runs thousands of
simulations to predict future death rates, calculates life expectancy
trends, and tests what happens financially if people start living
significantly longer.

Execute as a script or open with Jupytext/Jupyter.

Covers:
    - Fitting the Lee-Carter model to French death-rate data.
    - Projecting the kappa time index into the future (random walk).
    - Drawing a 25-year "fan chart" showing predicted death rates.
    - Predicting life expectancy at age 65.
    - Comparing male vs female mortality trends.
"""

# ───────────────────────────────────────────────────────
# WHAT THIS FILE DOES (in plain English):
#
# 1. Loads French death-rate data (or creates fake data if not available)
# 2. Fits the Lee-Carter model to learn how death rates have been changing
# 3. Plots the three key model components (age pattern, age sensitivity, time trend)
# 4. Predicts death rates 25 years into the future with uncertainty bands
# 5. Calculates how long a 65-year-old can expect to live in the future
# 6. Tests the financial impact of a regulatory "longevity shock" scenario
# 7. Summarizes how much death rates have improved at key ages
# ───────────────────────────────────────────────────────

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

# Import our custom mortality tools
from src.mortality_tables import LeeCarter, build_life_table, longevity_shock_impact

# Set up nice-looking charts
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120, "font.size": 11})

# %% [markdown]
# ## 1. Load HMD Data
#
# Source: Human Mortality Database (France)
# Register free at mortality.org

# %%
def load_hmd_france(data_dir: Path, gender: str = "Male") -> pd.DataFrame:
    """Read French death-rate data and arrange it as an ages-by-years table.

    Opens the HMD data file for the chosen gender, keeps only ages 40-100
    and years 1968-2022, and reshapes it into a table where each row is
    an age and each column is a year.

    Args:
        data_dir: The folder where the raw HMD data files are stored.
        gender: Which sex to load — either 'Male' or 'Female'.

    Returns:
        A table of death rates with ages as rows (40-100) and years
        as columns (1968-2022).
    """
    # Pick the right file based on gender
    suffix = "m" if gender == "Male" else "f"
    filepath = data_dir / f"FRA.Mx_{suffix}x1.txt"

    # Read the file, skipping the header rows
    df = pd.read_csv(filepath, sep=r"\s+", skiprows=2, na_values=".")

    # Remove the oldest age group (110+) since it's a catch-all bucket
    df = df[df["Age"] != "110+"].copy()
    df["Age"] = df["Age"].astype(int)

    # Keep only the ages and years we want
    df = df[df["Age"].between(40, 100)]
    df = df[df["Year"].between(1968, 2022)]

    # Reshape so each row is an age and each column is a year
    mx_matrix = df.pivot(index="Age", columns="Year", values="Total")
    mx_matrix = mx_matrix.astype(float).fillna(method="ffill", axis=1)
    return mx_matrix


# Point to where the data files live
DATA_DIR = Path("../data/raw")

# Try to load real data; if not found, create realistic fake data
try:
    mx_male = load_hmd_france(DATA_DIR, "Male")
    mx_female = load_hmd_france(DATA_DIR, "Female")
    print(f"Data loaded: ages {mx_male.index.min()}–{mx_male.index.max()}, "
          f"years {mx_male.columns.min()}–{mx_male.columns.max()}")
except FileNotFoundError:
    # No real data available — generate synthetic data that looks realistic
    print("HMD data not found — generating synthetic mortality surface...")
    ages = np.arange(40, 101)
    years = np.arange(1968, 2023)

    # Use the Gompertz formula (death rates rise exponentially with age)
    # and add a yearly improvement trend (death rates drop over time)
    A, B = 0.0001, 0.1
    improvement_rate = 0.015  # 1.5% annual improvement
    mx_vals = np.outer(
        A * np.exp(B * (ages - 40)),
        np.exp(-improvement_rate * (years - 1968))
    )
    mx_male = pd.DataFrame(mx_vals, index=ages, columns=years)
    # Women generally have lower death rates than men
    mx_female = pd.DataFrame(mx_vals * 0.65, index=ages, columns=years)

# %% [markdown]
# ## 2. Fit Lee-Carter Model

# %%
# Fit the model separately for men and women
lc_male = LeeCarter().fit(mx_male)
lc_female = LeeCarter().fit(mx_female)

# Show key numbers about the fitted model
print("Lee-Carter Fit — Males")
print(f"  κ_t range: [{lc_male.kappa.min():.1f}, {lc_male.kappa.max():.1f}]")
print(f"  κ_t drift: {lc_male._kappa_drift:.3f} (annual)")
print(f"  κ_t sigma: {lc_male._kappa_sigma:.3f}")

# %% [markdown]
# ## 3. Plot α_x and β_x Parameters

# %%
# Create three side-by-side charts showing the model components
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

ages = lc_male.ages

# Chart 1: Average death rate by age (higher = more dangerous age)
axes[0].plot(ages, lc_male.alpha, label="Males", color="steelblue")
axes[0].plot(ages, lc_female.alpha, label="Females", color="coral", linestyle="--")
axes[0].set_title("α_x — Average log-mortality by age")
axes[0].set_xlabel("Age")
axes[0].set_ylabel("ln(μ_x)")
axes[0].legend()

# Chart 2: How much each age benefits from the improvement trend
axes[1].plot(ages, lc_male.beta, label="Males", color="steelblue")
axes[1].plot(ages, lc_female.beta, label="Females", color="coral", linestyle="--")
axes[1].set_title("β_x — Age sensitivity to time trend")
axes[1].set_xlabel("Age")
axes[1].set_ylabel("β_x")
axes[1].legend()

# Chart 3: The overall mortality trend over time (downward = improving)
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
# Run 2,000 simulations to predict death rates 25 years into the future
projection_male = lc_male.project_qx(horizon=25, confidence=0.95, n_sim=2000)
projection_female = lc_female.project_qx(horizon=25, confidence=0.95, n_sim=2000)

# Focus on age 70 for the fan chart
age_focus = 70

# Create side-by-side fan charts for men and women
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"Projected Mortality Rate q_{age_focus} — France (2023–2047)", fontsize=13)

for ax, proj, gender, color in zip(
    axes,
    [projection_male, projection_female],
    ["Males", "Females"],
    ["steelblue", "coral"]
):
    # Get the central prediction and the upper/lower uncertainty bounds
    central = proj["central"].loc[age_focus]
    lower = proj["lower"].loc[age_focus]
    upper = proj["upper"].loc[age_focus]

    # Shaded area shows the range of likely outcomes (95% confidence)
    ax.fill_between(central.index, lower, upper, alpha=0.25, color=color, label="95% CI")
    # Solid line shows the best-guess prediction
    ax.plot(central.index, central, color=color, linewidth=2.5, label="Central estimate")

    # Add the last 10 years of actual data for comparison
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
# Calculate how many more years a 65-year-old can expect to live, each future year
e65_male = lc_male.life_expectancy(projection_male["central"], start_age=65)
e65_female = lc_female.life_expectancy(projection_female["central"], start_age=65)

# Plot the life expectancy trend
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(e65_male.index, e65_male.values, "o-", color="steelblue", label="Males e₆₅")
ax.plot(e65_female.index, e65_female.values, "s--", color="coral", label="Females e₆₅")
ax.set_title("Projected Life Expectancy at Age 65 — France")
ax.set_xlabel("Year")
ax.set_ylabel("Life expectancy (years)")
ax.legend()
ax.grid(True, alpha=0.3)

# Add a label showing the final projected value for males
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
# Take the projected 2023 death rates for males as our starting point
qx_2023_male = projection_male["central"].iloc[:, 0].values

# Run the regulatory stress test: what if death rates suddenly drop 20%?
result = longevity_shock_impact(
    qx_base=qx_2023_male,
    shock_factor=0.80,       # SII: 20% permanent reduction
    interest_rate=0.03,
    start_age=65
)

# Print the financial impact
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
# Pick key ages to summarize
ages_report = [50, 60, 65, 70, 75, 80, 85]

# For each age, compare historical (1990), current (2022), and projected (2047) death rates
summary_rows = []
for age in ages_report:
    # Convert from central rate to yearly probability for each time point
    qx_hist = mx_male.loc[age, 1990] / (1 + 0.5 * mx_male.loc[age, 1990])
    qx_curr = mx_male.loc[age, 2022] / (1 + 0.5 * mx_male.loc[age, 2022])
    qx_proj = projection_male["central"].loc[age].iloc[-1] if age in projection_male["central"].index else np.nan

    # Calculate how much death rates have improved (negative = lives saved)
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

# Display the summary table
summary = pd.DataFrame(summary_rows).set_index("Age")
print("\n── Mortality Improvement Summary — France Males ─────────────────")
print(summary.to_string())
