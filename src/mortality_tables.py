"""Build and analyze mortality tables — the core math behind this project.

Takes raw death and population data and turns it into clean mortality tables
that show how likely people are to die at each age. Also includes the
Lee-Carter model, which is the go-to method actuaries use to predict how
death rates will change in the future, and tools to measure the financial
risk when people live longer than expected.

Actuarial notation (the shorthand used throughout):
    qx: The chance of dying between age x and age x+1.
    px: 1 - qx, the chance of surviving from age x to age x+1.
    lx: Number of survivors at age x (we start with 100,000 people).
    dx: lx - l(x+1), number of deaths between age x and x+1.
    ex: How many more years a person aged x is expected to live.
    mu_x: The "force of mortality" — a continuous measure of death risk at age x.
"""

# ───────────────────────────────────────────────────────
# WHAT THIS FILE DOES (in plain English):
#
# 1. Turns raw death counts into death-rate-per-age numbers
# 2. Smooths out the noisy raw numbers so they form a clean curve
# 3. Builds a full "life table" — a standard actuarial summary
# 4. Fits the Lee-Carter model to learn how death rates change over time
# 5. Projects future death rates using random simulations
# 6. Calculates the financial cost of annuities and longevity risk
# ───────────────────────────────────────────────────────

import numpy as np
import pandas as pd
from scipy.linalg import svd
import warnings

warnings.filterwarnings("ignore")


# ── Mortality Table Construction ─────────────────────────────────────────────


def raw_qx(deaths: np.ndarray, exposures: np.ndarray) -> np.ndarray:
    """Calculate raw death rates from death counts and population data.

    Takes the number of deaths and the number of people exposed to risk,
    and figures out the chance of dying at each age. Uses a standard
    adjustment (called the Balducci assumption) to convert the rate from
    a "middle-of-year" basis to a "start-of-year" basis.

    Args:
        deaths: Number of deaths at each age.
        exposures: Number of people at risk at each age (the population
            we're tracking over the year).

    Returns:
        An array of death rates by age. If nobody was tracked at a
        certain age, the result is NaN (meaning "no data").
    """
    # Safely divide deaths by exposures; if exposures is zero, return NaN
    with np.errstate(divide="ignore", invalid="ignore"):
        mx = np.where(exposures > 0, deaths / exposures, np.nan)
        # Convert from mid-year rate to start-of-year probability
        qx = mx / (1 + 0.5 * mx)  # Balducci: central -> initial exposure
    return qx


def whittaker_henderson_graduation(
    qx_raw: np.ndarray, h: float = 0.1, z: int = 2
) -> np.ndarray:
    """Smooth out noisy death rates so they form a clean, realistic curve.

    Raw death rates jump around a lot because of small sample sizes.
    This method finds a balance between staying close to the raw data
    and making the curve smooth. Think of it like drawing a best-fit
    line, but for death rates across ages.

    Args:
        qx_raw: The noisy, raw death rates we want to smooth.
        h: Controls the trade-off between accuracy and smoothness.
            Closer to 0 = very smooth; closer to 1 = stays close to
            the raw data.
        z: Controls what kind of bumpiness to remove. A value of 2
            removes sharp changes in the curve's direction.

    Returns:
        Smoothed death rates, guaranteed to be between a tiny positive
        number and 1.0.
    """
    n = len(qx_raw)

    # Figure out which ages have actual data (not missing)
    mask = ~np.isnan(qx_raw)
    qx_filled = np.where(mask, qx_raw, 0)

    # Build a "difference matrix" — this measures how bumpy the curve is
    D = np.eye(n)
    for _ in range(z):
        D = np.diff(D, axis=0)

    # Set up and solve the smoothing equation:
    # We want rates that are both close to the raw data AND smooth
    A = h * np.eye(n) + (1 - h) * D.T @ D
    b = h * qx_filled
    qx_smooth = np.linalg.solve(A, b)

    # Make sure rates stay in a realistic range
    qx_smooth = np.clip(qx_smooth, 1e-6, 1.0)
    return qx_smooth


def build_life_table(
    qx: np.ndarray, age_start: int = 0, radix: int = 100_000
) -> pd.DataFrame:
    """Create a complete life table from death rates — the classic actuarial tool.

    Imagine starting with 100,000 newborns and watching what happens as
    they age. This function tracks how many survive to each age, how many
    die, and how long people can expect to live. It's the foundation of
    all life insurance and pension math.

    Args:
        qx: Death rates at each age (chance of dying within the year).
        age_start: What age the table starts at (usually 0 for birth).
        radix: How many people we start with in our imaginary group
            (100,000 is the standard).

    Returns:
        A table with columns: age, qx (death rate), px (survival rate),
        lx (survivors), dx (deaths), Lx (person-years lived),
        Tx (total future person-years), ex (life expectancy).
    """
    n = len(qx)
    ages = np.arange(age_start, age_start + n)

    # Track how many people survive to each age
    lx = np.zeros(n + 1)
    lx[0] = radix
    for i in range(n):
        # Each year, some people die — the rest move on to the next age
        lx[i + 1] = lx[i] * (1 - qx[i])

    # Calculate deaths at each age (difference in survivors)
    dx = lx[:-1] - lx[1:]

    # Person-years lived: average of survivors at start and end of year
    Lx = 0.5 * (lx[:-1] + lx[1:])  # person-years lived

    # Total future person-years: sum from this age onward
    Tx = np.cumsum(Lx[::-1])[::-1]  # total person-years above age x

    # Life expectancy: total future years divided by current survivors
    ex = np.where(lx[:-1] > 0, Tx / lx[:-1], np.nan)

    return pd.DataFrame(
        {
            "age": ages,
            "qx": qx,
            "px": 1 - qx,
            "lx": lx[:-1].astype(int),
            "dx": dx.astype(int),
            "Lx": Lx,
            "Tx": Tx,
            "ex": ex,
        }
    )


# ── Lee-Carter Model ─────────────────────────────────────────────────────────


class LeeCarter:
    """The Lee-Carter model — the standard way to predict future death rates.

    This model looks at how death rates have changed over many years and
    finds the main pattern of improvement. It breaks down death rates into
    three pieces:

    ln(death_rate at age x in year t) = alpha_x + beta_x * kappa_t

    In plain English:
        alpha_x: The typical death rate at each age (averaged over all years).
        beta_x: How much each age benefits from the overall improvement trend.
        kappa_t: A single number for each year that captures the overall
                 level of mortality — when this goes down, people are living longer.

    To predict the future, we assume kappa_t keeps drifting downward with
    some random variation (like a stock price with a downward trend).

    Stored values after fitting:
        alpha: The average death-rate pattern by age.
        beta: How sensitive each age is to the time trend.
        kappa: The mortality time index for each historical year.
        ages: The ages used when fitting.
        years: The years used when fitting.

    Example:
        >>> lc = LeeCarter()
        >>> lc.fit(mx_matrix)
        >>> projections = lc.project_qx(horizon=25)
    """

    def __init__(self):
        """Create a new, empty Lee-Carter model (no data fitted yet)."""
        self.alpha = None
        self.beta = None
        self.kappa = None
        self.ages = None
        self.years = None
        self._kappa_drift = None
        self._kappa_sigma = None

    def fit(self, mx_matrix: pd.DataFrame):
        """Learn the mortality pattern from historical death-rate data.

        Looks at a table of death rates (ages down the side, years across
        the top) and figures out the three key components: the average
        pattern by age, how much each age improves over time, and the
        overall mortality level for each year.

        Uses a math technique called SVD (a way to find the most important
        pattern in a table of numbers) to pull out these components.

        Args:
            mx_matrix: A table of death rates where each row is an age
                and each column is a year.

        Returns:
            This model, now fitted and ready to make predictions.
        """
        self.ages = mx_matrix.index.values
        self.years = mx_matrix.columns.values

        # Take the log of death rates (makes the math work better)
        log_mx = np.log(mx_matrix.values.astype(float))
        log_mx = np.where(np.isinf(log_mx), np.nan, log_mx)

        # Step 1: Find the average death rate at each age across all years
        self.alpha = np.nanmean(log_mx, axis=1)

        # Step 2: Subtract the average to see just the year-to-year changes
        Z = log_mx - self.alpha[:, np.newaxis]
        Z = np.nan_to_num(Z, nan=0.0)

        # Step 3: Use SVD to find the single most important pattern
        # This gives us how each age responds (beta) and the time trend (kappa)
        U, s, Vt = svd(Z, full_matrices=False)
        b_raw = U[:, 0] * s[0]
        k_raw = Vt[0, :]

        # Step 4: Rescale so the age sensitivities add up to 1
        beta_sum = b_raw.sum()
        self.beta = b_raw / beta_sum
        self.kappa = k_raw * beta_sum

        # Step 5: Measure how fast kappa is declining and how much it bounces around
        # This tells us the trend speed and uncertainty for predictions
        kappa_diff = np.diff(self.kappa)
        self._kappa_drift = kappa_diff.mean()
        self._kappa_sigma = kappa_diff.std()

        return self

    def project_kappa(
        self, n_years: int, n_simulations: int = 1000, seed: int = 42
    ) -> np.ndarray:
        """Simulate many possible future paths for the mortality time index.

        Takes the trend and randomness we learned from history, and runs
        thousands of "what if" scenarios into the future. Each scenario
        follows the same average trend but with different random bumps —
        like predicting where a leaf floating downstream might end up.

        Args:
            n_years: How many years into the future to predict.
            n_simulations: How many different random scenarios to run
                (more = more reliable results, but slower).
            seed: A number that makes the random results repeatable
                (use the same seed, get the same results).

        Returns:
            A table with one row per scenario and one column per future
            year, containing the predicted kappa values.
        """
        rng = np.random.default_rng(seed)

        # Start from the last known value of kappa
        kappa_last = self.kappa[-1]

        # Run each simulation: add random shocks around the average trend
        simulations = np.zeros((n_simulations, n_years))
        for i in range(n_simulations):
            # Each year gets a random change centered on the historical drift
            shocks = rng.normal(self._kappa_drift, self._kappa_sigma, n_years)
            # Add up the changes to get the running total
            simulations[i] = kappa_last + np.cumsum(shocks)

        return simulations

    def project_qx(
        self, horizon: int = 25, confidence: float = 0.95, n_sim: int = 1000
    ) -> dict:
        """Predict future death rates at every age, with uncertainty ranges.

        Runs thousands of simulations to produce a best guess for future
        death rates, plus upper and lower bounds showing how uncertain
        we are. The wider the bounds, the less sure we are about the future.

        Args:
            horizon: How many years ahead to predict.
            confidence: How wide the uncertainty range should be.
                0.95 means we're 95% sure the true value falls within
                the range.
            n_sim: Number of random simulations to run.

        Returns:
            A dictionary with three tables:
                central: Our best guess for death rates (ages by future years).
                lower: The optimistic end of the range (lower death rates).
                upper: The pessimistic end of the range (higher death rates).
        """
        # Set up the future year labels
        proj_years = np.arange(self.years[-1] + 1, self.years[-1] + horizon + 1)

        # Run all the random simulations for kappa
        kappa_sims = self.project_kappa(horizon, n_simulations=n_sim)

        # For each simulation, calculate what death rates would be
        all_log_mu = np.zeros((n_sim, len(self.ages), horizon))
        for sim_idx in range(n_sim):
            for t_idx in range(horizon):
                # Reconstruct death rates from the model components
                log_mu = self.alpha + self.beta * kappa_sims[sim_idx, t_idx]
                all_log_mu[sim_idx, :, t_idx] = log_mu

        # Best guess: average across all simulations
        central_log_mu = all_log_mu.mean(axis=0)
        central_mx = np.exp(central_log_mu)
        # Convert from continuous rate to yearly probability
        central_qx = central_mx / (1 + 0.5 * central_mx)

        # Find the upper and lower bounds of the uncertainty range
        alpha_ci = (1 - confidence) / 2
        lower_log_mu = np.quantile(all_log_mu, alpha_ci, axis=0)
        upper_log_mu = np.quantile(all_log_mu, 1 - alpha_ci, axis=0)

        return {
            "central": pd.DataFrame(central_qx, index=self.ages, columns=proj_years),
            "lower": pd.DataFrame(
                np.exp(lower_log_mu) / (1 + 0.5 * np.exp(lower_log_mu)),
                index=self.ages,
                columns=proj_years,
            ),
            "upper": pd.DataFrame(
                np.exp(upper_log_mu) / (1 + 0.5 * np.exp(upper_log_mu)),
                index=self.ages,
                columns=proj_years,
            ),
        }

    def life_expectancy(
        self, qx_matrix: pd.DataFrame, start_age: int = 65
    ) -> pd.Series:
        """Figure out how many more years a person at a given age can expect to live.

        For each future year, builds a life table and reads off the life
        expectancy at the chosen age. This shows whether people are
        expected to live longer or shorter in the future.

        Args:
            qx_matrix: A table of death rates with ages as rows and
                years as columns.
            start_age: The age we care about (e.g., 65 for retirement age).

        Returns:
            A list of life expectancy values, one for each year,
            labeled 'e{start_age}' (e.g., 'e65').
        """
        results = {}
        # Go through each year and build a life table starting at the chosen age
        for year in qx_matrix.columns:
            qx = qx_matrix.loc[start_age:, year].values
            lt = build_life_table(qx, age_start=start_age)
            # Pull out the life expectancy for our starting age
            results[year] = lt.loc[lt["age"] == start_age, "ex"].iloc[0]
        return pd.Series(results, name=f"e{start_age}")


# ── Longevity Risk ────────────────────────────────────────────────────────────


def annuity_present_value(
    qx: np.ndarray, interest_rate: float, start_age: int
) -> float:
    """Calculate how much money you need today to fund a lifetime annual payment.

    If you promise to pay someone $1 every year for the rest of their life
    (starting immediately), this function tells you how much money you
    need to set aside right now. It accounts for both the chance they
    might die (ending payments) and the interest you'd earn on the money.

    Args:
        qx: Death rates starting from the person's current age.
        interest_rate: The yearly return you expect to earn on the money
            (e.g., 0.03 for 3% per year).
        start_age: The person's current age.

    Returns:
        The lump sum needed today to fund $1/year for life.
    """
    # Calculate survival probabilities from the death rates
    px = 1 - qx
    n = len(px)

    # Build cumulative survival: chance of being alive after 0, 1, 2, ... years
    tpx = np.cumprod(np.concatenate([[1.0], px]))[:n]

    # Discount factor: $1 next year is worth less than $1 today
    v = 1 / (1 + interest_rate)
    vt = v ** np.arange(n)

    # Each year's payment is worth: (discount factor) * (chance of being alive)
    return float(np.sum(vt * tpx))


def longevity_shock_impact(
    qx_base: np.ndarray,
    shock_factor: float = 0.8,
    interest_rate: float = 0.03,
    start_age: int = 65,
) -> dict:
    """Measure the financial impact if people suddenly start living longer.

    Insurance regulators (under Solvency II rules) require companies to
    test what happens if death rates permanently drop by 20%. If people
    live longer, pension and annuity companies have to pay out for more
    years, which costs more money. This function calculates how much
    extra money would be needed.

    Args:
        qx_base: Current death rates by age (the "normal" scenario).
        shock_factor: How much to reduce death rates. 0.8 means death
            rates drop by 20% (people live longer).
        interest_rate: Yearly return on invested money (e.g., 0.03 for 3%).
        start_age: The age of the person receiving payments.

    Returns:
        A summary with:
            annuity_base: Cost of the annuity under normal death rates.
            annuity_stressed: Cost if people live longer (after the shock).
            scr_proxy_per_unit: Extra money needed per $1 of annual payment.
            reserve_increase_pct: How much more expensive the annuity becomes (%).
            shock_factor_applied: The shock factor that was used.
            start_age: The starting age that was used.
            interest_rate: The interest rate that was used.
    """
    # Apply the shock: lower death rates = people live longer
    qx_stressed = qx_base * shock_factor

    # Calculate annuity costs under both scenarios
    apv_base = annuity_present_value(qx_base, interest_rate, start_age)
    apv_stressed = annuity_present_value(qx_stressed, interest_rate, start_age)

    # The difference is the extra capital needed
    scr_proxy = apv_stressed - apv_base
    impact_pct = (apv_stressed - apv_base) / apv_base

    return {
        "annuity_base": round(apv_base, 4),
        "annuity_stressed": round(apv_stressed, 4),
        "scr_proxy_per_unit": round(scr_proxy, 4),
        "reserve_increase_pct": round(impact_pct * 100, 2),
        "shock_factor_applied": shock_factor,
        "start_age": start_age,
        "interest_rate": interest_rate,
    }
