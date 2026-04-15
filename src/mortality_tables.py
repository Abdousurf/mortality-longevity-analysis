"""Mortality table construction and analysis.

Builds mortality tables from raw experience data using raw qx computation,
graduated tables via Whittaker-Henderson smoothing, and the Lee-Carter
stochastic mortality model.

Actuarial notation:
    qx: Probability of dying between age x and x+1.
    px: 1 - qx, probability of surviving.
    lx: Number of survivors at age x (radix l0 = 100,000).
    dx: lx - l(x+1), number of deaths.
    ex: Complete life expectancy at age x.
    mu_x: Force of mortality at age x.
"""

import numpy as np
import pandas as pd
from scipy.linalg import svd
from scipy.optimize import minimize
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")


# ── Mortality Table Construction ─────────────────────────────────────────────

def raw_qx(deaths: np.ndarray, exposures: np.ndarray) -> np.ndarray:
    """Compute raw observed mortality rates using the Balducci assumption.

    Converts central death rates (mx) to initial rates of mortality (qx)
    using the Balducci assumption: qx = mx / (1 + 0.5 * mx).

    Args:
        deaths: Array of observed death counts by age.
        exposures: Array of central exposed-to-risk by age.

    Returns:
        Array of raw mortality rates (qx) by age. Values are NaN where
        exposures are zero.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        mx = np.where(exposures > 0, deaths / exposures, np.nan)
        qx = mx / (1 + 0.5 * mx)  # Balducci: central -> initial exposure
    return qx


def whittaker_henderson_graduation(qx_raw: np.ndarray, h: float = 0.1, z: int = 2) -> np.ndarray:
    """Graduate raw mortality rates using the Whittaker-Henderson method.

    Minimizes a weighted sum of fit and smoothness:
    h * sum((qx_smooth - qx_raw)^2) + (1-h) * sum((delta^z qx_smooth)^2)

    Args:
        qx_raw: Array of raw mortality rates to be graduated.
        h: Smoothness weight between 0 and 1. A value of 0 produces fully
            smooth output; a value of 1 returns the raw data unchanged.
        z: Order of differences to penalize. A value of 2 penalizes
            curvature.

    Returns:
        Array of graduated mortality rates, clipped to the range [1e-6, 1.0].
    """
    n = len(qx_raw)
    mask = ~np.isnan(qx_raw)
    qx_filled = np.where(mask, qx_raw, 0)

    # Difference matrix of order z
    D = np.eye(n)
    for _ in range(z):
        D = np.diff(D, axis=0)

    # Solve: (h*I + (1-h)*D'D) * q_smooth = h * q_raw
    A = h * np.eye(n) + (1 - h) * D.T @ D
    b = h * qx_filled
    qx_smooth = np.linalg.solve(A, b)
    qx_smooth = np.clip(qx_smooth, 1e-6, 1.0)
    return qx_smooth


def build_life_table(qx: np.ndarray, age_start: int = 0, radix: int = 100_000) -> pd.DataFrame:
    """Build a complete period life table from a qx array.

    Constructs all standard life table columns from the given mortality
    rates, starting from a specified radix population.

    Args:
        qx: Array of age-specific mortality rates.
        age_start: Starting age for the life table.
        radix: Initial population at the starting age (l0).

    Returns:
        DataFrame with columns: age, qx, px, lx, dx, Lx, Tx, ex.
    """
    n = len(qx)
    ages = np.arange(age_start, age_start + n)

    lx = np.zeros(n + 1)
    lx[0] = radix
    for i in range(n):
        lx[i + 1] = lx[i] * (1 - qx[i])

    dx = lx[:-1] - lx[1:]
    Lx = 0.5 * (lx[:-1] + lx[1:])  # person-years lived
    Tx = np.cumsum(Lx[::-1])[::-1]   # total person-years above age x
    ex = np.where(lx[:-1] > 0, Tx / lx[:-1], np.nan)

    return pd.DataFrame({
        "age": ages,
        "qx": qx,
        "px": 1 - qx,
        "lx": lx[:-1].astype(int),
        "dx": dx.astype(int),
        "Lx": Lx,
        "Tx": Tx,
        "ex": ex,
    })


# ── Lee-Carter Model ─────────────────────────────────────────────────────────

class LeeCarter:
    """Lee-Carter stochastic mortality model.

    Decomposes log central death rates into age and time components:
    ln(mu_{x,t}) = alpha_x + beta_x * kappa_t

    where:
        alpha_x: Average log-mortality at age x.
        beta_x: Age-specific sensitivity to the time trend.
        kappa_t: Time index capturing the overall mortality level.

    Projection of kappa_t follows an ARIMA(0,1,0) random walk with drift.

    Attributes:
        alpha: Array of fitted alpha_x parameters.
        beta: Array of fitted beta_x parameters.
        kappa: Array of fitted kappa_t parameters.
        ages: Array of ages used in fitting.
        years: Array of years used in fitting.

    Example:
        >>> lc = LeeCarter()
        >>> lc.fit(mx_matrix)
        >>> projections = lc.project_qx(horizon=25)
    """

    def __init__(self):
        """Initialize a LeeCarter model with empty parameters."""
        self.alpha = None
        self.beta = None
        self.kappa = None
        self.ages = None
        self.years = None
        self._kappa_drift = None
        self._kappa_sigma = None

    def fit(self, mx_matrix: pd.DataFrame):
        """Fit the Lee-Carter model to a matrix of central death rates.

        Uses singular value decomposition (SVD) on the centered log-mortality
        surface to extract the age and time components.

        Args:
            mx_matrix: DataFrame of central death rates with shape
                (ages x years), where the index contains ages and the
                columns contain years.

        Returns:
            The fitted LeeCarter instance (self).
        """
        self.ages = mx_matrix.index.values
        self.years = mx_matrix.columns.values

        log_mx = np.log(mx_matrix.values.astype(float))
        log_mx = np.where(np.isinf(log_mx), np.nan, log_mx)

        # α_x = row means
        self.alpha = np.nanmean(log_mx, axis=1)

        # Center: Z = log(μ) - α
        Z = log_mx - self.alpha[:, np.newaxis]
        Z = np.nan_to_num(Z, nan=0.0)

        # SVD decomposition: take first singular value/vectors
        U, s, Vt = svd(Z, full_matrices=False)
        b_raw = U[:, 0] * s[0]
        k_raw = Vt[0, :]

        # Normalize: sum(β_x) = 1, adjust κ_t accordingly
        beta_sum = b_raw.sum()
        self.beta = b_raw / beta_sum
        self.kappa = k_raw * beta_sum

        # Fit ARIMA(0,1,0) on κ_t → drift + sigma
        kappa_diff = np.diff(self.kappa)
        self._kappa_drift = kappa_diff.mean()
        self._kappa_sigma = kappa_diff.std()

        return self

    def project_kappa(self, n_years: int, n_simulations: int = 1000, seed: int = 42) -> np.ndarray:
        """Project kappa_t forward using a random walk with drift.

        Generates stochastic simulations of the mortality time index
        kappa_t using the fitted drift and volatility parameters.

        Args:
            n_years: Number of years to project forward.
            n_simulations: Number of Monte Carlo simulation paths.
            seed: Random number generator seed for reproducibility.

        Returns:
            Array of shape (n_simulations, n_years) containing the
            projected kappa_t paths.
        """
        rng = np.random.default_rng(seed)
        kappa_last = self.kappa[-1]

        simulations = np.zeros((n_simulations, n_years))
        for i in range(n_simulations):
            shocks = rng.normal(self._kappa_drift, self._kappa_sigma, n_years)
            simulations[i] = kappa_last + np.cumsum(shocks)

        return simulations

    def project_qx(self, horizon: int = 25, confidence: float = 0.95,
                   n_sim: int = 1000) -> dict:
        """Project age-specific mortality rates over a given horizon.

        Generates stochastic mortality projections with central estimates
        and confidence intervals derived from Monte Carlo simulation.

        Args:
            horizon: Number of years to project forward.
            confidence: Confidence level for the projection interval
                (e.g., 0.95 for a 95% confidence band).
            n_sim: Number of Monte Carlo simulations.

        Returns:
            Dictionary with three keys:
                central: DataFrame of best-estimate qx values
                    (ages x projection years).
                lower: DataFrame of lower confidence bound qx values.
                upper: DataFrame of upper confidence bound qx values.
        """
        proj_years = np.arange(self.years[-1] + 1, self.years[-1] + horizon + 1)
        kappa_sims = self.project_kappa(horizon, n_simulations=n_sim)

        all_log_mu = np.zeros((n_sim, len(self.ages), horizon))
        for sim_idx in range(n_sim):
            for t_idx in range(horizon):
                log_mu = self.alpha + self.beta * kappa_sims[sim_idx, t_idx]
                all_log_mu[sim_idx, :, t_idx] = log_mu

        # Central = mean across simulations
        central_log_mu = all_log_mu.mean(axis=0)
        central_mx = np.exp(central_log_mu)
        central_qx = central_mx / (1 + 0.5 * central_mx)

        alpha_ci = (1 - confidence) / 2
        lower_log_mu = np.quantile(all_log_mu, alpha_ci, axis=0)
        upper_log_mu = np.quantile(all_log_mu, 1 - alpha_ci, axis=0)

        return {
            "central": pd.DataFrame(central_qx, index=self.ages, columns=proj_years),
            "lower": pd.DataFrame(np.exp(lower_log_mu) / (1 + 0.5 * np.exp(lower_log_mu)),
                                   index=self.ages, columns=proj_years),
            "upper": pd.DataFrame(np.exp(upper_log_mu) / (1 + 0.5 * np.exp(upper_log_mu)),
                                   index=self.ages, columns=proj_years),
        }

    def life_expectancy(self, qx_matrix: pd.DataFrame, start_age: int = 65) -> pd.Series:
        """Compute period life expectancy at a given age for each year.

        Builds a life table for each year column in the provided qx matrix
        and extracts the life expectancy at the specified starting age.

        Args:
            qx_matrix: DataFrame of mortality rates with ages as index
                and years as columns.
            start_age: Age at which to compute life expectancy.

        Returns:
            Series of life expectancy values indexed by year, named
            'e{start_age}'.
        """
        results = {}
        for year in qx_matrix.columns:
            qx = qx_matrix.loc[start_age:, year].values
            lt = build_life_table(qx, age_start=start_age)
            results[year] = lt.loc[lt["age"] == start_age, "ex"].iloc[0]
        return pd.Series(results, name=f"e{start_age}")


# ── Longevity Risk ────────────────────────────────────────────────────────────

def annuity_present_value(qx: np.ndarray, interest_rate: float, start_age: int) -> float:
    """Compute the actuarial present value of a whole life annuity-due.

    Calculates a-double-dot_x = sum(v^t * t_p_x) for a life annuity-due
    paying 1 per year, where v is the discount factor and t_p_x is the
    probability of surviving t years from age x.

    Args:
        qx: Array of age-specific mortality rates starting from start_age.
        interest_rate: Annual effective interest rate for discounting.
        start_age: Age at which the annuity begins.

    Returns:
        Present value of the annuity per unit of annual payment.
    """
    px = 1 - qx
    n = len(px)
    tpx = np.cumprod(np.concatenate([[1.0], px]))[:n]
    v = 1 / (1 + interest_rate)
    vt = v ** np.arange(n)
    return float(np.sum(vt * tpx))


def longevity_shock_impact(qx_base: np.ndarray, shock_factor: float = 0.8,
                            interest_rate: float = 0.03, start_age: int = 65) -> dict:
    """Quantify the impact of a Solvency II longevity stress test.

    Applies a permanent multiplicative reduction to mortality rates
    (standard shock: 20% reduction per Article 142, Delegated Acts)
    and measures the resulting increase in annuity reserves.

    Args:
        qx_base: Array of baseline age-specific mortality rates.
        shock_factor: Multiplicative factor applied to qx. A value of 0.8
            represents a 20% reduction in mortality.
        interest_rate: Annual effective interest rate for discounting.
        start_age: Age at which the annuity begins.

    Returns:
        Dictionary containing:
            annuity_base: Present value of the annuity under base mortality.
            annuity_stressed: Present value under stressed mortality.
            scr_proxy_per_unit: Additional capital required per unit annuity.
            reserve_increase_pct: Percentage increase in reserves.
            shock_factor_applied: The shock factor used.
            start_age: The starting age used.
            interest_rate: The interest rate used.
    """
    qx_stressed = qx_base * shock_factor

    apv_base = annuity_present_value(qx_base, interest_rate, start_age)
    apv_stressed = annuity_present_value(qx_stressed, interest_rate, start_age)

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
