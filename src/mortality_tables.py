"""
Mortality Table Construction & Analysis
=========================================
Builds mortality tables from raw experience data.
Methods: raw qx, graduated tables (Whittaker-Henderson), Lee-Carter model.

Actuarial notation:
  qx  = probability of dying between age x and x+1
  px  = 1 - qx = probability of surviving
  lx  = number of survivors at age x (radix l0 = 100,000)
  dx  = lx - l(x+1) = number of deaths
  ex  = complete life expectancy at age x
  μx  = force of mortality at age x
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
    """
    Compute raw observed mortality rates.
    qx ≈ deaths / central exposed to risk (Balducci assumption).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        mx = np.where(exposures > 0, deaths / exposures, np.nan)
        qx = mx / (1 + 0.5 * mx)  # Balducci: central -> initial exposure
    return qx


def whittaker_henderson_graduation(qx_raw: np.ndarray, h: float = 0.1, z: int = 2) -> np.ndarray:
    """
    Whittaker-Henderson graduation of raw mortality rates.
    Minimizes: h * Σ(qx_smooth - qx_raw)² + (1-h) * Σ(Δ^z qx_smooth)²
    h: smoothness weight (0=fully smooth, 1=raw data)
    z: order of differences (2 = penalize curvature)
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
    """
    Build a complete life table from a qx array.

    Returns columns: age, qx, px, lx, dx, Lx, Tx, ex
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
    """
    Lee-Carter stochastic mortality model.

    Model: ln(μ_{x,t}) = α_x + β_x · κ_t

    where:
        α_x = average log-mortality at age x
        β_x = age-specific sensitivity to time trend
        κ_t = time index (mortality level driver)

    Projection: κ_t follows ARIMA(0,1,0) random walk with drift.
    """

    def __init__(self):
        self.alpha = None
        self.beta = None
        self.kappa = None
        self.ages = None
        self.years = None
        self._kappa_drift = None
        self._kappa_sigma = None

    def fit(self, mx_matrix: pd.DataFrame):
        """
        Fit Lee-Carter to a matrix of central death rates.
        mx_matrix: DataFrame with shape (ages × years), index=ages, columns=years
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
        """
        Project κ_t forward using random walk with drift.
        Returns array of shape (n_simulations, n_years).
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
        """
        Project age-specific mortality rates over a horizon.

        Returns:
            central: best estimate qx matrix (ages × horizon)
            lower / upper: confidence bands
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
        """Compute period life expectancy at start_age for each year column."""
        results = {}
        for year in qx_matrix.columns:
            qx = qx_matrix.loc[start_age:, year].values
            lt = build_life_table(qx, age_start=start_age)
            results[year] = lt.loc[lt["age"] == start_age, "ex"].iloc[0]
        return pd.Series(results, name=f"e{start_age}")


# ── Longevity Risk ────────────────────────────────────────────────────────────

def annuity_present_value(qx: np.ndarray, interest_rate: float, start_age: int) -> float:
    """
    Compute actuarial present value of a whole life annuity of 1 per year.
    ä_x = Σ v^t · t_p_x  (life annuity due)
    """
    px = 1 - qx
    n = len(px)
    tpx = np.cumprod(np.concatenate([[1.0], px]))[:n]
    v = 1 / (1 + interest_rate)
    vt = v ** np.arange(n)
    return float(np.sum(vt * tpx))


def longevity_shock_impact(qx_base: np.ndarray, shock_factor: float = 0.8,
                            interest_rate: float = 0.03, start_age: int = 65) -> dict:
    """
    Solvency II longevity stress: apply a permanent mortality reduction.
    Standard shock: qx_stress = qx_base * (1 - 20%)  [Article 142, Delegated Acts]

    Returns: base reserve, stressed reserve, reserve increase (%), SCR proxy.
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
