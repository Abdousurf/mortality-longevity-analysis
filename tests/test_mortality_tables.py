"""Tests for the core actuarial mortality table functions and Lee-Carter model.

Validates raw qx computation, Whittaker-Henderson graduation,
life table construction, Lee-Carter fit/project, and longevity risk functions.
"""

import numpy as np
import pandas as pd
import pytest

from src.mortality_tables import (
    raw_qx,
    whittaker_henderson_graduation,
    build_life_table,
    LeeCarter,
    annuity_present_value,
    longevity_shock_impact,
)


# ── raw_qx ─────────────────────────────────────────────────────────────────


class TestRawQx:
    """Tests for raw death rate calculation."""

    def test_basic_computation(self):
        deaths = np.array([10.0, 20.0, 50.0])
        exposures = np.array([1000.0, 1000.0, 1000.0])
        qx = raw_qx(deaths, exposures)
        assert qx.shape == (3,)
        assert all(0 < q < 1 for q in qx)

    def test_zero_exposure_gives_nan(self):
        deaths = np.array([10.0, 5.0])
        exposures = np.array([1000.0, 0.0])
        qx = raw_qx(deaths, exposures)
        assert not np.isnan(qx[0])
        assert np.isnan(qx[1])

    def test_qx_less_than_mx(self):
        """Balducci conversion: qx should be <= mx for small rates."""
        deaths = np.array([50.0])
        exposures = np.array([1000.0])
        mx = deaths / exposures
        qx = raw_qx(deaths, exposures)
        assert qx[0] <= mx[0]

    def test_no_deaths_gives_zero(self):
        deaths = np.array([0.0, 0.0])
        exposures = np.array([1000.0, 500.0])
        qx = raw_qx(deaths, exposures)
        assert all(q == 0.0 for q in qx)


# ── Whittaker-Henderson ────────────────────────────────────────────────────


class TestWhittakerHenderson:
    """Tests for the smoothing/graduation function."""

    def test_output_length_matches_input(self):
        qx_raw = np.random.uniform(0.001, 0.5, size=50)
        qx_smooth = whittaker_henderson_graduation(qx_raw)
        assert len(qx_smooth) == len(qx_raw)

    def test_output_bounded(self):
        qx_raw = np.random.uniform(0.001, 0.9, size=50)
        qx_smooth = whittaker_henderson_graduation(qx_raw)
        assert all(q > 0 for q in qx_smooth)
        assert all(q <= 1.0 for q in qx_smooth)

    def test_smoother_than_input(self):
        """Smoothed series should have less variation than raw."""
        np.random.seed(42)
        qx_raw = 0.001 * np.exp(0.06 * np.arange(80)) + np.random.normal(0, 0.005, 80)
        qx_raw = np.clip(qx_raw, 0.0001, 1.0)
        qx_smooth = whittaker_henderson_graduation(qx_raw, h=0.05)
        raw_var = np.var(np.diff(qx_raw))
        smooth_var = np.var(np.diff(qx_smooth))
        assert smooth_var < raw_var

    def test_handles_nan_in_input(self):
        qx_raw = np.array([0.01, 0.02, np.nan, 0.04, 0.05])
        qx_smooth = whittaker_henderson_graduation(qx_raw)
        assert len(qx_smooth) == 5
        assert not any(np.isnan(qx_smooth))


# ── Life Table ─────────────────────────────────────────────────────────────


class TestBuildLifeTable:
    """Tests for life table construction."""

    @pytest.fixture
    def simple_qx(self):
        """Exponentially increasing mortality — realistic shape."""
        return 0.001 * np.exp(0.07 * np.arange(100))

    def test_output_columns(self, simple_qx):
        lt = build_life_table(simple_qx)
        expected_cols = {"age", "qx", "px", "lx", "dx", "Lx", "Tx", "ex"}
        assert set(lt.columns) == expected_cols

    def test_radix(self, simple_qx):
        lt = build_life_table(simple_qx, radix=100_000)
        assert lt["lx"].iloc[0] == 100_000

    def test_lx_decreasing(self, simple_qx):
        lt = build_life_table(simple_qx)
        assert all(lt["lx"].diff().dropna() <= 0)

    def test_dx_positive(self, simple_qx):
        lt = build_life_table(simple_qx)
        assert all(lt["dx"] >= 0)

    def test_ex_at_birth_reasonable(self, simple_qx):
        lt = build_life_table(simple_qx)
        e0 = lt.loc[lt["age"] == 0, "ex"].iloc[0]
        assert 30 < e0 < 120

    def test_px_plus_qx_equals_one(self, simple_qx):
        lt = build_life_table(simple_qx)
        np.testing.assert_allclose(lt["px"] + lt["qx"], 1.0)

    def test_custom_start_age(self, simple_qx):
        lt = build_life_table(simple_qx[65:], age_start=65)
        assert lt["age"].iloc[0] == 65


# ── Lee-Carter Model ──────────────────────────────────────────────────────


class TestLeeCarter:
    """Tests for Lee-Carter model fit and projection."""

    @pytest.fixture
    def synthetic_mx(self):
        """Create a synthetic mortality matrix with known trend."""
        np.random.seed(42)
        ages = np.arange(0, 101)
        years = np.arange(2000, 2021)
        mx = np.zeros((len(ages), len(years)))
        for i, age in enumerate(ages):
            base = 0.0001 * np.exp(0.08 * age)
            for j, year in enumerate(years):
                improvement = 1 - 0.01 * (year - 2000)
                mx[i, j] = base * improvement * (1 + 0.01 * np.random.randn())
                mx[i, j] = max(mx[i, j], 1e-8)
        return pd.DataFrame(mx, index=ages, columns=years)

    def test_fit_sets_attributes(self, synthetic_mx):
        lc = LeeCarter()
        lc.fit(synthetic_mx)
        assert lc.alpha is not None
        assert lc.beta is not None
        assert lc.kappa is not None
        assert len(lc.alpha) == len(synthetic_mx.index)
        assert len(lc.kappa) == len(synthetic_mx.columns)

    def test_beta_sums_to_one(self, synthetic_mx):
        lc = LeeCarter()
        lc.fit(synthetic_mx)
        np.testing.assert_allclose(lc.beta.sum(), 1.0, atol=1e-10)

    def test_kappa_trend_negative(self, synthetic_mx):
        """With improving mortality, kappa should trend downward."""
        lc = LeeCarter()
        lc.fit(synthetic_mx)
        assert lc._kappa_drift < 0

    def test_project_kappa_shape(self, synthetic_mx):
        lc = LeeCarter()
        lc.fit(synthetic_mx)
        sims = lc.project_kappa(n_years=10, n_simulations=100)
        assert sims.shape == (100, 10)

    def test_project_qx_keys(self, synthetic_mx):
        lc = LeeCarter()
        lc.fit(synthetic_mx)
        proj = lc.project_qx(horizon=5, n_sim=50)
        assert "central" in proj
        assert "lower" in proj
        assert "upper" in proj

    def test_project_qx_bounded(self, synthetic_mx):
        lc = LeeCarter()
        lc.fit(synthetic_mx)
        proj = lc.project_qx(horizon=5, n_sim=50)
        assert (proj["central"].values > 0).all()
        assert (proj["central"].values <= 1).all()

    def test_confidence_interval_ordering(self, synthetic_mx):
        lc = LeeCarter()
        lc.fit(synthetic_mx)
        proj = lc.project_qx(horizon=5, n_sim=200)
        assert (proj["lower"].values <= proj["upper"].values).all()

    def test_life_expectancy(self, synthetic_mx):
        lc = LeeCarter()
        lc.fit(synthetic_mx)
        proj = lc.project_qx(horizon=5, n_sim=50)
        e65 = lc.life_expectancy(proj["central"], start_age=65)
        assert len(e65) == 5
        assert all(e > 0 for e in e65)

    def test_fit_returns_self(self, synthetic_mx):
        lc = LeeCarter()
        result = lc.fit(synthetic_mx)
        assert result is lc


# ── Annuity & Longevity Risk ──────────────────────────────────────────────


class TestAnnuityPresentValue:
    """Tests for annuity present value calculation."""

    def test_positive_value(self):
        qx = 0.001 * np.exp(0.07 * np.arange(35))
        apv = annuity_present_value(qx, interest_rate=0.03, start_age=65)
        assert apv > 0

    def test_higher_interest_lower_apv(self):
        qx = 0.001 * np.exp(0.07 * np.arange(35))
        apv_low = annuity_present_value(qx, interest_rate=0.01, start_age=65)
        apv_high = annuity_present_value(qx, interest_rate=0.05, start_age=65)
        assert apv_low > apv_high

    def test_certain_death_gives_one(self):
        """If qx=1 at all ages, only the first payment counts → APV ≈ 1."""
        qx = np.ones(10)
        apv = annuity_present_value(qx, interest_rate=0.03, start_age=65)
        np.testing.assert_allclose(apv, 1.0, atol=0.01)


class TestLongevityShockImpact:
    """Tests for Solvency II longevity shock calculation."""

    @pytest.fixture
    def base_qx(self):
        return 0.001 * np.exp(0.07 * np.arange(35))

    def test_shock_increases_annuity(self, base_qx):
        result = longevity_shock_impact(base_qx, shock_factor=0.8)
        assert result["annuity_stressed"] > result["annuity_base"]

    def test_positive_scr(self, base_qx):
        result = longevity_shock_impact(base_qx)
        assert result["scr_proxy_per_unit"] > 0

    def test_reserve_increase_positive(self, base_qx):
        result = longevity_shock_impact(base_qx)
        assert result["reserve_increase_pct"] > 0

    def test_result_keys(self, base_qx):
        result = longevity_shock_impact(base_qx)
        expected = {
            "annuity_base", "annuity_stressed", "scr_proxy_per_unit",
            "reserve_increase_pct", "shock_factor_applied", "start_age",
            "interest_rate",
        }
        assert set(result.keys()) == expected

    def test_no_shock_no_impact(self, base_qx):
        """shock_factor=1.0 → no change in mortality → no impact."""
        result = longevity_shock_impact(base_qx, shock_factor=1.0)
        np.testing.assert_allclose(result["scr_proxy_per_unit"], 0.0, atol=1e-6)
