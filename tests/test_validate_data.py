"""Tests for the Pandera-based mortality data validation functions.

Validates schema checks, actuarial monotonicity, and gender differential logic.
"""

import numpy as np
import pandas as pd
import pytest

from src.validate_data import (
    validate_mortality_table,
    check_actuarial_monotonicity,
    check_gender_differential,
)


@pytest.fixture
def valid_mortality_df():
    """Create a realistic mortality DataFrame that passes all validations."""
    rows = []
    for sexe in ["M", "F"]:
        for annee in [2010, 2015, 2020]:
            for age in range(0, 111):
                if sexe == "M":
                    qx = min(0.005 * np.exp(0.075 * age), 1.0)
                else:
                    qx = min(0.004 * np.exp(0.073 * age), 1.0)
                # Infant mortality: qx(0) must be > qx(1)
                if age == 0:
                    qx = 0.008 if sexe == "M" else 0.006
                elif age == 1:
                    qx = 0.001
                rows.append({"annee": annee, "age": age, "qx": qx, "sexe": sexe})
    return pd.DataFrame(rows)


# ── Schema validation ──────────────────────────────────────────────────────


class TestValidateMortalityTable:
    """Tests for Pandera schema validation."""

    def test_valid_data_passes(self, valid_mortality_df):
        assert validate_mortality_table(valid_mortality_df) is True

    def test_qx_out_of_range_fails(self, valid_mortality_df):
        df = valid_mortality_df.copy()
        df.loc[0, "qx"] = -0.1
        with pytest.raises(Exception):
            validate_mortality_table(df)

    def test_qx_above_one_fails(self, valid_mortality_df):
        df = valid_mortality_df.copy()
        df.loc[0, "qx"] = 1.5
        with pytest.raises(Exception):
            validate_mortality_table(df)

    def test_invalid_sexe_fails(self, valid_mortality_df):
        df = valid_mortality_df.copy()
        df.loc[0, "sexe"] = "X"
        with pytest.raises(Exception):
            validate_mortality_table(df)

    def test_year_out_of_range_fails(self, valid_mortality_df):
        df = valid_mortality_df.copy()
        df.loc[0, "annee"] = 1800
        with pytest.raises(Exception):
            validate_mortality_table(df)

    def test_nan_qx_fails(self, valid_mortality_df):
        df = valid_mortality_df.copy()
        df.loc[5, "qx"] = np.nan
        with pytest.raises(Exception):
            validate_mortality_table(df)


# ── Actuarial monotonicity ─────────────────────────────────────────────────


class TestActuarialMonotonicity:
    """Tests for the monotonicity check (qx increasing with age after 30)."""

    def test_valid_data_is_monotone(self, valid_mortality_df):
        result = check_actuarial_monotonicity(valid_mortality_df)
        assert result["M"] == True  # noqa: E712
        assert result["F"] == True  # noqa: E712

    def test_non_monotone_detected(self):
        """If qx drops with age after 30, the check should catch it."""
        rows = []
        for annee in [2020]:
            for age in range(0, 111):
                qx = 0.005 * np.exp(0.07 * age)
                if age == 50:
                    qx = 0.0001  # Anomalous drop
                rows.append({"annee": annee, "age": age, "qx": min(qx, 1.0), "sexe": "M"})
        df = pd.DataFrame(rows)
        result = check_actuarial_monotonicity(df)
        assert result["M"] == False  # noqa: E712


# ── Gender differential ────────────────────────────────────────────────────


class TestGenderDifferential:
    """Tests for the gender mortality differential check."""

    def test_valid_differential(self, valid_mortality_df):
        assert check_gender_differential(valid_mortality_df) is True

    def test_inverted_differential_detected(self):
        """If female qx > male qx everywhere, should fail."""
        rows = []
        for annee in [2020]:
            for age in range(0, 111):
                rows.append({"annee": annee, "age": age, "qx": 0.01 * np.exp(0.06 * age), "sexe": "F"})
                rows.append({"annee": annee, "age": age, "qx": 0.005 * np.exp(0.06 * age), "sexe": "M"})
        df = pd.DataFrame(rows)
        df["qx"] = df["qx"].clip(upper=1.0)
        assert check_gender_differential(df, min_age=20) is False

    def test_single_sex_passes(self):
        """If only one sex present, the check should pass (can't compare)."""
        rows = []
        for age in range(0, 111):
            rows.append({"annee": 2020, "age": age, "qx": 0.005 * np.exp(0.07 * age), "sexe": "M"})
        df = pd.DataFrame(rows)
        df["qx"] = df["qx"].clip(upper=1.0)
        assert check_gender_differential(df) is True
