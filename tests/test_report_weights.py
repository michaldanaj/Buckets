# coding: utf-8
"""
Testy raportu z wagami: rola WEIGHTS w ColumnTypes + przepływ wag przez
DatasetReport/VariableAnalysis (spec/2026-07-02-raport-spark.md, sekcje 3.4 i 4.2).

Równoważność: pełna analiza (buckety, gini, gini w czasie, dyskretyzacja
drzewem) na ramce ważonej == na ramce zreplikowanej wierszowo wg wag.
"""

import numpy as np
import pandas as pd
import pytest

import buckets.column_types as ct
from buckets.report import DatasetReport


def make_data(n=300, seed=11):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "kat": rng.choice(["a", "b", "c"], size=n, p=[0.5, 0.3, 0.2]),
        "x": np.round(rng.normal(size=n), 2),
        "czas": rng.choice(["2024-01", "2024-02", "2024-03"], size=n),
    })
    df["target"] = (rng.random(n) < 0.3 + 0.3 * (df["x"] > 0)).astype(int)
    df["w"] = rng.integers(1, 5, size=n)
    return df


def make_types(df, weights=None):
    types = ct.ColumnTypes(df)
    types.time_col = "czas"
    if weights is not None:
        types.weights_col = weights
    return types


DF = make_data()
REP = DF.loc[DF.index.repeat(DF["w"])].reset_index(drop=True).drop(columns=["w"])


class TestWeightsCol:
    def test_weights_col_domyslnie_none(self):
        types = ct.ColumnTypes(DF)
        assert types.weights_col is None

    def test_weights_col_setter(self):
        types = make_types(DF, weights="w")
        assert types.weights_col == "w"
        # kolumna wag nie jest zmienną objaśniającą
        role = types.types.loc["w", "role"]
        assert role == ct.Role.WEIGHTS


class TestReportWeighted:
    @pytest.fixture(scope="class")
    def analyses(self):
        types_w = make_types(DF, weights="w")
        types_rep = make_types(REP)
        an_w = DatasetReport(DF, types_w).analyses()
        an_rep = DatasetReport(REP, types_rep).analyses()
        return an_w, an_rep

    def test_te_same_zmienne_analizowane(self, analyses):
        an_w, an_rep = analyses
        # kolumna wag nie pojawia się w raporcie jako zmienna
        assert set(an_w) == set(an_rep) == {"kat", "x"}

    @pytest.mark.parametrize("var", ["kat", "x"])
    def test_buckets_rowne(self, analyses, var):
        an_w, an_rep = analyses
        pd.testing.assert_frame_equal(an_w[var].buckets, an_rep[var].buckets)

    @pytest.mark.parametrize("var", ["kat", "x"])
    def test_dyskretyzacja_rowna(self, analyses, var):
        an_w, an_rep = analyses
        pd.testing.assert_frame_equal(an_w[var].discrete, an_rep[var].discrete)

    @pytest.mark.parametrize("var", ["kat", "x"])
    def test_gini_rowne(self, analyses, var):
        an_w, an_rep = analyses
        pd.testing.assert_frame_equal(an_w[var].gini, an_rep[var].gini)

    @pytest.mark.parametrize("var", ["kat", "x"])
    def test_gini_w_czasie_rowne(self, analyses, var):
        an_w, an_rep = analyses
        pd.testing.assert_frame_equal(
            an_w[var].gini_over_time, an_rep[var].gini_over_time
        )
