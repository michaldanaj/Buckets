# coding: utf-8
"""
Testy poprawności obsługi wag w rdzeniu pakietu (spec/2026-07-02-raport-spark.md, sekcja 3).

Zasada: wynik na danych ważonych wagami całkowitymi musi być identyczny
z wynikiem na danych zreplikowanych wierszowo (wiersz o wadze k -> k wierszy
o wadze 1). To definicja semantyki wag jako krotności obserwacji.

Luki, które te testy obnażają (przed implementacją sekcji 3 spec):
- `from_bins`/`from_quantiles`: mean/median per bin, total mean/median
  i granice kwantylowe liczone BEZ wag,
- `statitics.gini`: brak parametru `weights`,
- `tree.make_tree`/`bckt_tree_stats`: brak `sample_weight` (TODO w tree.py).
"""

import numpy as np
import pandas as pd
import pytest

import buckets.buck as bckt
import buckets.statitics as st
from buckets.bucket_table import BucketTable


# ------------------------------------------------------------------ dane
def make_data(n=200, seed=42):
    """Ciągła zmienna x (z brakami), binarny target y, wagi całkowite w 1..5."""
    rng = np.random.default_rng(seed)
    x = np.round(rng.normal(size=n), 2)
    y = (rng.random(n) < 0.3 + 0.4 * (x > 0)).astype(int)
    w = rng.integers(1, 6, size=n)
    t = rng.integers(0, 3, size=n)  # pseudo-okresy dla gini by
    x[:10] = np.nan  # bin <NA> też ma być zgodny
    return pd.DataFrame({"x": x, "y": y, "w": w, "t": t})


def replicate(df, w_col="w"):
    """Rozwija ramkę wierszowo wg wag całkowitych (waga k -> k wierszy)."""
    return df.loc[df.index.repeat(df[w_col])].reset_index(drop=True)


DF = make_data()
REP = replicate(DF)


# ------------------------------------------------- helpery statystyczne
class TestWeightedHelpers:
    def test_weighted_median_rowna_medianie_rozwinietej(self):
        got = st.weighted_median(DF["x"], DF["w"])
        expected = REP["x"].median()
        assert got == pytest.approx(expected)

    def test_weighted_quantile_lower_rowna_kwantylowi_rozwinietemu(self):
        qs = [i / 10 for i in range(11)]
        expected = REP["x"].quantile(qs, interpolation="lower").to_numpy()
        got = np.array([
            st.weighted_quantile(DF["x"], DF["w"], q, interpolation="lower")
            for q in qs
        ])
        np.testing.assert_allclose(got, expected)

    def test_weighted_quantile_wagi_jednostkowe_jak_pandas(self):
        # przy wagach 1 helper musi odtwarzać dokładnie Series.quantile("lower")
        ones = pd.Series(np.ones(len(DF)))
        qs = [i / 7 for i in range(8)]
        expected = DF["x"].quantile(qs, interpolation="lower").to_numpy()
        got = np.array([
            st.weighted_quantile(DF["x"], ones, q, interpolation="lower")
            for q in qs
        ])
        np.testing.assert_allclose(got, expected)


# --------------------------------------------------------------- from_bins
class TestFromBinsWeighted:
    BINS = [-10.0, -0.5, 0.0, 0.5, 10.0]

    def test_wazone_rowne_zreplikowanym(self):
        wyn_w = BucketTable.from_bins(
            DF["x"], DF["y"], bins=self.BINS, weights=DF["w"]
        ).to_frame()
        wyn_rep = BucketTable.from_bins(REP["x"], REP["y"], bins=self.BINS).to_frame()
        pd.testing.assert_frame_equal(wyn_w, wyn_rep)

    def test_mean_median_per_bin_uwzgledniaja_wagi(self):
        # bin (-10, -0.5]: mean/median ważone różnią się od nieważonych,
        # jeśli wagi nie są stałe w binie — konstruujemy taki przypadek jawnie
        x = pd.Series([-2.0, -1.0, 1.0, 2.0])
        y = pd.Series([0, 1, 0, 1])
        w = pd.Series([3, 1, 1, 3])
        wyn = BucketTable.from_bins(x, y, bins=[-10, 0, 10], weights=w).to_frame()
        # rozwinięcie: [-2,-2,-2,-1] -> mean=-1.75, median=-2; [1,2,2,2] -> mean=1.75, median=2
        assert wyn.loc[wyn["bin"] != "TOTAL", "mean"].tolist() == [-1.75, 1.75]
        assert wyn.loc[wyn["bin"] != "TOTAL", "median"].tolist() == [-2.0, 2.0]
        # wiersz TOTAL: statystyki całej rozwiniętej próby
        # [-2,-2,-2,-1,1,2,2,2] -> mean=0, median=(-1+1)/2=0
        assert wyn.loc["TOTAL", "mean"] == pytest.approx(0.0)
        assert wyn.loc["TOTAL", "median"] == pytest.approx(0.0)


# ----------------------------------------------------------- from_quantiles
class TestFromQuantilesWeighted:
    def test_wazone_rowne_zreplikowanym(self):
        wyn_w = BucketTable.from_quantiles(
            DF["x"], DF["y"], n_bins=5, weights=DF["w"]
        ).to_frame()
        wyn_rep = BucketTable.from_quantiles(REP["x"], REP["y"], n_bins=5).to_frame()
        pd.testing.assert_frame_equal(wyn_w, wyn_rep)


# ---------------------------------------------------------------------- gini
class TestGiniWeighted:
    def test_wazony_rowny_zreplikowanemu(self):
        got = st.gini(DF["x"], DF["y"], weights=DF["w"])
        expected = st.gini(REP["x"], REP["y"])
        assert got == pytest.approx(expected)

    def test_wazony_by_rowny_zreplikowanemu(self):
        got = st.gini(DF["x"], DF["y"], by=DF["t"], weights=DF["w"])
        expected = st.gini(REP["x"], REP["y"], by=REP["t"])
        pd.testing.assert_series_equal(got, expected)


# --------------------------------------------------------------------- drzewo
class TestTreeWeighted:
    def test_bckt_tree_stats_wazone_rowne_zreplikowanym(self):
        wyn_w = bckt.bckt_tree_stats(
            DF.rename(columns={"y": "target"}), "x", "target",
            min_samples_split=30, weights="w",
        )
        wyn_rep = bckt.bckt_tree_stats(
            REP.rename(columns={"y": "target"}), "x", "target",
            min_samples_split=30,
        )
        pd.testing.assert_frame_equal(wyn_w, wyn_rep)
