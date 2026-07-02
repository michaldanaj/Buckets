# coding: utf-8
"""
Testy równoważności: pseudo-obserwacje z kanonicznego agregatu ≡ dane surowe.

To kluczowy test mechanizmu ze spec/raport-spark.md (sekcja 1) i celowo NIE
wymaga Sparka: agregat jest liczony pandasowym `groupby` o tym samym
kontrakcie, który w etapie 4 wyprodukuje `spark.aggregate_variable`.
Jeśli te testy przechodzą, poprawność ścieżki sparkowej sprowadza się do
poprawności samej agregacji (sum w groupBy).
"""

import numpy as np
import pandas as pd
import pytest

import buckets.buck as bckt
import buckets.statitics as st
from buckets.bucket_table import BucketTable
from buckets.over_time import DistributionOverTime
from buckets.spark import to_pseudo_obs


# ------------------------------------------------------------------ dane
def make_data(n=300, seed=7):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "kat": rng.choice(["a", "b", "c", None], size=n, p=[0.4, 0.3, 0.2, 0.1]),
        "x": np.round(rng.normal(size=n), 2),
        "czas": rng.choice(["2024-01", "2024-02", "2024-03"], size=n),
    })
    df["target"] = (rng.random(n) < 0.3 + 0.3 * (df["x"] > 0)).astype(int)
    df["pred"] = np.round(0.3 + 0.3 * (df["x"] > 0) + rng.normal(0, 0.05, n), 3)
    df.loc[:14, "x"] = np.nan  # bin <NA> dla ciągłej
    return df


def pandas_aggregate(df, var, target="target", pred=None, time_col=None):
    """Kanoniczny agregat liczony w pandas — symulacja wyniku Sparka."""
    tmp = pd.DataFrame({var: df[var], "_w": 1, "_wt": df[target]})
    keys = [var] + ([time_col] if time_col else [])
    if time_col:
        tmp[time_col] = df[time_col]
    if pred:
        tmp["_wp"] = df[pred]
    agg = tmp.groupby(keys, dropna=False, observed=False).agg(
        n_obs=("_w", "sum"), sum_target=("_wt", "sum"),
        **({"sum_pred": ("_wp", "sum")} if pred else {}),
    ).reset_index()
    return agg[agg["n_obs"] > 0].reset_index(drop=True)


DF = make_data()


# ------------------------------------------------------------- to_pseudo_obs
class TestToPseudoObs:
    def test_sumy_wag_odtwarzaja_agregat(self):
        agg = pandas_aggregate(DF, "kat")
        pobs = to_pseudo_obs(agg)
        assert pobs["weights"].sum() == len(DF)
        assert (pobs["weights"] * pobs["target"]).sum() == DF["target"].sum()

    def test_walidacja_brakujacych_kolumn(self):
        with pytest.raises(ValueError, match="wymaganych kolumn"):
            to_pseudo_obs(pd.DataFrame({"kat": ["a"], "n_obs": [1]}))

    def test_kolizja_nazw(self):
        agg = pandas_aggregate(DF.rename(columns={"kat": "weights"}), "weights")
        with pytest.raises(ValueError, match="kolidują"):
            to_pseudo_obs(agg)


# ------------------------------------------------- równoważność: dyskretna
class TestDiscreteEquivalence:
    def test_bckt_stats(self):
        pobs = to_pseudo_obs(pandas_aggregate(DF, "kat"))
        wyn_pseudo = bckt.bckt_stats(
            pobs["kat"], pobs["target"], weights=pobs["weights"]
        )
        wyn_raw = bckt.bckt_stats(DF["kat"], DF["target"])
        pd.testing.assert_frame_equal(wyn_pseudo, wyn_raw)

    def test_bckt_stats_z_pred(self):
        pobs = to_pseudo_obs(pandas_aggregate(DF, "kat", pred="pred"))
        wyn_pseudo = bckt.bckt_stats(
            pobs["kat"], pobs["target"], pred=pobs["pred"], weights=pobs["weights"]
        )
        wyn_raw = bckt.bckt_stats(DF["kat"], DF["target"], pred=DF["pred"])
        pd.testing.assert_frame_equal(wyn_pseudo, wyn_raw)


# --------------------------------------------------- równoważność: ciągła
class TestContinuousEquivalence:
    def test_bckt_cut_stats_kwantyle(self):
        # agregat po unikalnych wartościach — binowanie robi dopiero pandas
        pobs = to_pseudo_obs(pandas_aggregate(DF, "x"))
        wyn_pseudo = bckt.bckt_cut_stats(
            pobs["x"], pobs["target"], weights=pobs["weights"], bins=5
        )
        wyn_raw = bckt.bckt_cut_stats(DF["x"], DF["target"], bins=5)
        pd.testing.assert_frame_equal(wyn_pseudo, wyn_raw)

    def test_bckt_tree_stats(self):
        pobs = to_pseudo_obs(pandas_aggregate(DF, "x"))
        wyn_pseudo = bckt.bckt_tree_stats(
            pobs, "x", "target", min_samples_split=30, weights="weights"
        )
        wyn_raw = bckt.bckt_tree_stats(DF, "x", "target", min_samples_split=30)
        pd.testing.assert_frame_equal(wyn_pseudo, wyn_raw)


# ------------------------------------------------------ równoważność: gini
class TestGiniEquivalence:
    def test_gini(self):
        pobs = to_pseudo_obs(pandas_aggregate(DF, "x"))
        got = st.gini(pobs["x"], pobs["target"], weights=pobs["weights"])
        expected = st.gini(DF["x"], DF["target"])
        assert got == pytest.approx(expected)

    def test_gini_w_czasie(self):
        pobs = to_pseudo_obs(pandas_aggregate(DF, "x", time_col="czas"))
        got = st.gini(
            pobs["x"], pobs["target"], by=pobs["czas"], weights=pobs["weights"]
        )
        expected = st.gini(DF["x"], DF["target"], by=DF["czas"])
        pd.testing.assert_series_equal(got, expected)


# ---------------------------------------- równoważność: rozkład w czasie
class TestDistributionOverTimeEquivalence:
    def test_pivoty(self):
        pobs = to_pseudo_obs(pandas_aggregate(DF, "kat", time_col="czas"))
        dot_pseudo = DistributionOverTime(
            pobs["czas"], pobs["kat"], pobs["target"],
            weights=pobs["weights"].astype(float),
        )
        dot_raw = DistributionOverTime(DF["czas"], DF["kat"], DF["target"])
        pd.testing.assert_frame_equal(dot_pseudo.counts(), dot_raw.counts())
        pd.testing.assert_frame_equal(
            dot_pseudo.distribution(), dot_raw.distribution()
        )
        pd.testing.assert_frame_equal(
            dot_pseudo.avg_target(), dot_raw.avg_target(), check_dtype=False
        )
