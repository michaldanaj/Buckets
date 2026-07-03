# coding: utf-8
"""
Testy rozszerzeń `DistributionOverTime` (spec/raport-w-czasie.md, sekcja 3.1):
var_order, avg_target_total, estim, ramki prezentacyjne z TOTAL-ami oraz
równoważność wag (replikacja) i pseudo-obserwacji.
"""

import numpy as np
import pandas as pd
import pytest

from buckets.over_time import DistributionOverTime
from buckets.spark import to_pseudo_obs


# ---------------------------------------------------------- mała ramka ręczna
@pytest.fixture
def maly():
    return DistributionOverTime(
        czas=pd.Series([1, 1, 2, 2]),
        var=pd.Series(["a", "b", "a", "b"]),
        target=pd.Series([0, 1, 1, 1]),
        pred=pd.Series([0.1, 0.2, 0.3, 0.4]),
        weights=pd.Series([1, 1, 2, 1]),
    )


class TestAkcesoryPerOkres:
    def test_avg_target_total(self, maly):
        wyn = maly.avg_target_total()
        assert wyn.loc[1] == pytest.approx(0.5)      # (0*1 + 1*1) / 2
        assert wyn.loc[2] == pytest.approx(1.0)      # (1*2 + 1*1) / 3

    def test_estim(self, maly):
        wyn = maly.estim()
        assert wyn.loc[1] == pytest.approx(0.15)     # (0.1 + 0.2) / 2
        assert wyn.loc[2] == pytest.approx(1.0 / 3)  # (0.3*2 + 0.4) / 3

    def test_estim_z_pred_categorical(self):
        # buck.assign dla ciągłej zwraca Categorical (pd.cut) — estim ma działać
        dot = DistributionOverTime(
            czas=pd.Series([1, 1, 2]),
            var=pd.Series(["a", "b", "a"]),
            target=pd.Series([0, 1, 1]),
            pred=pd.Series(pd.Categorical([0.2, 0.4, 0.2])),
        )
        assert dot.estim().loc[1] == pytest.approx(0.3)
        assert dot.avg_pred().loc[1, "a"] == pytest.approx(0.2)

    def test_estim_none_bez_pred(self):
        dot = DistributionOverTime(
            pd.Series([1, 2]), pd.Series(["a", "a"]), pd.Series([0, 1])
        )
        assert dot.estim() is None


class TestVarOrder:
    def test_kolejnosc_kolumn(self, maly):
        dot = DistributionOverTime(
            czas=pd.Series([1, 1, 2, 2]),
            var=pd.Series(["a", "b", "a", "b"]),
            target=pd.Series([0, 1, 1, 1]),
            var_order=["b", "a"],
        )
        assert list(dot.counts().columns) == ["b", "a"]
        assert list(dot.avg_target().columns) == ["b", "a"]
        assert dot.bucket_order() == ["b", "a"]

    def test_nieznane_poziomy_pomijane_reszta_na_koncu(self):
        dot = DistributionOverTime(
            czas=pd.Series([1, 1]),
            var=pd.Series(["a", "b"]),
            target=pd.Series([0, 1]),
            var_order=["b", "nie_ma_takiego"],
        )
        assert list(dot.counts().columns) == ["b", "a"]


class TestFrames:
    def test_counts_frame(self, maly):
        wyn = maly.counts_frame()
        assert wyn.loc[1].tolist() == [1, 1, 2]
        assert wyn.loc[2].tolist() == [2, 1, 3]
        assert wyn.loc["TOTAL"].tolist() == [3, 2, 5]

    def test_distribution_frame(self, maly):
        wyn = maly.distribution_frame()
        assert wyn.loc[1].tolist() == pytest.approx([0.5, 0.5, 1.0])
        assert wyn.loc["TOTAL"].tolist() == pytest.approx([0.6, 0.4, 1.0])
        # kolumna TOTAL = 1 w każdym okresie
        assert (wyn["TOTAL"] == 1.0).all()

    def test_avg_target_frame(self, maly):
        wyn = maly.avg_target_frame()
        assert wyn.loc[1].tolist() == pytest.approx([0.0, 1.0, 0.5])
        assert wyn.loc[2].tolist() == pytest.approx([1.0, 1.0, 1.0])
        # TOTAL per bucket: a=(0*1+1*2)/3, b=(1+1)/2; narożnik: 4/5
        assert wyn.loc["TOTAL"].tolist() == pytest.approx([2 / 3, 1.0, 0.8])

    def test_model_bez_totali(self, maly):
        # TOTAL tylko w warstwie prezentacji
        assert "TOTAL" not in maly.counts().columns
        assert "TOTAL" not in maly.counts().index


# ------------------------------------------- równoważność: wagi i pseudo-obs
def make_data(n=200, seed=17):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "kat": rng.choice(["a", "b", "c"], size=n),
        "czas": rng.choice(["2024-01", "2024-02", "2024-03"], size=n),
    })
    df["target"] = (rng.random(n) < 0.4).astype(int)
    df["pred"] = np.round(rng.random(n), 3)
    df["w"] = rng.integers(1, 5, size=n)
    return df


def pandas_aggregate(df):
    tmp = df.assign(_w=df["w"], _wt=df["w"] * df["target"],
                    _wp=df["w"] * df["pred"])
    return tmp.groupby(["kat", "czas"], dropna=False).agg(
        n_obs=("_w", "sum"), sum_target=("_wt", "sum"), sum_pred=("_wp", "sum"),
    ).reset_index()


class TestRownowaznosc:
    def test_wagi_jak_replikacja(self):
        df = make_data()
        rep = df.loc[df.index.repeat(df["w"])].reset_index(drop=True)
        dot_w = DistributionOverTime(
            df["czas"], df["kat"], df["target"], pred=df["pred"],
            weights=df["w"].astype(float),
        )
        dot_rep = DistributionOverTime(
            rep["czas"], rep["kat"], rep["target"], pred=rep["pred"]
        )
        pd.testing.assert_series_equal(
            dot_w.avg_target_total(), dot_rep.avg_target_total()
        )
        pd.testing.assert_series_equal(dot_w.estim(), dot_rep.estim())
        pd.testing.assert_frame_equal(dot_w.counts_frame(), dot_rep.counts_frame())
        pd.testing.assert_frame_equal(
            dot_w.avg_target_frame(), dot_rep.avg_target_frame()
        )

    def test_pseudo_obserwacje_jak_surowe(self):
        df = make_data()
        pobs = to_pseudo_obs(pandas_aggregate(df))
        dot_p = DistributionOverTime(
            pobs["czas"], pobs["kat"], pobs["target"], pred=pobs["pred"],
            weights=pobs["weights"].astype(float),
        )
        dot_raw = DistributionOverTime(
            df["czas"], df["kat"], df["target"], pred=df["pred"],
            weights=df["w"].astype(float),
        )
        pd.testing.assert_series_equal(
            dot_p.avg_target_total(), dot_raw.avg_target_total()
        )
        pd.testing.assert_series_equal(dot_p.estim(), dot_raw.estim())
        pd.testing.assert_frame_equal(
            dot_p.distribution_frame(), dot_raw.distribution_frame()
        )
        pd.testing.assert_frame_equal(
            dot_p.avg_target_frame(), dot_raw.avg_target_frame()
        )
