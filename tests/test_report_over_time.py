# coding: utf-8
"""
Testy wpięcia sekcji "w czasie" w raport (spec/2026-07-03-raport-w-czasie.md, 3.3):
pola VariableAnalysis, kolejność/zawartość payloadu, regresja bez time_col
oraz równoważność wag dla rozkładów w czasie.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import buckets.column_types as ct
from buckets.report import DatasetReport


def make_data(n=300, seed=23):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "kat": rng.choice(["a", "b", "c"], size=n, p=[0.5, 0.3, 0.2]),
        "x": np.round(rng.normal(size=n), 2),
        "czas": rng.choice(["2024-01", "2024-02", "2024-03"], size=n),
    })
    df["target"] = (rng.random(n) < 0.3 + 0.3 * (df["x"] > 0)).astype(int)
    df["w"] = rng.integers(1, 5, size=n)
    df.loc[:9, "x"] = np.nan  # bucket <NA>
    return df


DF = make_data()


@pytest.fixture(autouse=True)
def zamykaj_figury():
    yield
    plt.close("all")


class TestZCzasem:
    @pytest.fixture(scope="class")
    def analyses(self):
        types = ct.ColumnTypes(DF.drop(columns=["w"]))
        types.time_col = "czas"
        return DatasetReport(DF.drop(columns=["w"]), types).analyses()

    def test_pola_wypelnione(self, analyses):
        for va in analyses.values():
            assert va.dist_over_time is not None
            # figury nie są stanem — wchodzą jako nazwane elementy-fabryki
            for key in ("pit_ttc", "distribution", "avg_target"):
                assert key in va
                html = va[key].render_html()
                assert "<img" in html

    def test_kolejnosc_bucketow_wg_dyskretyzacji(self, analyses):
        va = analyses["x"]
        oczekiwane = va.discrete.loc[
            va.discrete["bin"] != "TOTAL", "bin"
        ].tolist()
        assert va.dist_over_time.bucket_order() == oczekiwane

    def test_bucket_na_obecny(self, analyses):
        # braki zmiennej tworzą bucket <NA> w rozkładach
        assert "<NA>" in analyses["x"].dist_over_time.bucket_order()

    def test_estim_dostepny(self, analyses):
        # pred = przypisany avg_target -> estim istnieje
        assert analyses["x"].dist_over_time.estim() is not None

    def test_elementy(self, analyses):
        from buckets.report_elements import TableElement

        va = analyses["x"]
        # pełny zestaw sekcji (Discrimination -> PIT/TTC -> Buckets ->
        # Distribution -> Average target) = 13 elementów
        assert len(va.elements) == 13
        # ramki pivotowe mają czas jako kolumnę (report_html: index=False)
        tabele = [el.df for el in va.elements if isinstance(el, TableElement)]
        assert any("czas" in df.columns for df in tabele)

    def test_raport_html(self):
        types = ct.ColumnTypes(DF.drop(columns=["w"]))
        types.time_col = "czas"
        html = DatasetReport(DF.drop(columns=["w"]), types).to_html()
        assert "kat" in html and "x" in html


class TestBezCzasu:
    def test_brak_sekcji_w_czasie(self):
        types = ct.ColumnTypes(DF.drop(columns=["w", "czas"]))
        analyses = DatasetReport(DF.drop(columns=["w", "czas"]), types).analyses()
        va = analyses["x"]
        assert va.dist_over_time is None
        # bez time_col: brak sekcji PIT/TTC i "w czasie"
        assert "pit_ttc" not in va and "distribution" not in va
        # zostają tylko: gini (tabela), dyskretyzacja (tabela), buckets (wykres)
        assert len(va.elements) == 3
        assert "gini" in va and "buckets" in va


class TestWagi:
    def test_rozklady_wazone_jak_replikacja(self):
        rep = DF.loc[DF.index.repeat(DF["w"])].reset_index(drop=True).drop(columns=["w"])
        types_w = ct.ColumnTypes(DF)
        types_w.time_col = "czas"
        types_w.weights_col = "w"
        types_rep = ct.ColumnTypes(rep)
        types_rep.time_col = "czas"

        an_w = DatasetReport(DF, types_w).analyses()
        an_rep = DatasetReport(rep, types_rep).analyses()
        for var in ("kat", "x"):
            dot_w, dot_rep = an_w[var].dist_over_time, an_rep[var].dist_over_time
            pd.testing.assert_frame_equal(
                dot_w.counts_frame(), dot_rep.counts_frame()
            )
            pd.testing.assert_frame_equal(
                dot_w.avg_target_frame(), dot_rep.avg_target_frame()
            )
            pd.testing.assert_series_equal(dot_w.estim(), dot_rep.estim())


class TestInwariantEstim:
    """
    Inwarianty wykrywające błędy z realnych danych (credit_limit z brakami):
    1) buck.assign na ciągłej z pd.NA w Int64 przypisywał ~26% obserwacji do
       złych przedziałów (pd.cut z krawędziami object z kolumn Float64),
    2) estim pomijał obserwacje z brakiem zmiennej (pred=NaN), przez co
       średnia estymaty nie równała się średniej targetu.
    """

    @staticmethod
    def dane_z_brakami(n=2000, seed=5):
        rng = np.random.default_rng(seed)
        df = pd.DataFrame({
            "x": pd.array(rng.integers(10_000, 500_000, size=n), dtype="Int64"),
            "czas": rng.choice(["2024-01", "2024-02", "2024-03"], size=n),
        })
        df["target"] = (rng.random(n) < 0.1 + 0.4 * (df["x"] < 100_000)).astype(int)
        df.loc[df.sample(frac=0.1, random_state=1).index, "x"] = pd.NA
        return df

    def test_assign_poprawny_przy_brakach_w_int64(self):
        import buckets.buck as buck

        df = self.dane_z_brakami()
        discrete = buck.bckt_tree_stats(df, "x", "target", min_samples_split=100)
        x = buck.assign(df, "x", discrete, val="avg_target")
        x = pd.to_numeric(pd.Series(x).astype("object"), errors="coerce")

        # ręczne przypisanie po granicach od/do
        bc = discrete[(discrete["bin"] != "TOTAL") & discrete["od"].notna()]
        oczekiwane = pd.Series(np.nan, index=df.index)
        for _, row in bc.iterrows():
            maska = (df["x"] > row["od"]) & (df["x"] <= row["do"])
            # include_lowest dla pierwszego przedziału
            if row["od"] == bc["od"].min():
                maska = maska | (df["x"] == row["od"])
            oczekiwane[maska.fillna(False)] = float(row["avg_target"])
        pd.testing.assert_series_equal(x, oczekiwane, check_names=False)

    def test_srednia_estymaty_rowna_sredniej_targetu(self):
        df = self.dane_z_brakami()
        types = ct.ColumnTypes(df)
        types.time_col = "czas"
        va = DatasetReport(df, types).analyses()["x"]
        dot = va.dist_over_time

        estim = dot.estim()
        counts = dot.counts_frame()["TOTAL"].drop("TOTAL")
        srednia_estim = (estim * counts).sum() / counts.sum()
        assert srednia_estim == pytest.approx(df["target"].mean(), abs=1e-12)
