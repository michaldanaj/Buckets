# coding: utf-8
"""
Testy wpięcia sekcji "w czasie" w raport (spec/raport-w-czasie.md, 3.3):
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
            for fig in (va.fig_distribution, va.fig_target_by_bucket,
                        va.fig_target_by_period, va.fig_pit_ttc):
                assert isinstance(fig, plt.Figure)

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

    def test_payload(self, analyses):
        payload = analyses["x"].to_report_payload()
        assert len(payload) == 13
        # ramki pivotowe mają czas jako kolumnę (report_html: index=False)
        assert "czas" in payload[8].columns   # counts_frame
        assert "czas" in payload[12].columns  # avg_target_frame

    def test_raport_html(self):
        types = ct.ColumnTypes(DF.drop(columns=["w"]))
        types.time_col = "czas"
        html = DatasetReport(DF.drop(columns=["w"]), types).to_html()
        assert "kat" in html and "x" in html


class TestBezCzasu:
    def test_pola_none_i_payload_jak_dotychczas(self):
        types = ct.ColumnTypes(DF.drop(columns=["w", "czas"]))
        analyses = DatasetReport(DF.drop(columns=["w", "czas"]), types).analyses()
        va = analyses["x"]
        assert va.dist_over_time is None
        assert va.fig_pit_ttc is None
        payload = va.to_report_payload()
        # dotychczasowy kontrakt + dwa None sekcji PIT/TTC
        assert len(payload) == 7
        assert payload[1] is None and payload[3] is None


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
