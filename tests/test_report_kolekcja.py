# coding: utf-8
"""
Testy trwałej kolekcji analiz `DatasetReport`
(spec/2026-07-07-raport-kolekcja.md): dostęp dict-owy, cache, rebuild,
otwarty model elementów, powtarzalny render bez efektów ubocznych oraz
serializacja pickle (bez źródła danych).
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import buckets.column_types as ct
from buckets.report import DatasetReport, VariableAnalysis
from buckets.report_elements import FigureElement, TableElement, TextElement


def make_df(n=300, seed=11):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "kat": rng.choice(["a", "b", "c"], size=n, p=[0.5, 0.3, 0.2]),
        "x": np.round(rng.normal(size=n), 2),
        "czas": rng.choice(["2024-01", "2024-02", "2024-03"], size=n),
    })
    df["target"] = (rng.random(n) < 0.3 + 0.3 * (df["x"] > 0)).astype(int)
    return df


@pytest.fixture(autouse=True)
def zamykaj_figury():
    yield
    plt.close("all")


def make_report(time=True):
    df = make_df()
    cols = df if time else df.drop(columns=["czas"])
    types = ct.ColumnTypes(cols)
    if time:
        types.time_col = "czas"
    return DatasetReport(cols, types)


class TestDostepDoKolekcji:
    def test_build_cachuje_kolekcje(self):
        r = make_report()
        assert r.analyses() is r.analyses()      # nie przelicza
        assert r["x"] is r.analyses()["x"]

    def test_dict_api(self):
        r = make_report()
        assert set(r) == {"kat", "x"}
        assert "x" in r and "brak" not in r
        assert len(r) == 2
        assert isinstance(r["x"], VariableAnalysis)

    def test_setitem_i_delitem(self):
        r = make_report()
        r["x_kopia"] = r["x"]
        assert "x_kopia" in r and len(r) == 3
        del r["kat"]
        assert "kat" not in r

    def test_rebuild_podmienia_element(self):
        r = make_report()
        stary = r["x"]
        nowy = r.rebuild("x")
        assert r["x"] is nowy and nowy is not stary
        pd.testing.assert_frame_equal(stary.discrete, nowy.discrete)


class TestOtwartyModel:
    def test_add_text(self):
        va = make_report()["x"]
        va.add_text("Uwaga po korekcie 2026-06.", key="uwagi", title="Uwagi")
        assert "uwagi" in va and isinstance(va["uwagi"], TextElement)
        assert "Uwaga po korekcie" in va["uwagi"].render_html()

    def test_add_figure_gotowej_nie_zamyka(self):
        va = make_report()["x"]
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        va.add_figure(fig, key="adhoc")
        assert "<img" in va["adhoc"].render_html()
        assert plt.fignum_exists(fig.number)     # cudza figura żyje dalej

    def test_walidacja_typu_przy_add(self):
        va = make_report()["x"]
        with pytest.raises(TypeError):
            va.add_table("nie ramka")
        with pytest.raises(TypeError):
            va.add(object())

    def test_skipped_to_element_tekstowy_nie_sentinel(self):
        df = make_df()
        types = ct.ColumnTypes(df.drop(columns=["czas"]))
        r = DatasetReport(df.drop(columns=["czas"]), types, categorical_max_levels=2)
        va = r["kat"]                            # 3 poziomy > 2 -> skipped
        assert va.skipped is True
        assert va.gini_value == pytest.approx(-9.999)
        assert isinstance(va["skip_reason"], TextElement)
        assert "Too many" in va["skip_reason"].render_html()


class TestRenderPowtarzalny:
    def test_to_html_powtarzalny_bez_mutacji(self):
        r = make_report()
        h1 = r.to_html()
        h2 = r.to_html()
        assert "Zmienna: x" in h1
        assert h1.count("<img") == h2.count("<img")   # figury nie wyczerpane
        # dane w kolekcji nietknięte po renderze
        assert r["x"].dist_over_time.counts_frame() is not None

    def test_order_alpha_i_jawna_lista(self):
        r = make_report()
        h_alpha = r.to_html(order="alpha")
        assert h_alpha.index('id="kat"') < h_alpha.index('id="x"')
        h_lista = r.to_html(order=["x", "kat"])
        assert h_lista.index('id="x"') < h_lista.index('id="kat"')


class TestTrwalosc:
    def test_pickle_bez_zrodla_i_render_po_load(self, tmp_path):
        r = make_report()                        # z time_col -> figury-fabryki
        r.build()
        sciezka = tmp_path / "analizy.pkl"
        r.save(str(sciezka))
        r2 = DatasetReport.load(str(sciezka))
        assert r2.source is None                 # źródło nie serializowane
        assert set(r2) == {"kat", "x"}
        # render z wczytanej kolekcji działa (fabryki figur są picklowalne)
        assert "Zmienna: x" in r2.to_html()
        # statystyki przetrwały serializację
        pd.testing.assert_frame_equal(
            r["x"].dist_over_time.counts_frame(),
            r2["x"].dist_over_time.counts_frame(),
        )
