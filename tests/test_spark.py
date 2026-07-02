# coding: utf-8
"""
Testy ścieżki Spark (spec/raport-spark.md, etap 4).

Wymagają pyspark + JVM: bez pyspark cały plik jest pomijany
(importorskip), bez działającej JVM — skip w fixturze `spark`.
Równoważność mechanizmu agregat→pseudo-obserwacje jest już dowiedziona
bez Sparka w tests/test_pseudo_obs.py; tutaj sprawdzamy wyłącznie:
- kontrakt `aggregate_variable` (te same sumy co pandasowy groupby),
- pełny przepływ `SparkSource` → `DatasetReport` == ścieżka pandas,
- `column_types_from_spark`.
"""

import numpy as np
import pandas as pd
import pytest

pyspark = pytest.importorskip("pyspark")

import buckets.column_types as ct
import buckets.spark as sp
from buckets.report import DatasetReport

pytestmark = pytest.mark.spark


@pytest.fixture(scope="module")
def spark():
    from pyspark.sql import SparkSession

    try:
        session = (
            SparkSession.builder.master("local[1]")
            .appName("buckets-tests")
            .config("spark.ui.enabled", "false")
            .getOrCreate()
        )
    except Exception as exc:  # brak JVM itp.
        pytest.skip(f"Nie można uruchomić lokalnego Sparka: {exc}")
    yield session
    session.stop()


def make_data(n=300, seed=13):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "kat": rng.choice(["a", "b", "c", None], size=n, p=[0.4, 0.3, 0.2, 0.1]),
        "x": np.round(rng.normal(size=n), 2),
        "czas": rng.choice(["2024-01", "2024-02", "2024-03"], size=n),
    })
    df["target"] = (rng.random(n) < 0.3 + 0.3 * (df["x"] > 0)).astype(int)
    df["w"] = rng.integers(1, 5, size=n)
    return df


DF = make_data()
REP = DF.loc[DF.index.repeat(DF["w"])].reset_index(drop=True).drop(columns=["w"])


@pytest.fixture(scope="module")
def sdf(spark):
    return spark.createDataFrame(DF)


class TestAggregateVariable:
    def test_kontrakt_sum(self, sdf):
        agg = sp.aggregate_variable(sdf, "kat", "target", weights="w")
        assert set(agg.columns) == {"kat", "n_obs", "sum_target"}
        assert agg["n_obs"].sum() == DF["w"].sum()
        assert agg["sum_target"].sum() == (DF["w"] * DF["target"]).sum()
        # null jako osobna grupa
        assert agg["kat"].isna().sum() == 1

    def test_walidacja_brakow_targetu(self, spark):
        df = DF.copy()
        df.loc[0, "target"] = None
        sdf_na = spark.createDataFrame(df)
        with pytest.raises(ValueError, match="braków danych"):
            sp.aggregate_variable(sdf_na, "kat", "target")

    def test_micro_binning_ogranicza_poziomy(self, sdf):
        agg = sp.aggregate_variable(sdf, "x", "target", max_levels=10)
        assert agg["x"].nunique() <= 12  # 10 binów wewn. + 2 skrajne


class TestSparkSourceReport:
    """Pełna równoważność raportu: SparkSource == ścieżka pandas (replikacja)."""

    @pytest.fixture(scope="class")
    def analyses(self, sdf):
        types_spark = sp.column_types_from_spark(sdf)
        types_spark.time_col = "czas"
        types_spark.weights_col = "w"
        source = sp.SparkSource(sdf, types_spark)
        an_spark = DatasetReport(source, source.types).analyses()

        types_rep = ct.ColumnTypes(REP)
        types_rep.time_col = "czas"
        an_rep = DatasetReport(REP, types_rep).analyses()
        return an_spark, an_rep

    def test_te_same_zmienne(self, analyses):
        an_spark, an_rep = analyses
        assert set(an_spark) == set(an_rep) == {"kat", "x"}

    @pytest.mark.parametrize("var", ["kat", "x"])
    def test_buckets_rowne(self, analyses, var):
        an_spark, an_rep = analyses
        pd.testing.assert_frame_equal(an_spark[var].buckets, an_rep[var].buckets)

    @pytest.mark.parametrize("var", ["kat", "x"])
    def test_gini_rowne(self, analyses, var):
        an_spark, an_rep = analyses
        pd.testing.assert_frame_equal(an_spark[var].gini, an_rep[var].gini)

    @pytest.mark.parametrize("var", ["kat", "x"])
    def test_gini_w_czasie_rowne(self, analyses, var):
        an_spark, an_rep = analyses
        pd.testing.assert_frame_equal(
            an_spark[var].gini_over_time, an_rep[var].gini_over_time
        )

    @pytest.mark.parametrize("var", ["kat", "x"])
    def test_dyskretyzacja_rowna(self, analyses, var):
        an_spark, an_rep = analyses
        pd.testing.assert_frame_equal(an_spark[var].discrete, an_rep[var].discrete)

    def test_raport_html_generuje_sie(self, sdf):
        types = sp.column_types_from_spark(sdf)
        types.weights_col = "w"
        source = sp.SparkSource(sdf, types)
        html = DatasetReport(source, source.types).to_html()
        assert "kat" in html and "x" in html


class TestSparkSourceBezWag:
    def test_syntetyczna_kolumna_wag(self, sdf):
        types = sp.column_types_from_spark(sdf.drop("w"))
        source = sp.SparkSource(sdf.drop("w"), types)
        assert source.types.weights_col == sp.PSEUDO_WEIGHTS
        frame = source.frame_for("kat")
        assert sp.PSEUDO_WEIGHTS in frame.columns


class TestColumnTypesFromSpark:
    def test_typy_i_role(self, sdf):
        types = sp.column_types_from_spark(sdf)
        t = types.types
        assert t.loc["kat", "analytical_type"] == ct.AnalyticalType.CATEGORICAL
        assert t.loc["x", "analytical_type"] == ct.AnalyticalType.CONTINUOUS
        assert t.loc["target", "role"] == ct.Role.TARGET
        assert types.target == "target"
