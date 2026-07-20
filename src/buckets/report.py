# coding: utf-8
"""
Klasy raportowe: `VariableAnalysis` (komplet analizy jednej zmiennej) oraz
`DatasetReport` (orkiestracja po kolumnach ramki wg ról z `ColumnTypes`).

Dostęp do danych przechodzi przez "źródło per zmienna" (`PandasSource`;
docelowo także `SparkSource` — spec/2026-07-02-raport-spark.md, sekcja 4.2): analiza
dostaje małą ramkę [zmienna, target, czas?, wagi?] zamiast całego zbioru.
Dzięki temu ścieżka pandas i ścieżka zagregowana (pseudo-obserwacje) dzielą
jeden kod raportu.

Na tym etapie `VariableAnalysis` ma sztywne pola (gini, wykresy, dyskretyzacja).
Docelowo ma się stać otwartą kolekcją elementów `ReportElement` — patrz
spec/2026-05-31-backlog.md, pkt 1. `to_report_payload()` zachowuje obecny pozycyjny
kontrakt listy dla report_html na czas migracji.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import pandas as pd

import buckets.buck as buck
import buckets.column_types as ct
import buckets.statitics as st
from buckets.bucket_table import NA_BIN_NAME
from buckets.over_time import DistributionOverTime


def _validate_payload_element(element) -> None:
    """Element raportu musi być DataFrame, matplotlib Figure albo None."""
    if element is None or isinstance(element, (pd.DataFrame, plt.Figure)):
        return
    raise TypeError(
        f"Element raportu musi być pd.DataFrame, matplotlib.Figure lub None, "
        f"a jest {type(element)!r}."
    )


@dataclass
class VariableAnalysis:
    """Komplet analizy jednej zmiennej: buckety, gini, dyskretyzacja i wykresy."""

    name: str
    gini: pd.DataFrame
    buckets: pd.DataFrame
    discrete: pd.DataFrame | None = None
    gini_over_time: pd.DataFrame | None = None
    fig_buckets: "plt.Figure | None" = None
    fig_gini_over_time: "plt.Figure | None" = None
    # sekcje "w czasie" wzorowane na MDBinom (spec/2026-07-03-raport-w-czasie.md);
    # wypełniane tylko gdy zdefiniowano time_col
    dist_over_time: DistributionOverTime | None = None
    fig_distribution: "plt.Figure | None" = None
    fig_target_by_bucket: "plt.Figure | None" = None
    fig_target_by_period: "plt.Figure | None" = None
    fig_pit_ttc: "plt.Figure | None" = None
    skipped: bool = False

    @classmethod
    def build(cls, df: pd.DataFrame, types: ct.ColumnTypes, variable: str,
              buckets: pd.DataFrame) -> "VariableAnalysis":
        """
        Buduje analizę zmiennej na podstawie wyliczonej tabeli `buckets`.

        `df` musi zawierać kolumny: `variable`, `types.target` oraz —
        jeśli zdefiniowane — `types.time_col` i `types.weights_col`
        (wystarczy ramka z `PandasSource.frame_for`).
        """
        # zbyt dużo poziomów — bucket sygnalizuje to kolumną 'warning'
        if buckets.columns[0] == "warning":
            gini = pd.DataFrame({"GINI": [-9.999], "GINI discrete": [-9.999]})
            return cls(name=variable, gini=gini, buckets=buckets, skipped=True)

        is_continuous = types.types.loc[variable, "analytical_type"] == "continuous"
        weights = df[types.weights_col] if types.weights_col is not None else None

        # dyskretyzacja: dla ciągłej — drzewem; dla dyskretnej — same buckety
        if is_continuous:
            discrete = buck.bckt_tree_stats(
                df, variable, types.target, min_samples_split=100,
                weights=types.weights_col,
            )
        else:
            discrete = buckets

        # gini: pełne (oryginalna zmienna) i dyskretne (po przypisaniu binów)
        x = buck.assign(df, var=variable, buckets=discrete, val="avg_target")
        x_orig = df[variable] if is_continuous else x
        gini = pd.DataFrame({
            "GINI": [st.gini(x_orig, df[types.target], weights=weights)],
            "GINI discrete": [st.gini(x, df[types.target], weights=weights)],
        })

        # sekcje "w czasie" (gdy zdefiniowano główną kolumnę czasową)
        gini_over_time = None
        fig_gini_over_time = None
        dist_over_time = None
        fig_distribution = None
        fig_target_by_bucket = None
        fig_target_by_period = None
        fig_pit_ttc = None
        if types.time_col is not None:
            time_series = df[types.time_col]
            if pd.api.types.is_datetime64_any_dtype(time_series):
                time_series = time_series.dt.to_period("M")
            gini_over_time = pd.DataFrame({
                "GINI": st.gini(
                    x_orig, df[types.target], by=time_series, weights=weights
                ),
                "GINI discrete": st.gini(
                    x, df[types.target], by=time_series, weights=weights
                ),
            }).reset_index()
            fig_gini_over_time = buck.plot_gini_over_time(gini_over_time, variable)

            # rozkład/target bucketów w czasie na dyskretyzacji (jak MDBinom):
            # var = etykieta bucketu, pred = przypisany avg_target (-> estim)
            x_label = buck.assign(df, var=variable, buckets=discrete, val="bin")
            x_label = pd.Series(x_label).astype("string").fillna(NA_BIN_NAME)
            # pred dla braków zmiennej: avg_target bucketu <NA> (assign go nie
            # mapuje) — bez tego średnia estymaty != średnia targetu
            pred = pd.to_numeric(pd.Series(x).astype("object"), errors="coerce")
            na_avg = discrete.loc[discrete["bin"] == NA_BIN_NAME, "avg_target"]
            if len(na_avg):
                pred = pred.fillna(float(na_avg.iloc[0]))
            var_order = discrete.loc[discrete["bin"] != "TOTAL", "bin"].tolist()
            dist_over_time = DistributionOverTime(
                time_series, x_label, df[types.target], pred=pred,
                weights=weights, var_order=var_order,
            )
            fig_distribution = dist_over_time.plot_distribution(variable)
            fig_target_by_bucket = dist_over_time.plot_avg_target_by_bucket(variable)
            fig_target_by_period = dist_over_time.plot_avg_target_by_period(variable)
            fig_pit_ttc = dist_over_time.plot_pit_ttc(variable)

        fig_buckets = buck.plot(buckets, variable)

        return cls(
            name=variable, gini=gini, buckets=buckets, discrete=discrete,
            gini_over_time=gini_over_time, fig_buckets=fig_buckets,
            fig_gini_over_time=fig_gini_over_time,
            dist_over_time=dist_over_time,
            fig_distribution=fig_distribution,
            fig_target_by_bucket=fig_target_by_bucket,
            fig_target_by_period=fig_target_by_period,
            fig_pit_ttc=fig_pit_ttc,
        )

    def _pit_ttc_frame(self) -> pd.DataFrame | None:
        """Tabela Observed/Estimated target per okres (jak w MDBinom pod cycle)."""
        if self.dist_over_time is None:
            return None
        dot = self.dist_over_time
        frame = pd.DataFrame({"Observed target": dot.avg_target_total()})
        estim = dot.estim()
        if estim is not None:
            frame["Estimated target"] = estim
        return frame.reset_index(names="czas")

    def to_report_payload(self) -> list:
        """
        Adapter do report_html — pozycyjna lista elementów (gini jako pierwszy).

        Kolejność sekcji jak w raporcie MDBinom (spec/2026-07-03-raport-w-czasie.md, 3.3):
        Discrimination -> PIT/TTC -> Buckets -> Distribution -> Average target.
        Ramki pivotowe wchodzą po reset_index, bo report_html renderuje tabele
        z index=False.
        """
        if self.skipped:
            payload = [self.gini, self.buckets, None]
        else:
            payload = [
                # Discrimination
                self.gini, self.gini_over_time, self.fig_gini_over_time,
                # PIT/TTC
                self.fig_pit_ttc, self._pit_ttc_frame(),
                # Buckets
                self.discrete, self.fig_buckets,
            ]
            if self.dist_over_time is not None:
                dot = self.dist_over_time
                payload += [
                    # Distribution of buckets
                    self.fig_distribution,
                    dot.counts_frame().reset_index(names="czas"),
                    dot.distribution_frame().reset_index(names="czas"),
                    # Average target
                    self.fig_target_by_bucket,
                    self.fig_target_by_period,
                    dot.avg_target_frame().reset_index(names="czas"),
                ]
        for element in payload:
            _validate_payload_element(element)
        return payload


class PandasSource:
    """
    Źródło danych per zmienna dla `DatasetReport` — ścieżka pandas.

    Kontrakt (wspólny z przyszłym `SparkSource`, spec/2026-07-02-raport-spark.md 4.2):
    `frame_for(var)` zwraca ramkę z kolumnami [var, target, czas?, wagi?]
    o oryginalnych nazwach, `n_levels(var)` — liczbę poziomów zmiennej.
    """

    def __init__(self, df: pd.DataFrame, types: ct.ColumnTypes):
        self.df = df
        self.types = types

    def frame_for(self, var: str) -> pd.DataFrame:
        cols = [var, self.types.target]
        for extra in (self.types.time_col, self.types.weights_col):
            if extra is not None:
                cols.append(extra)
        return self.df[cols]

    def n_levels(self, var: str) -> int:
        return self.df[var].nunique()


class DatasetReport:
    """Orkiestracja analiz po kolumnach ramki wg ról z `ColumnTypes`."""

    def __init__(self, df: "pd.DataFrame | PandasSource", types: ct.ColumnTypes,
                 categorical_max_levels: int = 20):
        # pd.DataFrame jest opakowywany w PandasSource; można też podać
        # gotowe źródło (obiekt z frame_for/n_levels), np. SparkSource
        self.source = df if hasattr(df, "frame_for") else PandasSource(df, types)
        self.types = types
        self.categorical_max_levels = categorical_max_levels

    def _buckets_for(self, column_name: str, analytical_type: str) -> pd.DataFrame:
        """Wylicza tabelę bucketów dla jednej kolumny (z obsługą zbyt wielu poziomów)."""
        frame = self.source.frame_for(column_name)
        target = self.types.target
        weights_col = self.types.weights_col
        weights = frame[weights_col] if weights_col is not None else None
        if analytical_type in ("discrete", "categorical"):
            if self.source.n_levels(column_name) > self.categorical_max_levels:
                return pd.DataFrame({"warning": "Too many categorical levels"}, index=[0])
            return buck.bckt_stats(frame[column_name], frame[target], weights=weights)
        elif analytical_type == "continuous":
            return buck.bckt_cut_stats(frame[column_name], frame[target], weights=weights)
        raise ValueError(f"Nieznany typ analityczny: {analytical_type}")

    def buckets(self) -> dict[str, pd.DataFrame]:
        """Lekka ścieżka: same tabele bucketów dla analizowanych kolumn."""
        out = {}
        for _, row in self.types.types.iterrows():
            if row["role"] in ("skipped", "target", "main_time_col", "weights"):
                continue
            out[row["column_name"]] = self._buckets_for(
                row["column_name"], row["analytical_type"]
            )
        return out

    def analyses(self) -> dict[str, VariableAnalysis]:
        """Buduje `VariableAnalysis` dla każdej analizowanej kolumny."""
        return {
            name: VariableAnalysis.build(
                self.source.frame_for(name), self.types, name, buckets
            )
            for name, buckets in self.buckets().items()
        }

    def to_payload(self) -> dict[str, list]:
        """Słownik zmienna → pozycyjna lista elementów (kontrakt report_html)."""
        return {name: va.to_report_payload() for name, va in self.analyses().items()}

    def to_html(self) -> str:
        """Generuje raport HTML, delegując do report_html.generate_report."""
        import buckets.report_html as vr

        return vr.generate_report(self.to_payload())
