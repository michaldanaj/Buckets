# coding: utf-8
"""
Klasy raportowe: `VariableAnalysis` (komplet analizy jednej zmiennej) oraz
`DatasetReport` (orkiestracja po kolumnach ramki wg ról z `ColumnTypes`).

Na tym etapie `VariableAnalysis` ma sztywne pola (gini, wykresy, dyskretyzacja).
Docelowo ma się stać otwartą kolekcją elementów `ReportElement` — patrz
spec/backlog.md, pkt 1. `to_report_payload()` zachowuje obecny pozycyjny
kontrakt listy dla report_html na czas migracji.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import pandas as pd

import buckets.buck as buck
import buckets.column_types as ct
import buckets.statitics as st


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
    skipped: bool = False

    @classmethod
    def build(cls, df: pd.DataFrame, types: ct.ColumnTypes, variable: str,
              buckets: pd.DataFrame) -> "VariableAnalysis":
        """Buduje analizę zmiennej na podstawie wyliczonej tabeli `buckets`."""
        # zbyt dużo poziomów — bucket sygnalizuje to kolumną 'warning'
        if buckets.columns[0] == "warning":
            gini = pd.DataFrame({"GINI": [-9.999], "GINI discrete": [-9.999]})
            return cls(name=variable, gini=gini, buckets=buckets, skipped=True)

        is_continuous = types.types.loc[variable, "analytical_type"] == "continuous"

        # dyskretyzacja: dla ciągłej — drzewem; dla dyskretnej — same buckety
        if is_continuous:
            discrete = buck.bckt_tree_stats(
                df, variable, types.target, min_samples_split=100
            )
        else:
            discrete = buckets

        # gini: pełne (oryginalna zmienna) i dyskretne (po przypisaniu binów)
        x = buck.assign(df, var=variable, buckets=discrete, val="avg_target")
        x_orig = df[variable] if is_continuous else x
        gini = pd.DataFrame({
            "GINI": [st.gini(x_orig, df[types.target])],
            "GINI discrete": [st.gini(x, df[types.target])],
        })

        # gini w czasie (gdy zdefiniowano główną kolumnę czasową)
        gini_over_time = None
        fig_gini_over_time = None
        if types.time_col is not None:
            time_series = df[types.time_col]
            if pd.api.types.is_datetime64_any_dtype(time_series):
                time_series = time_series.dt.to_period("M")
            gini_over_time = pd.DataFrame({
                "GINI": st.gini(x_orig, df[types.target], by=time_series),
                "GINI discrete": st.gini(x, df[types.target], by=time_series),
            }).reset_index()
            fig_gini_over_time = buck.plot_gini_over_time(gini_over_time, variable)

        fig_buckets = buck.plot(buckets, variable)

        return cls(
            name=variable, gini=gini, buckets=buckets, discrete=discrete,
            gini_over_time=gini_over_time, fig_buckets=fig_buckets,
            fig_gini_over_time=fig_gini_over_time,
        )

    def to_report_payload(self) -> list:
        """Adapter do report_html — pozycyjna lista elementów (gini jako pierwszy)."""
        if self.skipped:
            payload = [self.gini, self.buckets, None]
        else:
            payload = [
                self.gini, self.gini_over_time, self.fig_gini_over_time,
                self.discrete, self.fig_buckets,
            ]
        for element in payload:
            _validate_payload_element(element)
        return payload


class DatasetReport:
    """Orkiestracja analiz po kolumnach ramki wg ról z `ColumnTypes`."""

    def __init__(self, df: pd.DataFrame, types: ct.ColumnTypes,
                 categorical_max_levels: int = 20):
        self.df = df
        self.types = types
        self.categorical_max_levels = categorical_max_levels

    def _buckets_for(self, column_name: str, analytical_type: str) -> pd.DataFrame:
        """Wylicza tabelę bucketów dla jednej kolumny (z obsługą zbyt wielu poziomów)."""
        target = self.types.target
        if analytical_type in ("discrete", "categorical"):
            if self.df[column_name].nunique() > self.categorical_max_levels:
                return pd.DataFrame({"warning": "Too many categorical levels"}, index=[0])
            return buck.bckt_stats(self.df[column_name], self.df[target])
        elif analytical_type == "continuous":
            return buck.bckt_cut_stats(self.df[column_name], self.df[target])
        raise ValueError(f"Nieznany typ analityczny: {analytical_type}")

    def buckets(self) -> dict[str, pd.DataFrame]:
        """Lekka ścieżka: same tabele bucketów dla analizowanych kolumn."""
        out = {}
        for _, row in self.types.types.iterrows():
            if row["role"] in ("skipped", "target", "main_time_col"):
                continue
            out[row["column_name"]] = self._buckets_for(
                row["column_name"], row["analytical_type"]
            )
        return out

    def analyses(self) -> dict[str, VariableAnalysis]:
        """Buduje `VariableAnalysis` dla każdej analizowanej kolumny."""
        return {
            name: VariableAnalysis.build(self.df, self.types, name, buckets)
            for name, buckets in self.buckets().items()
        }

    def to_payload(self) -> dict[str, list]:
        """Słownik zmienna → pozycyjna lista elementów (kontrakt report_html)."""
        return {name: va.to_report_payload() for name, va in self.analyses().items()}

    def to_html(self) -> str:
        """Generuje raport HTML, delegując do report_html.generate_report."""
        import buckets.report_html as vr

        return vr.generate_report(self.to_payload())
