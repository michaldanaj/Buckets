# coding: utf-8
"""
Klasy raportowe: `VariableAnalysis` (komplet analizy jednej zmiennej jako
otwarta kolekcja `ReportElement`) oraz `DatasetReport` (trwała kolekcja analiz
po kolumnach ramki wg ról z `ColumnTypes`).

Wzorzec „kolekcja najpierw, raport później" z MDBinom (R) —
spec/2026-07-07-raport-kolekcja.md. `DatasetReport` **posiada** kolekcję analiz:
liczy ją raz (`build()` jawnie lub leniwie przy pierwszym dostępie), pozwala
ją indeksować/modyfikować (`report["age"]`, `rebuild`, `del`) i serializować
(`save`/`load`), a render HTML jest tylko jedną z operacji na kolekcji —
powtarzalną i bez efektów ubocznych (figury powstają z danych w momencie
renderu, nie są stanem).

Dostęp do danych przechodzi przez "źródło per zmienna" (`PandasSource`;
także `SparkSource` — spec/2026-07-02-raport-spark.md, sekcja 4.2): analiza
dostaje małą ramkę [zmienna, target, czas?, wagi?] zamiast całego zbioru.
Dzięki temu ścieżka pandas i ścieżka zagregowana (pseudo-obserwacje) dzielą
jeden kod raportu.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

import buckets.buck as buck
import buckets.column_types as ct
import buckets.statitics as st
from buckets.bucket_table import NA_BIN_NAME
from buckets.over_time import DistributionOverTime
from buckets.report_elements import (
    FigureElement,
    ReportElement,
    TableElement,
    TextElement,
)


@dataclass
class VariableAnalysis:
    """
    Komplet analizy jednej zmiennej jako uporządkowana kolekcja `ReportElement`.

    Trzyma **dane** (ramki, `discrete`, `DistributionOverTime`) — figury NIE są
    stanem, wchodzą do `elements` jako fabryki i powstają dopiero przy renderze.
    `elements` jest jedynym źródłem prawdy dla `report_html`; skróty-właściwości
    (`buckets`, `gini`, …) i nazwany dostęp `va["gini"]` dają wygodę kodowi,
    który wie, czego szuka.
    """

    name: str
    gini: pd.DataFrame
    buckets: pd.DataFrame
    discrete: pd.DataFrame | None = None
    gini_over_time: pd.DataFrame | None = None
    # sekcje "w czasie" wzorowane na MDBinom (spec/2026-07-03-raport-w-czasie.md);
    # wypełniane tylko gdy zdefiniowano time_col
    dist_over_time: DistributionOverTime | None = None
    skipped: bool = False
    elements: list[ReportElement] = field(default_factory=list)

    # ---------------------------------------------------------------- budowa
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
            va = cls(name=variable, gini=gini, buckets=buckets, skipped=True)
            va.add_table(gini, key="gini", title="GINI")
            va.add_text(str(buckets.iloc[0, 0]), key="skip_reason",
                        title="Pominięto")
            return va

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
        dist_over_time = None
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

        va = cls(
            name=variable, gini=gini, buckets=buckets, discrete=discrete,
            gini_over_time=gini_over_time, dist_over_time=dist_over_time,
        )
        va._add_standard_elements()
        return va

    def _add_standard_elements(self) -> None:
        """
        Standardowy zestaw elementów w kolejności sekcji jak w raporcie MDBinom
        (spec/2026-07-03-raport-w-czasie.md, 3.3): Discrimination -> PIT/TTC ->
        Buckets -> Distribution -> Average target. Figury wchodzą jako fabryki
        (funkcje `buck.plot*` / metody `dist_over_time.plot_*` — picklowalne).
        Ramki pivotowe wchodzą po reset_index (report_html renderuje index=False).
        """
        name = self.name

        # Discrimination
        self.add_table(self.gini, key="gini", title="GINI")
        if self.gini_over_time is not None:
            self.add_table(self.gini_over_time, title="GINI w czasie")
            self.add(FigureElement(
                factory=buck.plot_gini_over_time,
                args=(self.gini_over_time, name),
                key="gini_over_time", title="GINI over time",
            ))

        # PIT/TTC
        if self.dist_over_time is not None:
            dot = self.dist_over_time
            self.add(FigureElement(factory=dot.plot_pit_ttc, args=(name,),
                                   key="pit_ttc", title="PIT/TTC"))
            self.add_table(self._pit_ttc_frame(), title="PIT/TTC")

        # Buckets
        if self.discrete is not None:
            self.add_table(self.discrete, title="Dyskretyzacja")
        self.add(FigureElement(factory=buck.plot, args=(self.buckets, name),
                               key="buckets", title="Buckets"))

        # Distribution + Average target
        if self.dist_over_time is not None:
            dot = self.dist_over_time
            self.add(FigureElement(factory=dot.plot_distribution, args=(name,),
                                   key="distribution", title="Distribution"))
            self.add_table(dot.counts_frame().reset_index(names="czas"),
                           title="Liczności")
            self.add_table(dot.distribution_frame().reset_index(names="czas"),
                           title="Rozkład")
            self.add(FigureElement(
                factory=dot.plot_avg_target_by_bucket, args=(name,),
                key="avg_target", title="Average target by bucket",
            ))
            self.add(FigureElement(factory=dot.plot_avg_target_by_period,
                                   args=(name,), title="Average target by period"))
            self.add_table(dot.avg_target_frame().reset_index(names="czas"),
                           title="Średni target")

    def _pit_ttc_frame(self) -> pd.DataFrame:
        """Tabela Observed/Estimated target per okres (jak w MDBinom pod cycle)."""
        dot = self.dist_over_time
        frame = pd.DataFrame({"Observed target": dot.avg_target_total()})
        estim = dot.estim()
        if estim is not None:
            frame["Estimated target"] = estim
        return frame.reset_index(names="czas")

    # -------------------------------------------------------- otwartość API
    def add(self, element: ReportElement) -> ReportElement:
        """Dokłada dowolny element raportu (zwraca go dla wygody)."""
        if not isinstance(element, ReportElement):
            raise TypeError(
                f"Oczekiwano ReportElement, a dostano {type(element)!r}."
            )
        self.elements.append(element)
        return element

    def add_table(self, df, *, key=None, title=None) -> ReportElement:
        return self.add(TableElement(df, key=key, title=title))

    def add_text(self, text, *, key=None, title=None) -> ReportElement:
        return self.add(TextElement(text, key=key, title=title))

    def add_figure(self, fig, *, key=None, title=None) -> ReportElement:
        """Dokłada gotową figurę użytkownika (render jej NIE zamyka)."""
        return self.add(FigureElement(figure=fig, key=key, title=title))

    # ------------------------------------------------ nazwany dostęp / skróty
    def __getitem__(self, key: str) -> ReportElement:
        for el in self.elements:
            if el.key == key:
                return el
        raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        return any(el.key == key for el in self.elements)

    def keys(self) -> list[str]:
        """Klucze nazwanych elementów (pomija elementy bez klucza)."""
        return [el.key for el in self.elements if el.key is not None]

    @property
    def gini_value(self) -> float:
        """Skalar GINI (kolumna 'GINI discrete') — sortowanie i nawigacja raportu."""
        return float(self.gini.iloc[0, 1])

    # --------------------------------------------- fabryki figur poza raportem
    # Kto chce obejrzeć wykres poza raportem — woła metodę (spec §7 pkt 2).
    def plot_buckets(self):
        return buck.plot(self.buckets, self.name)

    def plot_gini_over_time(self):
        return buck.plot_gini_over_time(self.gini_over_time, self.name)

    def plot_distribution(self):
        return self.dist_over_time.plot_distribution(self.name)

    def plot_pit_ttc(self):
        return self.dist_over_time.plot_pit_ttc(self.name)

    def plot_avg_target_by_bucket(self):
        return self.dist_over_time.plot_avg_target_by_bucket(self.name)

    def plot_avg_target_by_period(self):
        return self.dist_over_time.plot_avg_target_by_period(self.name)


class PandasSource:
    """
    Źródło danych per zmienna dla `DatasetReport` — ścieżka pandas.

    Kontrakt (wspólny z `SparkSource`, spec/2026-07-02-raport-spark.md 4.2):
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
    """
    Trwała kolekcja analiz — obiekt roboczy, nie przelot.

    `build()` liczy kolekcję raz i zapisuje ją na obiekcie; dostęp do
    niezbudowanej kolekcji buduje ją leniwie. Kolekcja jest dict-podobna
    (`report["age"]`, `list(report)`, `del report["x"]`) i modyfikowalna
    (`rebuild`, `report["x"] = ...`); `to_html()` niczego nie przelicza ani
    nie mutuje. Serializacja przez pickle pomija źródło danych (`source`) —
    po `load` dostępny jest render i odczyt, `rebuild` wymaga podpięcia danych.
    """

    def __init__(self, df: "pd.DataFrame | PandasSource", types: ct.ColumnTypes,
                 categorical_max_levels: int = 20):
        # pd.DataFrame jest opakowywany w PandasSource; można też podać
        # gotowe źródło (obiekt z frame_for/n_levels), np. SparkSource
        self.source = df if hasattr(df, "frame_for") else PandasSource(df, types)
        self.types = types
        self.categorical_max_levels = categorical_max_levels
        self._analyses: dict[str, VariableAnalysis] | None = None

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

    # ------------------------------------------------------- kolekcja analiz
    def build(self) -> "DatasetReport":
        """Liczy komplet analiz i zapisuje kolekcję na `self` (zwraca self)."""
        self._analyses = {
            name: VariableAnalysis.build(
                self.source.frame_for(name), self.types, name, buckets
            )
            for name, buckets in self.buckets().items()
        }
        return self

    def analyses(self) -> dict[str, VariableAnalysis]:
        """Kolekcja `{zmienna: VariableAnalysis}` — buduje ją leniwie, jeśli trzeba."""
        if self._analyses is None:
            self.build()
        return self._analyses

    def rebuild(self, var: str, buckets: pd.DataFrame | None = None) -> VariableAnalysis:
        """
        Przelicza pojedynczą zmienną (odpowiednik ręcznej redyskretyzacji
        w MDBinom) i podmienia ją w kolekcji. Bez `buckets` liczy je od nowa.
        """
        if buckets is None:
            analytical_type = self.types.types.loc[var, "analytical_type"]
            buckets = self._buckets_for(var, analytical_type)
        va = VariableAnalysis.build(
            self.source.frame_for(var), self.types, var, buckets
        )
        self.analyses()[var] = va
        return va

    # dict-podobny dostęp do kolekcji
    def __getitem__(self, name: str) -> VariableAnalysis:
        return self.analyses()[name]

    def __setitem__(self, name: str, va: VariableAnalysis) -> None:
        self.analyses()[name] = va

    def __delitem__(self, name: str) -> None:
        del self.analyses()[name]

    def __iter__(self):
        return iter(self.analyses())

    def __contains__(self, name: str) -> bool:
        return name in self.analyses()

    def __len__(self) -> int:
        return len(self.analyses())

    # ------------------------------------------------------------- render
    def to_html(self, order: "str | list[str]" = "gini") -> str:
        """
        Generuje raport HTML z kolekcji. `order`: "gini" (domyślnie, malejąco) |
        "alpha" | jawna lista nazw. Powtarzalne, nie mutuje kolekcji.
        """
        import buckets.report_html as vr
        return vr.generate_report(self.analyses(), order=order)

    def save_html(self, filename: str, order: "str | list[str]" = "gini") -> None:
        """Zapisuje raport HTML do pliku."""
        import buckets.report_html as vr
        vr.save(self.to_html(order=order), filename)

    # ---------------------------------------------------------- trwałość
    def save(self, filename: str) -> None:
        """Zapisuje kolekcję do pliku pickle (odpowiednik .RData z MDBinom)."""
        import pickle
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename: str) -> "DatasetReport":
        """Wczytuje kolekcję z pliku pickle (bez źródła danych — patrz `save`)."""
        import pickle
        with open(filename, "rb") as f:
            return pickle.load(f)

    def __getstate__(self) -> dict:
        # źródło danych nie jest serializowane (§7 pkt 3): po load dostępny jest
        # render i odczyt kolekcji; rebuild wymaga ponownego podpięcia danych
        state = self.__dict__.copy()
        state["source"] = None
        return state
