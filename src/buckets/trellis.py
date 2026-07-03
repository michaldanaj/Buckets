# coding: utf-8
"""
Wykresy w układzie Trellis (lattice) dla `DistributionOverTime` —
odpowiedniki wykresów z raportu MDBinom (spec/raport-w-czasie.md).

Implementacja na `seaborn.FacetGrid` (wprost implementuje układ trellis:
siatka paneli ze wspólnymi osiami, pasek tytułowy nad panelem, zawijanie
paneli). Każda funkcja zwraca `matplotlib.figure.Figure`, którą
`report_html` osadza jak każdy inny wykres.

Odpowiedniki (nazwy plików PNG z przykładowego raportu R w raport_aa/):
- `plot_distribution`         ↔ `* distribution.png`
- `plot_avg_target_by_bucket` ↔ `* target by bucket.png`
- `plot_avg_target_by_period` ↔ `* target over time.png`
- `plot_pit_ttc`              ↔ `* cycle.png`
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from buckets.over_time import DistributionOverTime

#: kolory nawiązujące do lattice/MDBinom
LINE_COLOR = "#0080ff"          # niebieski punkt-linia (type='b' w lattice)
BAR_COLOR = "#a6c8ff"           # jasnoniebieskie słupki barchart
STRIP_GREY = "#d9d9d9"          # szary pasek tytułowy panelu
STRIP_GREEN = "#00ff00"         # zielony pasek ("target by bucket" w MDBinom)
OBSERVED_COLOR = "black"        # PIT/TTC: target obserwowany
ESTIMATED_COLOR = "green"       # PIT/TTC: target estymowany


def _melt(pivot: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """Pivot okres × bucket -> format długi [czas, bucket, value] (jak reshape::melt)."""
    df = pivot.copy()
    df.index.name = "czas"
    df.columns.name = "bucket"
    out = df.reset_index().melt(id_vars="czas", value_name=value_name)
    out["czas"] = out["czas"].astype(str)
    out["bucket"] = out["bucket"].astype(str)
    return out


def _col_wrap(n_panels: int) -> int:
    """Liczba kolumn siatki ~kwadratowa, jak automatyczny układ lattice."""
    return max(1, math.ceil(math.sqrt(n_panels)))


def _facet(data: pd.DataFrame, order: list[str]) -> sns.FacetGrid:
    """Siatka paneli wg `bucket` — sama konstrukcja; styl nakłada `_style`."""
    return sns.FacetGrid(
        data,
        col="bucket",
        col_order=order,
        col_wrap=_col_wrap(len(order)),
        sharex=True,
        sharey=True,
        height=2.2,
        aspect=1.4,
        despine=False,
    )


def _style(
    g: sns.FacetGrid,
    title: str,
    xlabel: str,
    ylabel: str,
    strip_color: str = STRIP_GREY,
) -> None:
    """Styl à la lattice; wołać PO map_dataframe (rysowanie nadpisuje etykiety)."""
    g.set_titles(col_template="{col_name}", size=9)
    for ax in g.axes.flat:
        # pasek tytułowy panelu (strip) — ramka z tłem jak w lattice
        ax.set_title(
            ax.get_title(),
            bbox={"boxstyle": "square,pad=0.3", "facecolor": strip_color,
                  "edgecolor": "black", "linewidth": 0.6},
        )
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
    g.set_axis_labels(xlabel, ylabel, size=9)
    g.figure.suptitle(title, fontweight="bold")
    g.figure.tight_layout()


def _thin_xticks(g: sns.FacetGrid, labels: list[str], max_ticks: int = 8) -> None:
    """Przerzedza etykiety osi X (lattice zlewał daty — tu celujemy lepiej)."""
    step = max(1, math.ceil(len(labels) / max_ticks))
    keep = range(0, len(labels), step)
    for ax in g.axes.flat:
        ax.set_xticks(list(keep))
        ax.set_xticklabels([labels[i] for i in keep])


def plot_distribution(dot: DistributionOverTime, title: str | None = None) -> plt.Figure:
    """
    Panel na bucket; słupki = udział bucketu w kolejnych okresach
    (lattice: `barchart(value ~ okres | bucket)`).
    """
    data = _melt(dot.distribution(), "value")
    okresy = [str(c) for c in dot.distribution().index]
    g = _facet(data, [str(b) for b in dot.bucket_order()])
    g.map_dataframe(sns.barplot, x="czas", y="value", order=okresy,
                    color=BAR_COLOR, edgecolor="black", linewidth=0.5)
    _style(g, f"Distribution of {title}" if title else "Distribution",
           "Date", "Percent in given date")
    _thin_xticks(g, okresy)
    return g.figure


def plot_avg_target_by_bucket(dot: DistributionOverTime,
                              title: str | None = None) -> plt.Figure:
    """
    Panel na bucket; średni target w czasie, punkty połączone linią
    (lattice: `xyplot(value ~ okres | bucket, type='b')`, zielone paski).
    """
    data = _melt(dot.avg_target(), "value")
    okresy = [str(c) for c in dot.avg_target().index]
    g = _facet(data, [str(b) for b in dot.bucket_order()])
    g.map_dataframe(sns.pointplot, x="czas", y="value", order=okresy,
                    color=LINE_COLOR, markers="o", markersize=4, linewidth=1)
    _style(g, title or "", "Date", "Average target", strip_color=STRIP_GREEN)
    _thin_xticks(g, okresy)
    return g.figure


def plot_avg_target_by_period(dot: DistributionOverTime,
                              title: str | None = None) -> plt.Figure:
    """
    Panel na okres; średni target po bucketach
    (lattice: `xyplot(value ~ bucket | okres, type='b')` — "target over time").
    """
    buckets = [str(b) for b in dot.bucket_order()]
    data = _melt(dot.avg_target(), "value").rename(
        columns={"bucket": "poziom", "czas": "bucket"}
    )  # facetujemy po okresie -> okres wchodzi w rolę kolumny "bucket"
    okresy = [str(c) for c in dot.avg_target().index]
    g = _facet(data, okresy)
    g.map_dataframe(sns.pointplot, x="poziom", y="value", order=buckets,
                    color=LINE_COLOR, markers="o", markersize=4, linewidth=1)
    _style(g, title or "", "Bucket", "target")
    _thin_xticks(g, buckets)
    return g.figure


def plot_pit_ttc(dot: DistributionOverTime, title: str | None = None) -> plt.Figure:
    """
    Target obserwowany (czarne punkty) vs estymowany z dyskretyzacji
    (zielone) per okres — wykres PIT/TTC (`* cycle.png` w MDBinom).
    """
    observed = dot.avg_target_total()
    labels = [str(c) for c in observed.index]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(labels, observed.values, "o", color=OBSERVED_COLOR,
            fillstyle="none", label="Observed target")
    estim = dot.estim()
    if estim is not None:
        ax.plot(labels, estim.values, "o", color=ESTIMATED_COLOR,
                fillstyle="none", label="Estimated target")
    ax.set_title(title or "PIT/TTC", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Average target")
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
