# coding: utf-8
"""
Otwarty model elementów raportu (spec/2026-07-07-raport-kolekcja.md §5,
realizuje spec/2026-05-31-backlog.md pkt 1).

`ReportElement` to wspólny typ elementu, który zna swój rodzaj i potrafi się
wyrenderować do HTML. `VariableAnalysis` jest uporządkowaną kolekcją takich
elementów; `report_html` iteruje je i woła `render_html()`.

Zasada z §3: statystyki to dane, wykresy to renderowanie. Dlatego figury
NIE są stanem — `FigureElement` trzyma **fabrykę** (picklowalna funkcja/metoda
+ argumenty), a figura powstaje dopiero w `render_html`, jest osadzana jako
base64 i zamykana. Dzięki temu kolekcja jest lekka i picklowalna, a render
powtarzalny i bez efektów ubocznych. Walidacja typu wkładu jest w konstruktorze
elementu (błąd przy `add`, nie cicho w renderze) — przejmuje rolę dawnego
`_validate_payload_element`.
"""

from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from io import BytesIO
from typing import Callable

import matplotlib.pyplot as plt
import pandas as pd


class ReportElement(ABC):
    """Wspólny element raportu: zna swój tytuł/klucz i potrafi się wyrenderować."""

    def __init__(self, *, key: str | None = None, title: str | None = None):
        self.key = key
        self.title = title

    @abstractmethod
    def render_html(self) -> str:
        ...


class TableElement(ReportElement):
    """Tabela — opakowuje `pd.DataFrame` (renderowana z `index=False`)."""

    def __init__(self, df: pd.DataFrame, *, key: str | None = None,
                 title: str | None = None):
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"TableElement wymaga pd.DataFrame, a dostał {type(df)!r}."
            )
        super().__init__(key=key, title=title)
        self.df = df

    def render_html(self) -> str:
        stats_html = self.df.to_html(classes="table", border=0, index=False)
        return f"<h3>{self.title or 'Statystyki'}</h3>\n{stats_html}\n"


class TextElement(ReportElement):
    """Tekst / komentarz (Markdown-lite: akapit z zachowaniem łamań wierszy)."""

    def __init__(self, text: str, *, key: str | None = None,
                 title: str | None = None):
        if not isinstance(text, str):
            raise TypeError(
                f"TextElement wymaga str, a dostał {type(text)!r}."
            )
        super().__init__(key=key, title=title)
        self.text = text

    def render_html(self) -> str:
        head = f"<h3>{self.title}</h3>\n" if self.title else ""
        body = self.text.replace("\n", "<br>")
        return f"{head}<p>{body}</p>\n"


def _figure_to_html(fig: "plt.Figure", title: str | None) -> str:
    """Osadza figurę jako base64 PNG (jak dotąd w report_html)."""
    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    buffer.close()
    alt = title or "Wykres"
    return (
        f"<h3>{title or 'Wykres'}</h3>\n"
        f'<img src="data:image/png;base64,{image_base64}" alt="{alt}">\n'
    )


class FigureElement(ReportElement):
    """
    Wykres jako fabryka. Dwa tryby:

    - `factory` (+ `args`): figura powstaje przy renderze i JEST zamykana po
      osadzeniu; `factory` musi być picklowalna (funkcja modułu albo metoda
      picklowalnego obiektu), by kolekcja dała się zapisać do pickle,
    - `figure`: gotowa figura użytkownika (element ad hoc) — render jej NIE
      zamyka (to nie nasza własność); taki element nie jest picklowalny.
    """

    def __init__(self, *, factory: Callable[..., "plt.Figure"] | None = None,
                 args: tuple = (), figure: "plt.Figure | None" = None,
                 key: str | None = None, title: str | None = None):
        if (factory is None) == (figure is None):
            raise ValueError("Podaj dokładnie jedno: `factory` albo `figure`.")
        if figure is not None and not isinstance(figure, plt.Figure):
            raise TypeError(
                f"FigureElement `figure` musi być matplotlib.Figure, "
                f"a jest {type(figure)!r}."
            )
        super().__init__(key=key, title=title)
        self.factory = factory
        self.args = tuple(args)
        self.figure = figure

    def render_html(self) -> str:
        if self.figure is not None:          # cudza figura — nie zamykamy
            return _figure_to_html(self.figure, self.title)
        fig = self.factory(*self.args)       # nasza — tworzymy i zamykamy
        try:
            return _figure_to_html(fig, self.title)
        finally:
            plt.close(fig)
