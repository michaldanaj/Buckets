# coding: utf-8
"""
Most między danymi zagregowanymi (np. w Sparku) a pandasowym rdzeniem pakietu.

Projekt: spec/raport-spark.md. Kontrakt kanonicznego agregatu (sekcja 1):
ramka pandas z jedną grupą na wiersz, o kolumnach:

- kolumna zmiennej (dowolna nazwa; braki jako osobna grupa),
- opcjonalne kolumny dodatkowe przenoszone bez zmian (np. okres czasowy),
- ``n_obs``      — suma wag w grupie,
- ``sum_target`` — suma ``waga * target`` w grupie,
- ``sum_pred``   — suma ``waga * pred`` w grupie (opcjonalna).

`to_pseudo_obs` zamienia taki agregat na ważone pseudo-obserwacje, które
przechodzą przez istniejący, ważony pipeline (BucketTable, gini, drzewo,
DistributionOverTime) dając wyniki IDENTYCZNE jak dane wierszowe — dowód
równoważności: tabela w spec/raport-spark.md, sekcja 1; testy:
tests/test_pseudo_obs.py.

Ten moduł nie importuje pyspark na poziomie modułu — funkcje czysto
pandasowe działają bez Sparka; funkcje sparkowe importują pyspark leniwie.
"""

from __future__ import annotations

import pandas as pd

#: kolumny statystyk kanonicznego agregatu (wszystkie pozostałe kolumny
#: agregatu są przenoszone do pseudo-obserwacji bez zmian)
STAT_COLUMNS = ("n_obs", "sum_target", "sum_pred")

#: nazwy kolumn wynikowych pseudo-obserwacji
PSEUDO_TARGET = "target"
PSEUDO_WEIGHTS = "weights"
PSEUDO_PRED = "pred"


def to_pseudo_obs(agg: pd.DataFrame) -> pd.DataFrame:
    """
    Zamienia kanoniczny agregat na ważone pseudo-obserwacje.

    Każdy wiersz agregatu rozpada się na dwa wiersze (rozbicie targetu 0/1):

    - ``target=1`` z wagą ``sum_target``,
    - ``target=0`` z wagą ``n_obs - sum_target``,

    oba z ``pred = sum_pred / n_obs`` (gdy ``sum_pred`` obecne). Wiersze
    o wadze 0 są pomijane. Rozbicie 0/1 (a nie ``target=avg_target``)
    zachowuje całkowitość sum — kontrakt Int64 dla ``sum_target``/``n_obs``
    (spec/raport-spark.md, 5.1) — oraz dokładność drzewa i gini.

    Args:
        agg: kanoniczny agregat (patrz docstring modułu). Kolumny spoza
            ``STAT_COLUMNS`` (zmienna, okres czasowy, ...) są przenoszone
            do wyniku bez zmian.

    Returns:
        Ramka pseudo-obserwacji: kolumny przeniesione + ``target``,
        ``weights`` (+ ``pred``).
    """
    missing = [c for c in ("n_obs", "sum_target") if c not in agg.columns]
    if missing:
        raise ValueError(f"Agregat nie ma wymaganych kolumn: {missing}.")

    carry = [c for c in agg.columns if c not in STAT_COLUMNS]
    collisions = set(carry) & {PSEUDO_TARGET, PSEUDO_WEIGHTS, PSEUDO_PRED}
    if collisions:
        raise ValueError(
            f"Kolumny agregatu kolidują z nazwami pseudo-obserwacji: {collisions}."
        )

    ones = agg[carry].copy()
    ones[PSEUDO_TARGET] = 1
    ones[PSEUDO_WEIGHTS] = agg["sum_target"]

    zeros = agg[carry].copy()
    zeros[PSEUDO_TARGET] = 0
    zeros[PSEUDO_WEIGHTS] = agg["n_obs"] - agg["sum_target"]

    if "sum_pred" in agg.columns:
        avg_pred = agg["sum_pred"] / agg["n_obs"]
        ones[PSEUDO_PRED] = avg_pred
        zeros[PSEUDO_PRED] = avg_pred

    out = pd.concat([ones, zeros], ignore_index=True)
    return out[out[PSEUDO_WEIGHTS] > 0].reset_index(drop=True)
