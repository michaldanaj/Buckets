# coding: utf-8
"""
Klasa `DistributionOverTime` — rozkład/target/predykcja zmiennej dyskretnej
w czasie.

Zastępuje dawną funkcję `bckt_stats_over_time`, która zwracała listę pozycyjną
`[pivot, pivot_normalized, pivot_target, pivot_pred]`. Tutaj te same wyniki są
dostępne przez nazwane akcesory — co usuwa kruchy kontrakt listy i godzi
rozbieżne warianty API (pojedynczy pivot vs lista czterech).

Pivoty liczone leniwie i cache'owane. Szczegóły: spec/buck-refaktor-klasy.md (4.4).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class DistributionOverTime:
    def __init__(
        self,
        czas: pd.Series,
        var: pd.Series,
        target: pd.Series,
        pred: pd.Series | None = None,
        weights: pd.Series | None = None,
    ):
        if target.isnull().any():
            raise ValueError("W zmiennej 'target' nie może być braków danych!")

        if weights is None:
            weights = pd.Series(np.ones(len(var)), index=var.index)

        self._has_pred = pred is not None
        self._df = pd.DataFrame(
            {
                "czas": czas.values,
                "var": var.values,
                "target": target.values,
                "weights": weights.values,
            }
        )
        if self._has_pred:
            self._df["pred"] = pred.values

        self._cache: dict[str, pd.DataFrame] = {}

    @property
    def has_pred(self) -> bool:
        return self._has_pred

    def _weighted_pivot(self, value_col: str) -> pd.DataFrame:
        """Pivot sumy `weights * value_col` w przecięciu czas × var."""
        tmp = self._df.copy()
        tmp["_wv"] = tmp["weights"] * tmp[value_col]
        return tmp.pivot_table(
            index="czas", columns="var", values="_wv", aggfunc="sum", fill_value=0
        )

    def counts(self) -> pd.DataFrame:
        """Sumy wag dla każdej wartości `var` w przecięciu z okresami (liczności w czasie)."""
        if "counts" not in self._cache:
            self._cache["counts"] = self._df.pivot_table(
                index="czas", columns="var", values="weights",
                aggfunc="sum", fill_value=0,
            )
        return self._cache["counts"]

    def distribution(self) -> pd.DataFrame:
        """Rozkład znormalizowany — udział wartości `var` w obrębie każdego okresu (suma = 1)."""
        if "distribution" not in self._cache:
            counts = self.counts()
            self._cache["distribution"] = counts.div(counts.sum(axis=1), axis=0)
        return self._cache["distribution"]

    def avg_target(self) -> pd.DataFrame:
        """Ważona średnia targetu dla każdej wartości `var` w okresie."""
        if "avg_target" not in self._cache:
            # mianownik: suma wag; 0 → NA, by 0/0 dało NA zamiast np.nan
            denom = self.counts().replace(0, pd.NA)
            self._cache["avg_target"] = self._weighted_pivot("target") / denom
        return self._cache["avg_target"]

    def avg_pred(self) -> pd.DataFrame | None:
        """Ważona średnia predykcji w okresie, lub None gdy `pred` nie podano."""
        if not self._has_pred:
            return None
        if "avg_pred" not in self._cache:
            denom = self.counts().replace(0, pd.NA)
            self._cache["avg_pred"] = self._weighted_pivot("pred") / denom
        return self._cache["avg_pred"]
