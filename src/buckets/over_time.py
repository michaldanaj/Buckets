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
        var_order: list | None = None,
    ):
        """
        Args:
            czas: podział na okresy.
            var: zmienna dyskretna (np. etykieta bucketu).
            target: cel binarny, bez braków.
            pred: opcjonalna predykcja (np. przypisany avg_target bucketu) —
                źródło `estim()`.
            weights: wagi obserwacji (krotność).
            var_order: kolejność poziomów `var` w kolumnach pivotów
                (np. kolejność wierszy tabeli dyskretyzacji — odpowiednik
                `ordered(levels=...)` z MDBinom). Poziomy spoza danych są
                pomijane; poziomy nienazwane lądują na końcu.
        """
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
            # pred bywa Categorical (np. wynik buck.assign przez pd.cut) —
            # koercja do float, żeby ważone średnie działały
            self._df["pred"] = pd.to_numeric(
                pd.Series(pred.values).astype("object"), errors="coerce"
            )

        self._var_order = var_order
        self._cache: dict[str, pd.DataFrame] = {}

    @property
    def has_pred(self) -> bool:
        return self._has_pred

    def _order_columns(self, pivot: pd.DataFrame) -> pd.DataFrame:
        """Ustawia kolumny pivotu w kolejności `var_order` (reszta na końcu)."""
        if self._var_order is None:
            return pivot
        known = [v for v in self._var_order if v in pivot.columns]
        rest = [c for c in pivot.columns if c not in known]
        return pivot[known + rest]

    def _weighted_pivot(self, value_col: str) -> pd.DataFrame:
        """Pivot sumy `weights * value_col` w przecięciu czas × var."""
        tmp = self._df.copy()
        tmp["_wv"] = tmp["weights"] * tmp[value_col]
        return self._order_columns(tmp.pivot_table(
            index="czas", columns="var", values="_wv", aggfunc="sum", fill_value=0
        ))

    def counts(self) -> pd.DataFrame:
        """Sumy wag dla każdej wartości `var` w przecięciu z okresami (liczności w czasie)."""
        if "counts" not in self._cache:
            self._cache["counts"] = self._order_columns(self._df.pivot_table(
                index="czas", columns="var", values="weights",
                aggfunc="sum", fill_value=0,
            ))
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

    # --------------------------------------------------- agregaty per okres
    def _weighted_mean_by_time(self, value_col: str) -> pd.Series:
        """Ważona średnia `value_col` per okres: sum(w*v)/sum(w)."""
        tmp = self._df
        num = (tmp["weights"] * tmp[value_col]).groupby(tmp["czas"]).sum()
        den = tmp["weights"].groupby(tmp["czas"]).sum()
        return num / den

    def avg_target_total(self) -> pd.Series:
        """
        Średni target per okres, po wszystkich wartościach `var`
        (odpowiednik wiersza TOTAL z `avg_t_tbl` w MDBinom) — szereg
        „obserwowany" wykresu PIT/TTC.
        """
        return self._weighted_mean_by_time("target")

    def estim(self) -> pd.Series | None:
        """
        Ważona średnia `pred` per okres (odpowiednik `estim` z MDBinom).

        Gdy `pred` to przypisany avg_target bucketu, jest to prognoza targetu
        w okresie wynikająca wyłącznie ze zmiany struktury bucketów — szereg
        „estymowany" wykresu PIT/TTC. None, gdy `pred` nie podano.
        """
        if not self._has_pred:
            return None
        return self._weighted_mean_by_time("pred")

    def bucket_order(self) -> list:
        """Kolejność poziomów `var` używana w kolumnach pivotów."""
        return list(self.counts().columns)

    # --------------------------------------------- prezentacja (z TOTAL-ami)
    # Model (counts/distribution/avg_target) nie zna TOTAL-i — jak
    # w BucketTable, TOTAL istnieje tylko w warstwie prezentacji.
    def counts_frame(self) -> pd.DataFrame:
        """Liczności okres × bucket + kolumna TOTAL (okres) i wiersz TOTAL (bucket)."""
        wyn = self.counts().copy()
        wyn["TOTAL"] = wyn.sum(axis=1)
        wyn.loc["TOTAL"] = wyn.sum(axis=0)
        return wyn

    def distribution_frame(self) -> pd.DataFrame:
        """
        Udziały bucketów w obrębie okresu + kolumna TOTAL (=1) i wiersz TOTAL
        (rozkład bucketów w całej próbie) — odpowiednik `pct_all_tbl` z MDBinom.
        """
        counts = self.counts_frame()
        return counts.div(counts["TOTAL"], axis=0)

    def avg_target_frame(self) -> pd.DataFrame:
        """
        Średni target okres × bucket + kolumna TOTAL (średnia okresu) i wiersz
        TOTAL (średnia bucketu w całym czasie) — odpowiednik `avg_t_tbl` z MDBinom.
        """
        wyn = self.avg_target().copy()
        wyn["TOTAL"] = self.avg_target_total()
        counts = self.counts()
        sum_target = (self.avg_target() * counts).sum(axis=0)
        total_row = sum_target / counts.sum(axis=0)
        total_row["TOTAL"] = (
            (self._df["weights"] * self._df["target"]).sum()
            / self._df["weights"].sum()
        )
        wyn.loc["TOTAL"] = total_row
        return wyn
