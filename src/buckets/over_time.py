# coding: utf-8
"""
Klasa `DistributionOverTime` — rozkład/target/predykcja zmiennej dyskretnej
w czasie.

Zastępuje dawną funkcję `bckt_stats_over_time`, która zwracała listę pozycyjną
`[pivot, pivot_normalized, pivot_target, pivot_pred]`. Tutaj te same wyniki są
dostępne przez nazwane akcesory — co usuwa kruchy kontrakt listy i godzi
rozbieżne warianty API (pojedynczy pivot vs lista czterech).

Model jak w `BucketTable`: rdzeń = JEDEN agregat czas × var, liczony raz
w konstruktorze; surowe obserwacje nie są zatrzymywane. Wszystkie akcesory
wyprowadzają wyniki z rdzenia, a wykresy są metodami obiektu.
Szczegóły: spec/2026-07-07-dist-over-time.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import buckets.trellis as trellis


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

        Konstruktor liczy rdzeń-agregat (sumy ważone w przecięciu czas × var)
        i nie zatrzymuje surowych obserwacji — pamięć O(okresy × buckety).
        """
        if target.isnull().any():
            raise ValueError("W zmiennej 'target' nie może być braków danych!")

        if weights is None:
            weights = pd.Series(np.ones(len(var)), index=var.index)

        self._has_pred = pred is not None
        self._var_order = var_order

        # zawsze float: sumy wag mają jeden dtype niezależnie od tego,
        # czy wagi podano (int/float), czy domyślne jedynki
        w = pd.to_numeric(pd.Series(weights.values), errors="coerce").astype(float)
        agg = pd.DataFrame({
            "czas": czas.values,
            "var": var.values,
            "sum_w": w,
            "sum_w_target": w * pd.Series(target.values).astype(float),
        })
        if self._has_pred:
            # pred bywa Categorical (np. wynik buck.assign przez pd.cut) —
            # koercja do float, żeby ważone średnie działały
            pred_num = pd.to_numeric(
                pd.Series(pred.values).astype("object"), errors="coerce"
            )
            # licznik ważonej średniej pred (pary z brakiem pred pomijane —
            # NaN nie wchodzi do sumy) oraz osobny mianownik (waga tylko tam,
            # gdzie pred istnieje) — dzisiejsza semantyka estim()
            agg["sum_w_pred"] = w * pred_num
            agg["sum_w_pred_obs"] = w.where(pred_num.notna())

        # rdzeń: JEDEN agregat czas × var, niemutowalny nośnik statystyk.
        # observed=False jawnie: gdy `var` jest Categorical, nieużywane poziomy
        # zostają jako kolumny zerowe (zachowanie starego pivot_table) — bez
        # tego przyszły pandas domyślnie by je usunął, zmieniając zestaw kolumn.
        self._core = agg.groupby(["czas", "var"], sort=True, observed=False).sum()

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

    def _pivot(self, col: str) -> pd.DataFrame:
        """Rozwija kolumnę rdzenia do pivotu okres × var (brakujące pary → 0)."""
        pivot = self._core[col].unstack("var").fillna(0.0)
        pivot.index.name = "czas"
        pivot.columns.name = "var"
        return self._order_columns(pivot)

    def counts(self) -> pd.DataFrame:
        """Sumy wag dla każdej wartości `var` w przecięciu z okresami (liczności w czasie)."""
        return self._pivot("sum_w")

    def distribution(self) -> pd.DataFrame:
        """Rozkład znormalizowany — udział wartości `var` w obrębie każdego okresu (suma = 1)."""
        counts = self.counts()
        return counts.div(counts.sum(axis=1), axis=0)

    def avg_target(self) -> pd.DataFrame:
        """Ważona średnia targetu dla każdej wartości `var` w okresie."""
        # mianownik: suma wag; 0 → NA, by 0/0 dało NA zamiast np.nan
        denom = self.counts().replace(0, pd.NA)
        return self._pivot("sum_w_target") / denom

    def avg_pred(self) -> pd.DataFrame | None:
        """Ważona średnia predykcji w okresie, lub None gdy `pred` nie podano."""
        if not self._has_pred:
            return None
        denom = self.counts().replace(0, pd.NA)
        return self._pivot("sum_w_pred") / denom

    # --------------------------------------------------- agregaty per okres
    def avg_target_total(self) -> pd.Series:
        """
        Średni target per okres, po wszystkich wartościach `var`
        (odpowiednik wiersza TOTAL z `avg_t_tbl` w MDBinom) — szereg
        „obserwowany" wykresu PIT/TTC. Z rdzenia: Σ_var sum_w_target / Σ_var sum_w.
        """
        g = self._core.groupby(level="czas", sort=True)
        return g["sum_w_target"].sum() / g["sum_w"].sum()

    def estim(self) -> pd.Series | None:
        """
        Ważona średnia `pred` per okres (odpowiednik `estim` z MDBinom).

        Gdy `pred` to przypisany avg_target bucketu, jest to prognoza targetu
        w okresie wynikająca wyłącznie ze zmiany struktury bucketów — szereg
        „estymowany" wykresu PIT/TTC. None, gdy `pred` nie podano.

        Pary z brakiem `pred` są pomijane w liczniku I mianowniku (osobny
        mianownik `sum_w_pred_obs`) — inaczej braki zaniżałyby średnią. Okres
        bez ani jednej predykcji daje `NaN` (0/0), ale **zostaje w indeksie** —
        indeks `estim()` pokrywa się z `avg_target_total()`, więc `plot_pit_ttc`
        rysuje wtedy lukę zamiast urywać serię.
        """
        if not self._has_pred:
            return None
        g = self._core.groupby(level="czas", sort=True)
        return g["sum_w_pred"].sum() / g["sum_w_pred_obs"].sum()

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
            self._core["sum_w_target"].sum() / self._core["sum_w"].sum()
        )
        wyn.loc["TOTAL"] = total_row
        return wyn

    # ----------------------------------------------------------- wykresy
    # Wykresy jako metody obiektu (spójnie z BucketTable.plot) — delegują do
    # modułu implementacyjnego `trellis`; wołający sam zamyka figurę.
    def plot_distribution(self, title: str | None = None):
        """Panel na bucket; słupki = udział bucketu w kolejnych okresach."""
        return trellis.plot_distribution(self, title)

    def plot_avg_target_by_bucket(self, title: str | None = None):
        """Panel na bucket; średni target w czasie (punkty połączone linią)."""
        return trellis.plot_avg_target_by_bucket(self, title)

    def plot_avg_target_by_period(self, title: str | None = None):
        """Panel na okres; średni target po bucketach."""
        return trellis.plot_avg_target_by_period(self, title)

    def plot_pit_ttc(self, title: str | None = None):
        """Target obserwowany vs estymowany z dyskretyzacji per okres (PIT/TTC)."""
        return trellis.plot_pit_ttc(self, title)
