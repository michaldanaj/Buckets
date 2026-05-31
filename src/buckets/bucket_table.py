# coding: utf-8
"""
Klasa `BucketTable` — rdzeń statystyk bucketów.

Model danych (otypowana tabela agregatów per-bin) jest oddzielony od prezentacji
(wiersz `TOTAL`, kolejność wierszy, numeracja). `BucketTable` trzyma wyłącznie
wiersze binów (w tym bin braków `<NA>`), nigdy wiersza `TOTAL` — ten powstaje
dopiero w `to_frame(total=True)`. Dzięki temu kontrakt typów jest narzucany
w jednym miejscu (`_canonicalize`) i nie psują go wstrzykiwane do kolumn
wartości `"TOTAL"`/`pd.NA`.

Szczegóły projektu: spec/buck-refaktor-klasy.md, kontrakt typów: spec/typy-danych.md.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd

NA_BIN_NAME = "<NA>"

# Kolumny o stałym, zadeklarowanym typie. `astype` z tej mapy jest aplikowany
# jako ostatni krok kanonizacji i ma "ostatnie słowo" nad `convert_dtypes`.
# Świadomie NIE ma tu `sum_target`/`n_obs` (przy wagach całkowitych mają być
# `Int64`, przy ułamkowych `Float64` — to dobiera `convert_dtypes`) ani
# `discrete` (zachowuje naturalny typ zmiennej wejściowej).
_FIXED_TYPES = {
    "nr": "Int64",
    "bin": "string",
    "od": "Float64",
    "srodek": "Float64",
    "do": "Float64",
    "mean": "Float64",
    "median": "Float64",
    "avg_target": "Float64",
    "avg_pred": "Float64",
    "pct_obs": "Float64",
}

_FULL_COLUMNS = [
    "nr", "bin", "discrete", "od", "srodek", "do", "mean", "median",
    "sum_target", "n_obs", "avg_target", "pct_obs",
]
_MIN_INFO_COLUMNS = ["sum_target", "n_obs", "avg_target", "pct_obs"]


class Kind(Enum):
    """Rodzaj bucketu: dyskretny (grupowanie po wartościach) lub ciągły (przedziały)."""
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"


def _validate_target(target: pd.Series) -> None:
    if target.isnull().any():
        raise ValueError("W zmiennej 'target' nie może być braków danych!")


def _aggregate(
    var: pd.Series,
    target: pd.Series,
    pred: pd.Series | None,
    weights: pd.Series | None,
) -> tuple[pd.DataFrame, bool]:
    """
    Rdzeniowa agregacja: grupuje po `var` i liczy ważone sumy targetu/predykcji.

    Zwraca (core, has_pred), gdzie core jest indeksowany RangeIndex i ma kolumny:
    `discrete` (klucz grupy, naturalny typ), `bin` (string), `sum_target`,
    `n_obs`, `sum_pred`, `avg_target`, `avg_pred`, `pct_obs`. Bin braków (`<NA>`)
    jest osobnym wierszem. Wiersz `TOTAL` NIE jest tu dodawany.
    """
    _validate_target(target)

    has_pred = pred is not None
    if weights is None:
        weights = pd.Series(np.ones(len(var)), index=var.index)
    if pred is None:
        pred = target.copy()
        pred.index = var.index

    df = pd.DataFrame(
        {"var": var.values, "target": target.values,
         "pred": pred.values, "weights": weights.values}
    )
    df["target_w"] = df["target"] * df["weights"]
    df["pred_w"] = df["pred"] * df["weights"]

    grouped = df.groupby(by="var", dropna=False, sort=True, observed=False)
    core = grouped.agg(
        sum_target=("target_w", "sum"),
        n_obs=("weights", "sum"),
        sum_pred=("pred_w", "sum"),
    )

    # discrete = klucz grupy (zachowuje naturalny typ zmiennej wejściowej)
    core["discrete"] = core.index
    # bin = stringowa reprezentacja; braki jako NA_BIN_NAME
    core["bin"] = [NA_BIN_NAME if pd.isna(v) else str(v) for v in core.index]
    core = core.reset_index(drop=True)

    total_n_obs = core["n_obs"].sum()
    core["avg_target"] = core["sum_target"] / core["n_obs"]
    core["avg_pred"] = core["sum_pred"] / core["n_obs"]
    core["pct_obs"] = core["n_obs"] / total_n_obs

    return core, has_pred


class BucketTable:
    """
    Otypowana tabela agregatów per-bin dla jednej zmiennej (bez wiersza TOTAL).

    Tworzona przez fabryki: `from_discrete`, `from_bins`, `from_quantiles`,
    `from_tree`, `from_auto`. Materializacja do DataFrame (z TOTAL, kolejnością,
    numeracją) przez `to_frame`.
    """

    def __init__(
        self,
        core: pd.DataFrame,
        *,
        kind: Kind,
        is_numeric: bool,
        has_pred: bool,
        total_mean: float | None = None,
        total_median: float | None = None,
    ):
        self._core = core
        self.kind = kind
        self.is_numeric = is_numeric
        self.has_pred = has_pred
        self._total_mean = total_mean
        self._total_median = total_median

    # ------------------------------------------------------------------ fabryki
    @classmethod
    def from_discrete(
        cls,
        var: pd.Series,
        target: pd.Series,
        pred: pd.Series | None = None,
        weights: pd.Series | None = None,
    ) -> "BucketTable":
        """Statystyki zmiennej dyskretnej (grupowanie po wartościach)."""
        core, has_pred = _aggregate(var, target, pred, weights)
        # kolumny przedziałowe puste dla zmiennej dyskretnej
        for col in ("od", "srodek", "do", "mean", "median"):
            core[col] = pd.NA
        return cls(
            core,
            kind=Kind.DISCRETE,
            is_numeric=pd.api.types.is_numeric_dtype(var),
            has_pred=has_pred,
        )

    @classmethod
    def from_bins(
        cls,
        variable: pd.Series,
        target: pd.Series,
        *,
        bins: list[float],
        pred: pd.Series | None = None,
        weights: pd.Series | None = None,
    ) -> "BucketTable":
        """
        Statystyki zmiennej ciągłej po jawnie podanych granicach `bins`.

        Jedyna implementacja binowania ciągłego — `from_quantiles` i `from_tree`
        wyznaczają granice i delegują tutaj.
        """
        if not pd.api.types.is_numeric_dtype(variable):
            raise TypeError(
                f"Zmienna 'variable' musi być numeryczna, a jest {variable.dtype}."
            )
        if not pd.api.types.is_numeric_dtype(target):
            raise TypeError(
                f"Zmienna 'target' musi być numeryczna, a jest {target.dtype}."
            )
        _validate_target(target)

        edges = pd.Series(bins).sort_values().drop_duplicates().reset_index(drop=True)
        bin_cat = pd.cut(variable, edges, include_lowest=True, ordered=True)

        core, has_pred = _aggregate(bin_cat, target, pred, weights)

        # mean/median z oryginalnej zmiennej per bin; mapowane po etykiecie bina
        per_bin = pd.DataFrame({"variable": variable.values, "bin_cat": bin_cat.values})
        stats = per_bin.groupby("bin_cat", observed=False).agg(
            mean=("variable", "mean"), median=("variable", "median")
        )
        # mapa: string(interval) -> (od, do). Granice bierzemy z `edges` wg
        # kolejności kategorii, NIE z interval.left — najniższy przedział ma
        # przez include_lowest sztucznie zaniżoną lewą granicę (np. 0.999).
        edge_list = edges.tolist()
        bounds = {}
        for i, interval in enumerate(bin_cat.cat.categories):
            bounds[str(interval)] = (float(edge_list[i]), float(edge_list[i + 1]))
        mean_map = {str(k): v for k, v in stats["mean"].items()}
        median_map = {str(k): v for k, v in stats["median"].items()}

        core["od"] = [bounds.get(b, (pd.NA, pd.NA))[0] for b in core["bin"]]
        core["do"] = [bounds.get(b, (pd.NA, pd.NA))[1] for b in core["bin"]]
        core["srodek"] = (
            pd.to_numeric(core["od"], errors="coerce")
            + pd.to_numeric(core["do"], errors="coerce")
        ) / 2
        core["mean"] = [mean_map.get(b, pd.NA) for b in core["bin"]]
        core["median"] = [median_map.get(b, pd.NA) for b in core["bin"]]
        # przedział definiuje bin — discrete nie niesie informacji (NA, Float64)
        core["discrete"] = pd.array([pd.NA] * len(core), dtype="Float64")

        # porządek domyślny: rosnąco po dolnej granicy przedziału
        core = core.sort_values("od", na_position="last").reset_index(drop=True)

        return cls(
            core,
            kind=Kind.CONTINUOUS,
            is_numeric=True,
            has_pred=has_pred,
            total_mean=float(variable.mean()),
            total_median=float(variable.median()),
        )

    @classmethod
    def from_quantiles(
        cls,
        variable: pd.Series,
        target: pd.Series,
        *,
        n_bins: int = 50,
        pred: pd.Series | None = None,
        weights: pd.Series | None = None,
    ) -> "BucketTable":
        """Binowanie kwantylowe — wyznacza granice z kwantyli i deleguje do `from_bins`."""
        edges = (
            variable.quantile(
                [i / n_bins for i in range(n_bins + 1)], interpolation="lower"
            )
            .drop_duplicates()
            .to_list()
        )
        return cls.from_bins(
            variable, target, bins=edges, pred=pred, weights=weights
        )

    @classmethod
    def from_tree(
        cls,
        df: pd.DataFrame,
        var: str,
        target: str,
        *,
        max_depth: int = 3,
        min_samples_split: int = 2,
        skipna: bool = True,
    ) -> "BucketTable":
        """Binowanie drzewem decyzyjnym — wyznacza granice i deleguje do `from_bins`."""
        import buckets.tree as tree

        df_tree = df[[var, target]].dropna(subset=[var]) if skipna else df[[var, target]]
        tr = tree.make_tree(
            df_tree, [var], target, max_depth=max_depth,
            min_samples_leaf=min_samples_split,
        )
        bounds = tree.extract_leaf_bounds(tr)
        bounds.insert(0, df[var].min() - 1)
        bounds.append(df[var].max() + 1)
        return cls.from_bins(df[var], df[target], bins=bounds)

    @classmethod
    def from_auto(
        cls,
        variable: pd.Series,
        target: pd.Series,
        *,
        pred: pd.Series | None = None,
        weights: pd.Series | None = None,
        bins: int | list[float] = 50,
        discrete_threshold: int = 20,
    ) -> "BucketTable":
        """Dispatch po typie analitycznym (guess_column_type) → odpowiednia fabryka."""
        import buckets.column_types as ct

        analytical_type = ct.guess_column_type(variable, discrete_threshold)
        if analytical_type in ("discrete", "categorical"):
            return cls.from_discrete(variable, target, pred=pred, weights=weights)
        elif analytical_type == "continuous":
            if isinstance(bins, int):
                return cls.from_quantiles(
                    variable, target, n_bins=bins, pred=pred, weights=weights
                )
            return cls.from_bins(
                variable, target, bins=bins, pred=pred, weights=weights
            )
        raise ValueError(f"Nieznany typ analityczny: {analytical_type}")

    # ------------------------------------------------------------- prezentacja
    def _total_row(self) -> dict:
        sum_target = self._core["sum_target"].sum()
        n_obs = self._core["n_obs"].sum()
        sum_pred = self._core["sum_pred"].sum()
        # discrete dla TOTAL: "TOTAL" tylko dla dyskretnej nienumerycznej
        if self.kind == Kind.CONTINUOUS or self.is_numeric:
            discrete = pd.NA
        else:
            discrete = "TOTAL"
        return {
            "bin": "TOTAL",
            "discrete": discrete,
            "sum_target": sum_target,
            "n_obs": n_obs,
            "sum_pred": sum_pred,
            "avg_target": sum_target / n_obs,
            "avg_pred": sum_pred / n_obs,
            "pct_obs": 1.0,
            "od": pd.NA,
            "do": pd.NA,
            "srodek": pd.NA,
            "mean": self._total_mean if self._total_mean is not None else pd.NA,
            "median": self._total_median if self._total_median is not None else pd.NA,
        }

    def _canonicalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.convert_dtypes()
        for col, typ in _FIXED_TYPES.items():
            if col not in df.columns:
                continue
            if typ == "Float64":
                # to_numeric radzi sobie z kolumną object złożoną z samych pd.NA,
                # której astype("Float64") nie potrafi przekonwertować wprost.
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Float64")
            else:
                df[col] = df[col].astype(typ)
        # sum_target/n_obs: Int64 gdy wartości całkowite (wagi całkowite),
        # Float64 przy wagach ułamkowych. Deterministycznie, nie przez inferencję.
        for col in ("sum_target", "n_obs"):
            if col not in df.columns:
                continue
            s = pd.to_numeric(df[col], errors="coerce")
            whole = s.dropna().mod(1).eq(0).all()
            df[col] = s.astype("Int64" if whole else "Float64")
        return df

    def to_frame(
        self,
        total: bool = True,
        min_info: bool = False,
        sort_by: str | None = None,
        ascending: bool = True,
    ) -> pd.DataFrame:
        """
        Materializuje tabelę do DataFrame zgodnego z kontraktem typów.

        Dokleja wiersz `TOTAL` (jeśli `total`), sortuje po `sort_by`, ustawia
        `<NA>` na początek i `TOTAL` na koniec, numeruje `nr` i kanonizuje typy.
        Nie modyfikuje stanu obiektu.
        """
        # Budujemy z rekordów (nie przez concat), bo doklejanie wiersza TOTAL
        # z kolumnami all-NA wywołuje FutureWarning o ustalaniu dtype. Typy i tak
        # narzuca _canonicalize na końcu.
        rows = self._core.to_dict("records")
        if total:
            rows.append(self._total_row())
        wyn = pd.DataFrame(rows)

        if sort_by is not None:
            wyn = wyn.sort_values(by=sort_by, ascending=ascending, kind="stable")

        # permutacja: <NA> na początek, TOTAL na koniec, reszta zachowuje kolejność
        n = len(wyn)
        order_key = []
        for b in wyn["bin"]:
            if b == NA_BIN_NAME:
                order_key.append(-1)
            elif b == "TOTAL":
                order_key.append(n)
            else:
                order_key.append(len(order_key))
        perm = sorted(range(n), key=lambda i: order_key[i])
        wyn = wyn.iloc[perm].reset_index(drop=True)

        wyn["nr"] = range(1, n + 1)
        wyn.index = pd.Index(wyn["bin"].tolist())

        columns = _MIN_INFO_COLUMNS if min_info else list(_FULL_COLUMNS)
        if self.has_pred and not min_info:
            columns = columns + ["avg_pred"]
        wyn = wyn.reindex(columns=columns)

        wyn = self._canonicalize(wyn)
        # discrete dla zmiennej ciągłej to same braki — deterministycznie Float64
        # (bez tego convert_dtypes ustaliłby Int64 dla kolumny all-NA).
        if self.kind == Kind.CONTINUOUS and "discrete" in wyn.columns:
            wyn["discrete"] = wyn["discrete"].astype("Float64")
        return wyn
