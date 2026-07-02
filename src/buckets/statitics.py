import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def _clean_xw(x, weights) -> tuple[np.ndarray, np.ndarray]:
    """Konwertuje (x, weights) na pary numpy float bez braków w x."""
    x = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
    w = pd.to_numeric(pd.Series(weights), errors="coerce").to_numpy(dtype=float)
    mask = ~np.isnan(x)
    return x[mask], w[mask]


def weighted_quantile(x, weights, q: float, interpolation: str = "linear") -> float:
    """
    Kwantyl ważony, zgodny z `Series.quantile` na danych rozwiniętych wg wag.

    Semantyka wag = krotność obserwacji: dla wag całkowitych wynik jest
    DOKŁADNIE równy `pd.Series(np.repeat(x, weights)).quantile(q, interpolation)`.
    Dla wag ułamkowych — naturalne uogólnienie (element i pokrywa przedział
    pozycji [cumw_{i-1}, cumw_i) w ciągu o długości sum(wag)).

    Args:
        x: wartości (braki są pomijane wraz ze swoimi wagami).
        weights: wagi nieujemne.
        q: rząd kwantyla w [0, 1].
        interpolation: "linear" (jak Series.median) lub "lower"
            (jak Series.quantile(..., interpolation="lower")).
    """
    x, w = _clean_xw(x, weights)
    if len(x) == 0:
        return float("nan")
    order = np.argsort(x, kind="stable")
    x, w = x[order], w[order]
    cumw = np.cumsum(w)
    total = cumw[-1]

    def value_at(pos: float) -> float:
        # element pokrywający pozycję pos: pierwszy i, dla którego cumw[i] > pos
        idx = np.searchsorted(cumw, pos, side="right")
        return x[min(idx, len(x) - 1)]

    p = q * (total - 1)
    if interpolation == "lower":
        return float(value_at(np.floor(p)))
    if interpolation == "linear":
        lo, hi = value_at(np.floor(p)), value_at(np.ceil(p))
        return float(lo + (p - np.floor(p)) * (hi - lo))
    raise ValueError(f"Nieznana interpolacja: {interpolation!r}")


def weighted_median(x, weights) -> float:
    """Mediana ważona — równa `Series.median()` na danych rozwiniętych wg wag."""
    return weighted_quantile(x, weights, 0.5, interpolation="linear")


def weighted_mean(x, weights) -> float:
    """Średnia ważona sum(w*x)/sum(w); braki w x pomijane wraz z wagami."""
    x, w = _clean_xw(x, weights)
    if len(x) == 0 or w.sum() == 0:
        return float("nan")
    return float(np.average(x, weights=w))


def gini(var: pd.Series, target: pd.Series, by: pd.Series = None,
         weights: pd.Series | None = None, skipna: bool = True) -> float | pd.Series:
    """
    Funkcja oblicza współczynnik Giniego dla zmiennej i celu.

    Args:
        var (pd.Series): Zmienna, dla której obliczamy współczynnik Giniego.
            Może zawierać NaN — obsługa zależy od parametru skipna.
        target (pd.Series): Cel binarny (0/1).
        by (pd.Series, optional): Zmienna grupująca (np. okres czasu). Jeśli podana,
            zwraca Series z Gini obliczonym osobno dla każdej grupy.
        weights (pd.Series, optional): Wagi obserwacji. Dla wag całkowitych wynik
            jest równy gini na danych zreplikowanych wierszowo wg wag.
        skipna (bool): Jeśli True (domyślnie), wiersze z NaN w var są pomijane.
            Jeśli False, zwraca NaN gdy var zawiera jakikolwiek NaN.

    Returns:
        float: Współczynnik Giniego (gdy by=None).
        pd.Series: Współczynnik Giniego dla każdej grupy (gdy by podane).
    """
    def _gini(v, t, w):
        if not skipna and v.isna().any():
            return np.nan
        mask = v.notna()
        v, t = v[mask], t[mask]
        sample_weight = None if w is None else np.asarray(w[mask], dtype=float)
        return 2 * roc_auc_score(t, v, sample_weight=sample_weight) - 1

    if by is None:
        return _gini(var, target, weights)

    df = pd.DataFrame({"var": var, "target": target, "by": by})
    if weights is not None:
        df["weights"] = weights
    return df.groupby("by").apply(
        lambda g: _gini(g["var"], g["target"], g.get("weights")),
        include_groups=False,
    ).rename("gini")
