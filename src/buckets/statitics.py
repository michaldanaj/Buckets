import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def gini(var: pd.Series, target: pd.Series, by: pd.Series=None, skipna: bool = True) -> float | pd.Series:
    """
    Funkcja oblicza współczynnik Giniego dla zmiennej i celu.

    Args:
        var (pd.Series): Zmienna, dla której obliczamy współczynnik Giniego.
            Może zawierać NaN — obsługa zależy od parametru skipna.
        target (pd.Series): Cel binarny (0/1).
        by (pd.Series, optional): Zmienna grupująca (np. okres czasu). Jeśli podana,
            zwraca Series z Gini obliczonym osobno dla każdej grupy.
        skipna (bool): Jeśli True (domyślnie), wiersze z NaN w var są pomijane.
            Jeśli False, zwraca NaN gdy var zawiera jakikolwiek NaN.

    Returns:
        float: Współczynnik Giniego (gdy by=None).
        pd.Series: Współczynnik Giniego dla każdej grupy (gdy by podane).
    """
    def _gini(v, t):
        if not skipna and v.isna().any():
            return np.nan
        mask = v.notna()
        v, t = v[mask], t[mask]
        return 2 * roc_auc_score(t, v) - 1

    if by is None:
        return _gini(var, target)

    df = pd.DataFrame({"var": var, "target": target, "by": by})
    return df.groupby("by").apply(
        lambda g: _gini(g["var"], g["target"])
    ).rename("gini")
