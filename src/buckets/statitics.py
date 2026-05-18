import pandas as pd
from sklearn.metrics import roc_auc_score


def gini(var, target, by=None):
    """
    Funkcja oblicza współczynnik Giniego dla zmiennej i celu.

    Args:
        var (pd.Series): Zmienna, dla której obliczamy współczynnik Giniego.
        target (pd.Series): Cel binarny (0/1).
        by (pd.Series, optional): Zmienna grupująca (np. okres czasu). Jeśli podana,
            zwraca Series z Gini obliczonym osobno dla każdej grupy.

    Returns:
        float: Współczynnik Giniego (gdy by=None).
        pd.Series: Współczynnik Giniego dla każdej grupy (gdy by podane).
    """
    if by is None:
        return 2 * roc_auc_score(target, var) - 1

    df = pd.DataFrame({"var": var, "target": target, "by": by})
    return df.groupby("by").apply(
        lambda g: 2 * roc_auc_score(g["target"], g["var"]) - 1
    ).rename("gini")
