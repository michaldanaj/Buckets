from sklearn.metrics import roc_auc_score

def gini(var, target):
    """
    Funkcja oblicza współczynnik Giniego dla zmiennej i celu.

    Args:
        var (pd.Series): Zmienna, dla której obliczamy współczynnik Giniego.
        target (pd.Series): Cel, dla którego obliczamy współczynnik Giniego.

    Returns:
        float: Współczynnik Giniego.
    """
    return 2*roc_auc_score(target, var) - 1
    