import sklearn

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import pandas as pd

#TODO: dodać wagi
def make_tree(df: pd.DataFrame, var:str, target: str, max_depth: int = 3) -> DecisionTreeClassifier:
    """
    Tworzy drzewo decyzyjne na podstawie danych.

    Args:
        df (pd.DataFrame): Ramka danych z danymi.
        target (str): Nazwa kolumny docelowej.
        max_depth (int): Maksymalna głębokość drzewa.

    Returns:
        DecisionTreeClassifier: Wytrenuj drzewo decyzyjne.
    """
    X = df[var]
    y = df[target]

    tree = DecisionTreeClassifier(max_depth=max_depth)
    tree.fit(X, y)

    return tree


def extract_leaf_info(tree, feature_names, class_names):
    tree_ = tree.tree_
    feature = tree_.feature
    threshold = tree_.threshold

    leaf_info = []

    def recurse(node, path, lower_bound=None, upper_bound=None):
        if feature[node] != -2:  # jeśli nie liść
            # Lewa gałąź: feature <= threshold
            recurse(tree_.children_left[node], path + [(feature[node], "<=", threshold[node])],
                    lower_bound, threshold[node])
            # Prawa gałąź: feature > threshold
            recurse(tree_.children_right[node], path + [(feature[node], ">", threshold[node])],
                    threshold[node], upper_bound)
        else:
            # Liść: zbierz info
            samples = tree_.n_node_samples[node]
            value = tree_.value[node][0]
            total = value.sum()
            probs = value / total if total > 0 else value
            leaf_info.append({
                "path": path,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "samples": samples,
                "class_counts": value,
                "class_probs": probs,
            })

    recurse(0, [])
    # Zamiana indeksów cech na nazwy
    for leaf in leaf_info:
        leaf['intervals'] = []
        for f_idx, op, thr in leaf['path']:
            leaf['intervals'].append(f"{feature_names[f_idx]} {op} {thr:.2f}")
        # Dodaj czytelne statystyki
        leaf['class_stats'] = {class_names[i]: int(leaf['class_counts'][i]) for i in range(len(class_names))}
        leaf['class_probs'] = {class_names[i]: float(leaf['class_probs'][i]) for i in range(len(class_names))}
    return leaf_info

def extract_leaf_bounds(tree):
    """
    Ekstrahuje granice liści z drzewa decyzyjnego.

    Args:
        tree (DecisionTreeClassifier): Wytrenuj drzewo decyzyjne.
        feature_names (list): Lista nazw cech.

    Returns:
        list: Lista granic liści.
    """
    tree_ = tree.tree_
    feature = tree_.feature
    threshold = tree_.threshold

    leaf_bounds = set()

    def recurse(node, lower_bound=None, upper_bound=None):
        if feature[node] != -2:  # jeśli nie liść
            # Lewa gałąź: feature <= threshold
            recurse(tree_.children_left[node], lower_bound, threshold[node])
            # Prawa gałąź: feature > threshold
            recurse(tree_.children_right[node], threshold[node], upper_bound)
        else:
            # Liść: zbierz info
            if lower_bound is not None:
                leaf_bounds.add(float(lower_bound))
            if upper_bound is not None:
                leaf_bounds.add(float(upper_bound))

    recurse(0)
    ret = list(leaf_bounds)
    ret.sort()
    return ret


# Przykład użycia:
dane = pd.read_csv("data/default of credit card clients.csv")

tree = make_tree(dane, ['AGE'], 'default payment next month', max_depth=3)
plt.figure(figsize=(12, 8))
plot_tree(tree, filled=True, feature_names=['AGE'], class_names=['No Default', 'Default'])
plt.show()

# Przykład użycia:
leaf_info = extract_leaf_info(tree, ['AGE'], ['No Default', 'Default'])

for i, leaf in enumerate(leaf_info):
    print(f"Liść {i+1}:")
    print("  Bounds: ({0}, {1}] ".format(leaf['lower_bound'], leaf['upper_bound']))
    print("  Upper bound:", leaf['upper_bound'])
    print("  Przedziały:", " & ".join(leaf['intervals']))
    print("  Liczba próbek:", leaf['samples'])
    print("  Prawdopodobieństwa klas:", leaf['class_probs'])
    print()

bounds = extract_leaf_bounds(tree)
print("Granice liści:", bounds)
