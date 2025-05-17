# coding: utf-8
"""
## Pakiet buckets

Funkcje generujące statystyki dla zmiennej objaśniającej, z uwzględnieniem
	zmiennej celu (target), oraz opcjonalnie predykcji modelu
"""


__doc__ = """Funkcje generujące statystyki dla zmiennej objaśniającej, z uwzględnieniem
	zmiennej celu (target), oraz opcjonalnie predykcji modelu
    =====================================================================
    =====================================================================   
    bckt_stats - statystyki dla zmiennej dyskretnej
    bckt_cut_stats - statystyki dla zmiennej ciągłej
"""

# TODO: Nazewnictwo. 
# może niech będzie klasa, w której będą funkcje do generowania bucketów:
# bckt_stats -> bckt_discrete
# bckt_cut_stats -> bckt_quantiles
# bckt_tree

# TODO: zająć się obszarami poza krańcami przedziałów

__version__ = 0.1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from buckets.column_types import ColumnTypes
import buckets.tree as tree
import buckets.statitics as st

# TODO: kolumna label zamiast bin?
# TODO: zamiast zamieniać zmienną na stringa zawsze, sprawdzić różne inne
# typy. Przykładowo, dla Categorical może można by zostawić, choć
# co później z <NA> i TOTAL? Można ją wtedy sortować po Categorical.
# TODO: może wydzielić funkcję pomocniczą, działającą na strukturze groupBy,
#       i/lub na jego podstawowych agregatach. Wtedy mając już zrobione
# nie trzeba by go robić jeszcze raz. Przy drugim podejściu, mając np.
# dane w Spark, można by je tam najpierw podagregować, a później wrzucić
# tutaj. No ale z ciągłym chyba już by tak dobrze tutaj nie było
# TODO: Dodać sortowanie po dowolenej kolumnie, aby na koniec Total był ostatnim
#       a NaN pierwszym wierszem
# TODO: Sprawdzić jak będzie z sortowaniem, gdy
# TODO: Dopisać co jest oczekiwanym rezultatem w przypadku pustych kwantyli

NA_BIN_NAME = "<NA>"


def bckt_stats(
    var: pd.Series,
    target: pd.Series,
    pred: pd.Series | None = None,
    total: bool = True,
    min_info: bool = False,
    sort_by: str | None = None,
    ascending: bool = True,
    weights: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Funkcja wyliczająca statystyki targetu i predykcji(jeśli jest dostępna)
    na dyskretnych kolumnach ramki pandasowej.

    Zwraca tabelkę pandasową z wyliczonymi agregatami dla każdej wartości
    zmiennej var.

    Args:
      var: zmienna dyskretna, po której nastąpi grupowanie (kolumna ramki Pandas)
      target: zmienna celu, o wartościach 0 lub 1 (kolumna ramki Pandas)
      pred: opcjonalna predykcja zmiennej celu (kolumna ramki Pandas)
      total: czy dodać w ostatnim wierszu statystyki dla całej próby
      min_info: jeśli True, zostanie ograniczona liczba kolumn do najbardziej
                istotnych
      sort_by: po której kolumnie sortować wynikową tabelę. Bez podania wartości,
                sortowanie będzie zgodne z wynikiem działania group_by
      ascending: czy sortować wyniki rosnąco
      weights: kolumna z wagami
    """
    # sprawdzam braki danych w target
    if any(target.isnull()):
        raise ValueError("W zmiennej 'target' nie może być braków danych!")

    if weights is None:
        weights = pd.Series(np.ones(len(var)))
        weights.index = var.index

    pred_none = False
    if pred is None:
        pred = target
        pred_none = True
        pred.index = var.index

    # jeśli są braki danych, to znaczy że została podana zmienna numeryczna (dyskretna)
    df = pd.DataFrame(
        {"var": var, "target": target, "pred": pred, "weights": weights}
    )

    df["bin"] = var.astype(str)
    nulle = var.isnull()
    df.loc[nulle, "bin"] = NA_BIN_NAME

    df["target_w"] = df.target * df.weights
    df["pred_w"] = df.pred * df.weights

    groupby_struct = df.groupby(by="bin")
    wyn = groupby_struct.agg(
        sum_target=("target_w", "sum"),
        n_obs=("weights", "sum"),
        sum_pred=("pred_w", "sum"),
    )
    wyn["avg_target"] = wyn.sum_target / wyn.n_obs
    wyn["avg_pred"] = wyn.sum_pred / wyn.n_obs
    wyn["pct_obs"] = wyn["n_obs"] / (wyn["n_obs"].sum())

    # Doliczenie totala
    # Dlatego robie to z groupby, bo nie wiadomo czemu agregacja
    # na DataFrame zwraca mi błąd. Muszę taki workaround zrobić
    if total:
        wyn_tot = wyn.copy()
        wyn_tot["bin_tot"] = "TOTAL"
        total_row = wyn_tot.groupby("bin_tot").agg(
            sum_target=("sum_target", "sum"),
            n_obs=("n_obs", "sum"),
            sum_pred=("sum_pred", "sum"),
        )

        total_row["avg_target"] = total_row.sum_target / total_row.n_obs
        total_row["avg_pred"] = total_row.sum_pred / total_row.n_obs
        total_row["pct_obs"] = total_row["n_obs"] / (total_row["n_obs"].sum())
        wyn = pd.concat([wyn, total_row], axis=0)

    wyn["bin"] = wyn.index

    # Dodaję kolumnę discrete zachowującą typ danych wejściowej zmiennej.
    # Z indeksu pobieram unikalne wartości zmiennej
    # Jeśli zmienna była numeryczna, to muszę zrobić konwersję bez zgłaszania
    # błędu w przypadku wystąpienia w indeksie stringu - np. 'TOTAL'
    # stąd trzeba konwersję przeprowadzić z opcją errors (tylko w Series)
    pom = wyn.index.to_series()
    if pd.api.types.is_numeric_dtype(var):
        # Zamiana int na Int, bo w tabelce mam NaN dla Totala
        if var.dtype == "int64":
            pom = pd.to_numeric(pom, errors="coerce").astype("Int64")
        else:
            pom = pd.to_numeric(pom, errors="coerce").astype(var.dtype)


    wyn["discrete"] = pom

    # sortowanie
    if sort_by is not None:
        wyn.sort_values(by=sort_by, ascending=ascending, inplace=True)

    # robię permutację wierszy, aby nulle były na początku
    # a Total na końcu
    temp_df = pd.DataFrame(
        {"i": list(range(wyn.shape[0])), "j": list(range(wyn.shape[0]))}
    )
    temp_df.loc[wyn.index == NA_BIN_NAME, "j"] = -1
    temp_df.loc[wyn.index == "TOTAL", "j"] = wyn.shape[0]
    temp_df.sort_values("j", inplace=True)
    wyn = wyn.iloc[temp_df.i]

    temp_list = list(range(1, wyn.shape[0] + 1))
    # temp_list.append(np.nan)
    wyn["nr"] = temp_list

    # dodaję nadmiarowe kolumny, żeby struktura tabeli była spójna ze
    # strukturą z funkcji dla zmiennej ciągłej
    if min_info:
        columns = ["sum_target", "n_obs", "avg_target", "pct_obs"]
    else:
        columns = [
            "nr",
            "bin",
            "discrete",
            "od",
            "srodek",
            "do",
            "mean",
            "median",
            "sum_target",
            "n_obs",
            "avg_target",
            "pct_obs",
        ]
    if not pred_none:
        columns += ["avg_pred"]

    wyn = wyn.reindex(columns=columns)
    return wyn


# TODO: Sprawdzić, jak to jest z tym domykaniem przedziałów
def bckt_cut_stats(
    variable: pd.Series,
    target: pd.Series,
    pred: pd.Series | None = None,
    weights: pd.Series | None = None,
    bins: int|list[float] = 50,
    total: bool = True,
    plot: bool = False,
    min_info: bool = False,
    sort_by: str | None = None,
    ascending: bool = True,
) -> pd.DataFrame:
    """
    Funkcja dzieląca zmienną ciągłą na kwantyle i wyliczająca statystyki
    targetu i predykcji(jeśli jest dostępna)
    na wyznaczonych przedziałach.

    Po dyskretyzacji zmiennej ciągłej, wykorzystana jest funkcja bckt_stats.

    Zwraca tabelkę pandasową z wyliczonymi agregatami dla przedziałów wartości
    zmiennej variable.

    Args:
       variable: zmienna ciągła, po której nastąpi grupowanie (kolumna ramki Pandas)
       target: zmienna celu, o wartościach 0 lub 1 (kolumna ramki Pandas)
       pred: opcjonalna predykcja zmiennej celu (kolumna ramki Pandas)
       weights: kolumna z wagami
       n: liczba przedziałów
       total: czy dodać w ostatnim wierszu statystyki dla całej próby
       plot: czy wyrysować zależność
       min_info: jeśli True, zostanie ograniczona liczba kolumn do najbardziej
             istotnych
       sort_by: po której kolumnie sortować wynikową tabelę. Bez podania wartości,
                 sortowanie będzie zgodne z wynikiem działania group_by
       ascending: czy sortować wyniki rosnąco
    """

    if weights is None:
        weights = pd.Series(np.ones(len(variable)))
        weights.index = variable.index

    df = pd.DataFrame(
        {
            "variable": variable,
            "target": target,
            "pred": pred,
            "weights": weights,
        }
    )

    # sprawdzam braki danych w target
    if any(target.isnull()):
        raise ValueError("W zmiennej 'target' nie może być braków danych!")

    if isinstance(bins, int):
        kwantyle = pd.Series(
            df.variable.quantile(
                [i / bins for i in range(bins + 1)], interpolation="lower"
            ).drop_duplicates()
        )
    #TODO: dodać testy tego ifa
    elif isinstance(bins, list):
        #bins.insert(0, -np.inf) 
        #bins.append(np.inf)
        kwantyle = pd.Series(bins).sort_values().drop_duplicates()
    else:
        raise ValueError("bins musi być liczbą całkowitą lub listą wartości.")

    df["bin"] = pd.cut(df.variable, kwantyle, include_lowest=True).astype(str)

    # obsługa braków danych w zmiennej
    df["braki"] = df.variable.isnull()
    df.loc[df["braki"], "bin"] = NA_BIN_NAME

    # wywołanie statystyk
    wyn2 = bckt_stats(
        var=df.bin,
        target=df.target,
        pred=pred,
        weights=df.weights,
        total=total,
        sort_by=None,
    )

    groupby_str = df.groupby(by="bin")

    wyn1 = groupby_str.agg(
        median=("variable", "median"),
        mean=("variable", "mean"),
    )

    # doliczenie totala
    if total:
        df.bin = "TOTAL"
        total_series = df.groupby("bin").agg(
            median=("variable", "median"),
            mean=("variable", "mean"),
        )

        wyn1 = pd.concat([wyn1, total_series], axis=0)

    # Robię update wartościami z wyn1
    wyn2.update(wyn1)
    wyn = wyn2
    wyn["bin"] = wyn.index

    # sortuję
    if sort_by is not None:
        wyn.sort_values(by=sort_by, ascending=ascending, inplace=True)
    else:
        wyn.sort_values("median", inplace=True)

    # robię permutację wierszy, aby nulle były na początku
    # a Total na końcu
    temp_df = pd.DataFrame(
        {"i": list(range(wyn.shape[0])), "j": list(range(wyn.shape[0]))}
    )
    temp_df.loc[wyn.index == NA_BIN_NAME, "j"] = -1
    temp_df.loc[wyn.index == "TOTAL", "j"] = wyn.shape[0]
    temp_df.sort_values("j", inplace=True)
    wyn = wyn.iloc[temp_df.i]

    # uzupełniam wartości od, do, srodek
    wyn.loc[~wyn.index.isin(["TOTAL", NA_BIN_NAME]), "od"] = kwantyle.iloc[
        :-1
    ].to_list()
    wyn.loc[~wyn.index.isin(["TOTAL", NA_BIN_NAME]), "do"] = kwantyle.iloc[
        1:
    ].to_list()
    wyn["srodek"] = (wyn["od"] + wyn["do"]) / 2

    # uzupełniam kolumnę nr, bo po przesortowaniu jest bez sensu
    temp_list = list(range(1, wyn.shape[0] + 1))
    # temp_list.append(np.nan)
    wyn["nr"] = temp_list

    # Definicaj jest przy pomocy od-do, dlatego usuwam discrete w tym przypadku
    wyn["discrete"] = np.nan

    if plot:
        plt1 = wyn.plot.scatter("srodek", "avg_target", alpha=0.5, label="target")
        if pred is not None:
            wyn.plot.scatter(
                "srodek",
                "avg_pred",
                ax=plt1,
                color="g",
                alpha=0.5,
                label="predykcja",
            )
        plt1.legend()

    if min_info:
        return wyn[["sum_target", "n_obs", "avg_target", "pct_obs"]]
    return wyn


def plot(bucket, title=None):
    """
    Funkcja rysująca wykres na podstawie danych w bucket.
    Jeśli wszystkie wartości w kolumnie 'srodek' są brakami, rysuje scatter plot,
    gdzie wielkość kropek odzwierciedla kolumnę 'n_obs'.
    W przeciwnym wypadku rysuje scatter plot.
    """
    if 'TOTAL' in bucket.index:
        bucket = bucket.drop(index='TOTAL')

    fig, ax = plt.subplots()  # Tworzenie obiektu Figure i Axes

    if bucket['srodek'].isnull().all():
        x_var = 'bin'
    else:
        x_var = 'srodek'

    # wielkość punktu
    #size = np.sqrt(bucket['n_obs']/(bucket['n_obs'].sum()/bucket.shape[0]))*50
    size = (bucket['n_obs']/(bucket['n_obs'].sum()/bucket.shape[0]))*25

    # Rysowanie scatter plotu z wielkością kropek odzwierciedlającą 'n_obs'
    bucket.plot.scatter(
        x=x_var, 
        y='avg_target', 
        s=size,  # Skalowanie wielkości kropek
        alpha=0.5, 
        legend=True,
        label="target",
        ax=ax  # Użycie wcześniej utworzonego obiektu Axes
    )
    ax.legend()

    ax.set_xlabel("Bin")
    ax.set_ylabel("Avg Target")
    ax.set_title(title)

    return fig  # Zwracanie obiektu Figure

def assign(df, var, buckets, val)-> pd.Series:

    buckets = buckets[buckets.index != "TOTAL"]
    #print('1')
    #print(buckets)

    # Rozdziealam definicje przedziałową od wartości dyskretnych
    buckets_continuous = buckets[~ buckets['od'].isna()]
    buckets_discrete = buckets[buckets['od'].isna()]

    if buckets_continuous.shape[0] > 0:
        # Określamy granice przedziałów
        bins = np.unique(np.sort(buckets_continuous[['od', 'do']].values.flatten()))
        #print('2')
        #print(bins)

        # Określamy etykiety na podstawie kolumny 'val' w buckets
        labels = buckets_continuous[val].values
        
        # Sprawdzamy, czy liczba etykiet jest zgodna z liczbą przedziałów (bins - 1)
        if len(labels) != len(bins) - 1:
            raise ValueError("Liczba etykiet musi odpowiadać liczbie przedziałów.")
        
        #print('---- labels ----')
        #print(labels)
        # Przypisanie odpowiednich przedziałów do wartości z df[var]
        wyn = pd.cut(df[var], bins=bins, labels=labels, include_lowest=True, ordered=False)
    elif buckets_discrete.shape[0] > 0:
        # Przypisanie wartości z kolumny 'val' w buckets do zmiennej df[var]
        # dla wartości dyskretnych
        bins = pd.Series(buckets_discrete[val])
        bins.index =buckets_discrete['discrete']
        wyn = bins[df[var]]
    #print('3')
    #print(wyn)
    return wyn
    

def bckt_tree(
    df: pd.DataFrame,
    var: str,
    target: str,
    max_depth: int = 3,
    min_samples_split: int = 2,
) -> pd.DataFrame:
    """
    Funkcja do generowania drzewa decyzyjnego na podstawie ramki danych.

    Args:
        df: Ramka danych Pandas.
        target: Nazwa kolumny docelowej (target).
        max_depth: Maksymalna głębokość drzewa.
        min_samples_split: Minimalna liczba próbek wymagana do podziału węzła.

    Returns:
        DataFrame z wynikami drzewa decyzyjnego.
    """
    tr = tree.make_tree(df, [var], target, max_depth=max_depth, min_samples_leaf=min_samples_split)
    bounds = tree.extract_leaf_bounds(tr)
    # TODO: ogarnąć poniższe, może z wykorzystaniem Categorical
    bounds.insert(0, df[var].min()-1)
    bounds.append(df[var].max()+1)
    # TODO: dodać resztę parametrów funkcji bckt_cut_stats
    wyn = bckt_cut_stats(variable=df[var], target=df[target], bins=bounds, total=True)
    return wyn


def gen_buckets(df, types: ColumnTypes) -> dict[str, pd.DataFrame]:
    """
    Funkcja do iteracji po kolumnach ramki danych i wywoływania funkcji bckt_stats
    dla zmiennych dyskretnych oraz bckt_cut_stats dla zmiennych ciągłych.

    Args:
        df: Ramka danych Pandas.
        types: Obiekt klasy ColumnTypes.

    Returns:
        None
    """
    results = {}
    
    for index, row in types.types.iterrows():
        if row['role'] == 'target':
           target_col = row['column_name']
    assert target_col is not None, "Nie znaleziono kolumny docelowej (target)."

    for index, row in types.types.iterrows():
        column_name = row['column_name']
        analytical_type = row['analytical_type']
        role = row['role']

        if role in ['skipped', 'target']:
            continue

        if analytical_type == 'discrete':
            #print(f"Analizuję zmienną dyskretną: {column_name}")
            # Wywołanie funkcji bckt_stats
            result = bckt_stats(df[column_name], df[target_col])
            #print(result)

        elif analytical_type == 'continuous':
            #print(f"Analizuję zmienną ciągłą: {column_name}")
            # Wywołanie funkcji bckt_cut_stats
            result = bckt_cut_stats(df[column_name], df[target_col])
            #print(result)
        
        results[column_name] = result
    return results

def gen_report_objects(df: pd.DataFrame, types:ColumnTypes ) -> dict[str, list]:
    """
    Funkcja generująca raport na podstawie wyników funkcji make_loop.

    Args:
        results: Słownik, którego kluczem jest nazwa zmiennej, a wartością DataFrame ze statystykami.

    Returns:
        Słownik, którego kluczem jest nazwa zmiennej, a wartością lista:
        [tabelka ze statystykami, wykres utworzony na jej podstawie].
    """
    buckets = gen_buckets(df, types)
    report = {}


    for variable, stats in buckets.items():

        #####    dyskretyzacja drzewskiem    #####
        if types.types.loc[variable, "analytical_type"] == 'continuous':
            discrete = bckt_tree(df, variable, 'default payment next month', min_samples_split=100)
        else:
            discrete = stats

        ####   gini calc   #####
        x = assign(df, var=variable, buckets=discrete, val='avg_target')

        #####    gini    #####
        gini = pd.DataFrame({'GINI': [st.gini(df[variable], df[types.target])],
                            'GINI discrete': [st.gini(x, df[types.target])]})

        wykres = plot(stats, variable)

        # Dodanie tabelki i wykresu do raportu
        # TODO: dać tu raczej słownik, niż listę
        report[variable] = [gini, discrete, wykres]

    return report

if __name__ == "__main__":

    # Przykład użycia
    df = pd.DataFrame({
        'col1': [1, 2, 3, 4, 5],
        'col2': ['a', 'b', 'a', 'c', 'b'],
        'col3': [1.1, 2.2, 3.3, 4.4, 5.5],
        'col4': [1, 1, 1, 1, 1],
        'target': [0, 1, 0, 1, 0]
    })

    bckt_cut_stats(
        variable=df['col3'],
        target=df['target'],
        #bins=[0, 3, 6],
        bins= 2,
        total=True,
        plot=True
    )

    column_types = ColumnTypes(df, discrete_threshold=3)
    print(column_types.types)
    # Przykładowe dane
    df = pd.DataFrame({
        'value': [2, 5, 8, 15, 25]
    })

    buckets = pd.DataFrame({
        'od': [0, 5, 10, 20],
        'do': [5, 10, 20, 30],
        'fit': [0.1, 3, 1, -0.1]  
    })

    # Wywołanie funkcji
    df2 = assign(df, 'value', buckets,  'fit')
    print(df2)
