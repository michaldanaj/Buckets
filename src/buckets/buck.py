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
import buckets.column_types as ct
from buckets.bucket_table import BucketTable
from buckets.over_time import DistributionOverTime

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


def bckt_stats_over_time(
    czas: pd.Series,
    var: pd.Series,
    target: pd.Series,
    pred: pd.Series | None = None,
    weights: pd.Series | None = None,
) -> DistributionOverTime:
    """
    Buduje `DistributionOverTime` ze statystykami zmiennej dyskretnej w czasie.

    Zwraca obiekt z nazwanymi akcesorami (`counts`, `distribution`, `avg_target`,
    `avg_pred`) zamiast dawnej listy pozycyjnej.

    Args:
      czas: zmienna czasowa (kolumna ramki Pandas)
      var: zmienna dyskretna, po której nastąpi grupowanie (kolumna ramki Pandas)
      target: zmienna celu, o wartościach 0 lub 1 (kolumna ramki Pandas)
      pred: opcjonalna predykcja zmiennej celu (kolumna ramki Pandas)
      weights: kolumna z wagami
    """
    return DistributionOverTime(czas, var, target, pred=pred, weights=weights)


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

    Returns:
      Zwraca tabelkę pandasową z wyliczonymi agregatami dla każdej wartości. Typy są zgodne
      z wersją Pandas 2.
      zmiennej var. Struktura tabeli:
      - bin [str]: wartość zmiennej var
      - discrete: wartość zmiennej var
      - od: dolna granica przedziału
      - srodek: środek przedziału
      - do: górna granica przedziału
      - mean: średnia wartość zmiennej target
      - median: mediana wartości zmiennej target
    """
    return BucketTable.from_discrete(
        var, target, pred=pred, weights=weights
    ).to_frame(total=total, min_info=min_info, sort_by=sort_by, ascending=ascending)


# TODO: Sprawdzić, jak to jest z tym domykaniem przedziałów
def bckt_cut_stats(
    variable: pd.Series,
    target: pd.Series,
    pred: pd.Series | None = None,
    weights: pd.Series | None = None,
    bins: int | list[float] = 50,
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

    if isinstance(bins, int):
        bt = BucketTable.from_quantiles(
            variable, target, n_bins=bins, pred=pred, weights=weights
        )
    elif isinstance(bins, list):
        bt = BucketTable.from_bins(
            variable, target, bins=bins, pred=pred, weights=weights
        )
    else:
        raise ValueError("bins musi być liczbą całkowitą lub listą wartości.")

    wyn = bt.to_frame(total=total, sort_by=sort_by, ascending=ascending)

    if plot:
        plt1 = wyn.plot.scatter("srodek", "avg_target", alpha=0.5, label="target")
        if pred is not None:
            wyn.plot.scatter(
                "srodek", "avg_pred", ax=plt1, color="g", alpha=0.5, label="predykcja"
            )
        plt1.legend()

    if min_info:
        return wyn[["sum_target", "n_obs", "avg_target", "pct_obs"]]
    return wyn


def plot(bucket, title=None):
    """
    Funkcja rysująca wykres na podstawie danych w bucket.
    Jeśli wszystkie wartości w kolumnie 'median' są brakami, rysuje scatter plot,
    gdzie wielkość kropek odzwierciedla kolumnę 'n_obs'.
    W przeciwnym wypadku rysuje scatter plot.
    """

    bucket = bucket.copy()
    if "TOTAL" in bucket.index:
        bucket = bucket.drop(index="TOTAL")

    fig, ax = plt.subplots()  # Tworzenie obiektu Figure i Axes

    if bucket["median"].isnull().all():
        x_var = "bin"
    else:
        x_var = "median"

    # wielkość punktu
    # size = np.sqrt(bucket['n_obs']/(bucket['n_obs'].sum()/bucket.shape[0]))*50
    bucket["size"] = (
        (bucket["n_obs"] / (bucket["n_obs"].sum() / bucket.shape[0])) * 25
    ).astype(float)

    # Rysowanie scatter plotu z wielkością kropek odzwierciedlającą 'n_obs'
    bucket.plot.scatter(
        x=x_var,
        y="avg_target",
        s="size",
        alpha=0.5,
        legend=True,
        label="target",
        ax=ax,  # Użycie wcześniej utworzonego obiektu Axes
    )
    ax.legend()

    ax.set_xlabel("Bin")
    ax.set_ylabel("Avg Target")
    ax.set_title(title)

    return fig  # Zwracanie obiektu Figure


def plot_gini_over_time(gini_over_time: pd.DataFrame, title=None):
    labels = gini_over_time["by"].astype(str)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(title)

    ax1.plot(labels, gini_over_time["GINI"] * 100, marker="o")
    ax1.set_title("GINI")
    ax1.set_xlabel("Miesiąc")
    ax1.set_ylabel("GINI (%)")
    ax1.tick_params(axis="x", rotation=45)

    ax2.plot(labels, gini_over_time["GINI discrete"] * 100, marker="o")
    ax2.set_title("GINI discrete")
    ax2.set_xlabel("Miesiąc")
    ax2.set_ylabel("GINI (%)")
    ax2.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    return fig


def assign(df, var, buckets, val) -> pd.Series:
    buckets = buckets[buckets.index != "TOTAL"]
    # print('1')
    # print(buckets)

    # Rozdziealam definicje przedziałową od wartości dyskretnych
    buckets_continuous = buckets[~buckets["od"].isna()]
    buckets_discrete = buckets[buckets["od"].isna()]

    # TODO: obsłużyć przypadek gdy buckets zawiera jednocześnie wiersze continuous i discrete (union)
    if buckets_continuous.shape[0] > 0:
        # Określamy granice przedziałów
        bins = np.unique(np.sort(buckets_continuous[["od", "do"]].values.flatten()))
        # print('2')
        # print(bins)

        # Określamy etykiety na podstawie kolumny 'val' w buckets
        labels = buckets_continuous[val].values

        # Sprawdzamy, czy liczba etykiet jest zgodna z liczbą przedziałów (bins - 1)
        if len(labels) != len(bins) - 1:
            raise ValueError("Liczba etykiet musi odpowiadać liczbie przedziałów.")

        # print('---- labels ----')
        # print(labels)
        # Przypisanie odpowiednich przedziałów do wartości z df[var]
        wyn = pd.cut(
            df[var], bins=bins, labels=labels, include_lowest=True, ordered=False
        )
    elif buckets_discrete.shape[0] > 0:
        # Przypisanie wartości z kolumny 'val' w buckets do zmiennej df[var]
        # dla wartości dyskretnych
        bins = pd.Series(
            buckets_discrete[val].values, index=buckets_discrete["discrete"]
        )
        wyn = df[var].map(bins)
    # print('3')
    # print(wyn)
    return wyn


def bckt_tree_stats(
    df: pd.DataFrame,
    var: str,
    target: str,
    max_depth: int = 3,
    min_samples_split: int = 2,
    skipna: bool = True,
) -> pd.DataFrame:
    """
    Funkcja do generowania drzewa decyzyjnego na podstawie ramki danych.

    Args:
        df: Ramka danych Pandas.
        target: Nazwa kolumny docelowej (target).
        max_depth: Maksymalna głębokość drzewa.
        min_samples_split: Minimalna liczba próbek wymagana do podziału węzła.
        skipna: Jeśli True (domyślnie), wiersze z NaN w var są pomijane przy
            budowie drzewa. Pełny df (z NaN) jest używany do statystyk bucketu.

    Returns:
        DataFrame z wynikami drzewa decyzyjnego.
    """
    return BucketTable.from_tree(
        df, var, target, max_depth=max_depth,
        min_samples_split=min_samples_split, skipna=skipna,
    ).to_frame(total=True)


def bckt_guessed_type_stats(
    variable: pd.Series,
    target: pd.Series,
    pred: pd.Series | None = None,
    weights: pd.Series | None = None,
    bins: int | list[float] = 50,
    total: bool = True,
    plot: bool = False,
    min_info: bool = False,
    sort_by: str | None = None,
    ascending: bool = True,
    discrete_threshold: int = 20,
    categorical_max_levels: int = 20,
) -> pd.DataFrame:

    ct.guess_column_type(variable)
    analytical_type = ct.guess_column_type(variable, discrete_threshold)

    if analytical_type in ["discrete", "categorical"]:
        # print(f"Analizuję zmienną dyskretną: {column_name}")
        # TODO: zobaczyć, jak było ogarnięte w R, żeby jednak robić statystyki zmiennej
        # numerycznej, określonej jako dyskretna. A może i tak jest lepiej?
        # Najpierw zmienną numeryczną klasyfikujemy jako dyskretną, żeby później stwierdzić,
        # że jest ich za dużo i nie robić statystyk? Uspójnić to jakoś.
        if variable.nunique() > categorical_max_levels:
            result = pd.DataFrame({"warning": "Too many categorical levels"}, index=[0])
        else:
            # Wywołanie funkcji bckt_stats
            result = bckt_stats(
                var=variable,
                target=target,
                pred=pred,
                weights=weights,
                total=total,
                min_info=min_info,
                sort_by=sort_by,
                ascending=ascending,
            )
        # print(result)

    elif analytical_type == "continuous":
        # print(f"Analizuję zmienną ciągłą: {column_name}")
        # Wywołanie funkcji bckt_cut_stats
        result = bckt_cut_stats(
            variable=variable,
            target=target,
            pred=pred,
            weights=weights,
            bins=bins,
            total=total,
            min_info=min_info,
            sort_by=sort_by,
            ascending=ascending,
        )
        # print(result)
    else:
        raise ValueError(f"Nieznany typ analityczny: {analytical_type}")

    if plot:
        globals()["plot"](result, title=variable.name)

    return result


def gen_buckets_for_df(
    df: pd.DataFrame, types: ct.ColumnTypes, categorical_max_levels: int = 20
) -> dict[str, pd.DataFrame]:
    """
    Funkcja do iteracji po kolumnach ramki danych i wywoływania funkcji bckt_stats
    dla zmiennych dyskretnych oraz bckt_cut_stats dla zmiennych ciągłych.

    Jeśli zmienna dyskretna ma zbyt wiele poziomów, statystyki nie zostaną wygenerowane.

    Args:
        df: Ramka danych Pandas.
        types: Obiekt klasy ColumnTypes.
        categorical_max_levels: Maksymalna liczba poziomów dla zmiennych dyskretnych.

    Returns:
        None
    """
    from buckets.report import DatasetReport

    return DatasetReport(df, types, categorical_max_levels).buckets()


def gen_report_objects(
    df: pd.DataFrame, types: ct.ColumnTypes, max_levels: int = 20
) -> dict[str, list]:
    """
    Funkcja generująca raport ze statystykami dla zmiennych w ramce danych,
    opisanych w `types`.

    Args:
        types: Obiekt klasy ColumnTypes.
        max_levels: Maksymalna liczba poziomów dla zmiennych dyskretnych.
        df: Ramka danych Pandas.

    Returns:
        Słownik, którego kluczem jest nazwa zmiennej, a wartością lista:
        [tabelka ze statystykami, wykres utworzony na jej podstawie].
    """
    from buckets.report import DatasetReport

    return DatasetReport(df, types, categorical_max_levels=max_levels).to_payload()


if __name__ == "__main__":
    # Przykład użycia
    df = pd.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["a", "b", "a", "c", "b"],
            "col3": [1.1, 2.2, 3.3, 4.4, 5.5],
            "col4": [1, 1, 1, 1, 1],
            "target": [0, 1, 0, 1, 0],
        }
    )

    wyn = bckt_cut_stats(
        variable=df["col3"],
        target=df["target"],
        # bins=[0, 3, 6],
        bins=2,
        total=True,
        plot=True,
    )
    print(wyn)

    column_types = ct.ColumnTypes(df, discrete_threshold=3)
    print(column_types.types)
    # Przykładowe dane
    df = pd.DataFrame({"value": [2, 5, 8, 15, 25]})

    buckets = pd.DataFrame(
        {"od": [0, 5, 10, 20], "do": [5, 10, 20, 30], "fit": [0.1, 3, 1, -0.1]}
    )

    # Wywołanie funkcji
    df2 = assign(df, "value", buckets, "fit")
    print(df2)

    # Przykładowe dane
    df = pd.DataFrame(
        {
            "czas": [
                "2024-01",
                "2024-01",
                "2024-01",
                "2024-02",
                "2024-02",
                "2024-02",
                "2024-03",
                "2024-03",
            ],
            "var": ["A", "B", "A", "A", "B", "C", "A", "C"],
            "weights": [1, 2, 1, 3, 1, 2, 2, 1],
        }
    )

    result = bckt_stats_over_time(
        df["czas"], df["var"], target=pd.Series([0] * len(df)), weights=df["weights"]
    )
    print(df)
    print(result)
