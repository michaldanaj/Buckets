import pandas as pd
from enum import StrEnum


class Role(StrEnum):
    """Rola kolumny w analizie: wyjaśniająca, docelowa, główna kolumna czasowa lub pomijana.

    - `explanatory`: Kolumna, która jest używana jako zmienna objaśniająca (cecha).
    - `target`: Kolumna, która jest używana jako zmienna objaśniana (etykieta).
    - `main_time_col`: Kolumna, która jest używana jako główna kolumna czasowa do analizowania zmian w czasie.
    - `weights`: Kolumna z wagami obserwacji (krotność) — używana w statystykach,
      nie analizowana jako zmienna. Przypisywana WYŁĄCZNIE jawnie
      (setter `weights_col`), nigdy heurystycznie.
    - `skipped`: Kolumna, która jest pomijana w analizie (np. identyfikatory, daty).
    """
    EXPLANATORY = "explanatory"
    TARGET = "target"
    MAIN_TIME_COL = "main_time_col"
    WEIGHTS = "weights"
    SKIPPED = "skipped"


class AnalyticalType(StrEnum):
    """Typ analityczny kolumny: dyskretny, ciągły lub kategoryczny.
    
    - `categorical`: Zmienna **nie numeryczna**, która reprezentuje kategorie (np. płeć, kolor).
    - `discrete`: Zmienna **numeryczna** z niewielką liczbą unikalnych wartości (np. liczba dzieci).
    - `continuous`: Zmienna **numeryczna** z dużą liczbą unikalnych wartości (np. dochód).
    """
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"


def guess_column_type(var: pd.Series, discrete_threshold: int = 20) -> AnalyticalType:
    """
    Określa typ analityczny zmiennej na podstawie jej dtype i liczby unikalnych wartości.

    Logika klasyfikacji:
    - Zmienna nienumeryczna → `categorical`
    - Zmienna numeryczna z liczbą unikalnych wartości < `discrete_threshold` → `discrete`
    - Zmienna numeryczna z liczbą unikalnych wartości >= `discrete_threshold` → `continuous`

    NaN-y są ignorowane przez `nunique()`, więc nie wpływają na klasyfikację.

    Args:
        var: Kolumna Pandas, dla której określamy typ.
        discrete_threshold: Próg liczby unikalnych wartości rozdzielający `discrete` od
            `continuous`. Domyślnie 20.

    Returns:
        AnalyticalType: Wykryty typ analityczny zmiennej.
    """
    if pd.api.types.is_numeric_dtype(var):
        if var.nunique() < discrete_threshold:
            return AnalyticalType.DISCRETE
        else:
            return AnalyticalType.CONTINUOUS
    else:
        return AnalyticalType.CATEGORICAL


class ColumnTypes:
    """
    Przechowuje metadane kolumn ramki danych: typ analityczny i rolę każdej zmiennej.

    Typy analityczne (`analytical_type`) są wykrywane automatycznie przez `guess_column_type`.
    Role (`role`) są przypisywane heurystycznie (kolumna `target` → TARGET, kolumny
    z `id` w nazwie lub `date` → SKIPPED, pozostałe → EXPLANATORY) i można je
    nadpisać po utworzeniu obiektu (np. przez setter `time_col`).

    Attributes:
        discrete_threshold: Próg liczby unikalnych wartości używany przy klasyfikacji
            numerycznych zmiennych jako `discrete` vs `continuous`.
        types: DataFrame z kolumnami `column_name`, `dtype`, `analytical_type`, `role`,
            indeksowany nazwami kolumn wejściowego DataFrame.
    """

    def __init__(self, df: pd.DataFrame, discrete_threshold: int = 20):
        self.discrete_threshold = discrete_threshold
        self.types = self.determine_column_types(df)

    @property
    def target(self) -> str:
        """
        Zwraca nazwę kolumny docelowej.

        Returns:
            str: Nazwa kolumny docelowej.
        """
        return self.types.loc[self.types["role"] == Role.TARGET, "column_name"].values[
            0
        ]

    @target.setter
    def target(self, value: str = "target"):
        """
        Ustawia nazwę kolumny docelowej.

        Args:
            value: Nazwa kolumny docelowej.
        """
        self.types.loc[self.types["role"] == Role.TARGET, "column_name"] = value

    @property
    def time_col(self) -> str | None:
        """
        Zwraca nazwę głównej kolumny czasowej lub None, jeśli nie ustawiono.

        Returns:
            str | None: Nazwa kolumny z rolą MAIN_TIME_COL.
        """
        result = self.types.loc[self.types["role"] == Role.MAIN_TIME_COL, "column_name"]
        return result.values[0] if len(result) else None

    @time_col.setter
    def time_col(self, value: str):
        """
        Ustawia podaną kolumnę jako główną kolumnę czasową.

        Args:
            value: Nazwa kolumny, która ma otrzymać rolę MAIN_TIME_COL.
        """
        self.types.loc[self.types["role"] == Role.MAIN_TIME_COL, "role"] = Role.SKIPPED
        self.types.loc[self.types["column_name"] == value, "role"] = Role.MAIN_TIME_COL

    @property
    def weights_col(self) -> str | None:
        """
        Zwraca nazwę kolumny wag lub None, jeśli nie ustawiono.

        Returns:
            str | None: Nazwa kolumny z rolą WEIGHTS.
        """
        result = self.types.loc[self.types["role"] == Role.WEIGHTS, "column_name"]
        return result.values[0] if len(result) else None

    @weights_col.setter
    def weights_col(self, value: str):
        """
        Ustawia podaną kolumnę jako kolumnę wag obserwacji.

        Args:
            value: Nazwa kolumny, która ma otrzymać rolę WEIGHTS.
        """
        self.types.loc[self.types["role"] == Role.WEIGHTS, "role"] = Role.SKIPPED
        self.types.loc[self.types["column_name"] == value, "role"] = Role.WEIGHTS

    def set(self, colnames: list[str], analytical_type: str):
        """
        Ustawia typ analityczny dla podanych kolumn.

        Args:
            colnames: Lista nazw kolumn.
            analytical_type: Typ analityczny do ustawienia.
        """
        if isinstance(colnames, str):
            colnames = [colnames]

        for col in colnames:
            self.types.loc[self.types["column_name"] == col, "role"] = analytical_type

    def determine_column_types(self, df) -> pd.DataFrame:
        """
        Określa typ zmiennej (dtype) oraz typ analityczny dla każdej kolumny w ramce danych.

        Args:
            df: Ramka danych Pandas.
            discrete_threshold: Liczba unikalnych wartości, poniżej której zmienna numeryczna
                                jest uznawana za dyskretną.

        Returns:
            DataFrame z kolumnami: 'column_name', 'dtype', 'analytical_type', 'role'.
        """
        results = []

        for col in df.columns:
            dtype = df[col].dtype
            analytical_type = guess_column_type(df[col], self.discrete_threshold)

            role = Role.EXPLANATORY if col != "target" else Role.TARGET
            if col.startswith("id"):
                role = Role.SKIPPED
            elif "date" in col.lower():
                role = Role.SKIPPED

            results.append(
                {
                    "column_name": col,
                    "dtype": dtype,
                    "analytical_type": analytical_type,
                    "role": role,
                }
            )

        return pd.DataFrame(results, index=df.columns)
