import pandas as pd

class ColumnTypes:
    """
    Klasa do określania typów zmiennych w ramce danych Pandas.
    """

    def __init__(self, df: pd.DataFrame, discrete_threshold: int = 20):
        self.discrete_threshold = discrete_threshold
        self.types = self.determine_column_types(df)

    @property
    def target(self) -> str:
        """
        Funkcja zwracająca nazwę kolumny docelowej.

        Args:
            value: Nazwa kolumny docelowej.

        Returns:
            str: Nazwa kolumny docelowej.
        """
        return self.types.loc[self.types['role'] == 'target', 'column_name'].values[0]

    @target.setter
    def target(self, value: str = 'target'):
        """
        Funkcja do ustawiania nazwy kolumny docelowej.

        Args:
            value: Nazwa kolumny docelowej.
        """
        self.types.loc[self.types['role'] == 'target', 'column_name'] = value 

    
    def set(self, colnames: list[str], analytical_type: str):
        """
        Funkcja do ustawiania typu analitycznego dla określonych kolumn.

        Args:
            colnames: Lista nazw kolumn.
            analytical_type: Typ analityczny do ustawienia.
        """
        if type(colnames) == str:
            colnames = [colnames]

        for col in colnames:
            self.types.loc[self.types['column_name'] == col, 'role'] = analytical_type

    def determine_column_types(self, df) -> pd.DataFrame:
        """
        Funkcja określająca typ zmiennej (dtype) oraz typ analityczny dla każdej kolumny w ramce danych.

        Args:
            df: Ramka danych Pandas.
            discrete_threshold: Liczba unikalnych wartości, poniżej której zmienna numeryczna
                                jest uznawana za dyskretną.

        Returns:
            DataFrame z kolumnami: 'column_name', 'dtype', 'analytical_type'.
        """
        results = []

        for col in df.columns:
            # Określenie typu zmiennej (dtype)
            dtype = df[col].dtype

            # Określenie typu analitycznego
            if pd.api.types.is_numeric_dtype(dtype):
                unique_values = df[col].nunique()
                if unique_values < self.discrete_threshold:
                    analytical_type = 'discrete'
                else:
                    analytical_type = 'continuous'
            else:
                analytical_type = 'categorical'

            # określenie roli
            role = 'explanatory' if col != 'target' else 'target'
            # jeśli zaczyan się na id, to uznajemy że jest to zmienna identyfikująca i jej nie analizujemy
            if col.startswith('id'):
                role = 'skipped'
            # jeśli zawiera słowo 'date', to uznajemy że jest to zmienna czasowa i jej nie analizujemy
            elif 'date' in col.lower():
                role = 'skipped'

            # Dodanie wyników do listy
            results.append({
                'column_name': col,
                'dtype': dtype,
                'analytical_type': analytical_type,
                'role': role,
            })

        # Konwersja wyników do DataFrame
        return pd.DataFrame(results)