import unittest
import pandas as pd
import numpy as np
#from pandas.util.testing import assert_frame_equal # <-- for testing dataframes
import buckets.buck as bckt
#target = __import__("buckets.py")
pd.set_option('display.max_columns', None)  # Pokazuje wszystkie kolumny
pd.set_option('display.max_rows', None)     # Pokazuje wszystkie wiersze

class BucketTests(unittest.TestCase):

    """ class for running unittests """

    #zmienna x kategoryczna
    test_df_1 = pd.DataFrame({'x':['a','a','a','d','b','b','c'],
                              'y':[1,1,0,0,1,0,1],
							  'w':[2.0,1,1,1,1,1,1],
                              })

    #zmienna x numeryczna, dyskretna
    test_df_2 = pd.DataFrame({'x':[1,1,1,8,2,2,3],
                              'y':[1,1,0,0,1,0,1]
                              })

    #zmienna x numeryczna, dyskretna, z nan
    # convert_dtypes() -> x jako Int64 (nullable), poprawna reprezentacja
    # zmiennej numerycznej dyskretnej z brakiem (zamiast object).
    test_df_3 = pd.DataFrame({'x':[1,1,1,8,2,2,3,pd.NA],
                              'y':[1,1,0,0,1,0,1,1]
                              }).convert_dtypes()


    def df_from_array(self, x, index, discr_type='string'):
        """
            Buduje referencyjny DataFrame z listy rekordów, nakładając jawnie typy
            zgodne z kontraktem (spec/typy-danych.md) — bez polegania na inferencji
            `convert_dtypes`. Kolumny Float64 są tworzone przez `to_numeric`, by
            poprawnie obsłużyć kolumny złożone z samych pd.NA.

            discr_type: docelowy typ kolumny `discrete` (zależny od typu wejścia
            danego testu): 'string' dla kategorycznych, 'Int64'/'Float64' dla
            numerycznych.
        """
        wyn = pd.DataFrame.from_records(x, index=index,
            columns=['nr', 'bin', 'discrete', 'od', 'srodek', 'do', 'mean', 'median',
                'sum_target', 'n_obs', 'avg_target', 'pct_obs'])

        float_cols = ['od', 'srodek', 'do', 'mean', 'median', 'avg_target', 'pct_obs']
        for col in float_cols:
            wyn[col] = pd.to_numeric(wyn[col], errors='coerce').astype('Float64')
        wyn = wyn.astype({
            'nr':         'Int64',
            'bin':        'string',
            'discrete':   discr_type,
            'sum_target': 'Int64',
            'n_obs':      'Int64',
        })
        return wyn

    def test_bckt_stat_simple_cat(self):
        """ Test zmiennej kategorycznej"""
        wyn_array = np.array(
            [(1, 'a', 'a', pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, 2, 3, 0.66666667, 0.42857143),
            (2, 'b', 'b', pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, 1, 2, 0.5       , 0.28571429),
            (3, 'c', 'c', pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, 1, 1, 1.        , 0.14285714),
            (4, 'd', 'd', pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, 0, 1, 0.        , 0.14285714),
            (5, 'TOTAL', 'TOTAL', pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, 4, 7, 0.57142857, 1.        )]
            )
        wyn_ref = self.df_from_array(wyn_array, index = ['a','b','c','d','TOTAL'])
        
        
        wyn = bckt.bckt_stats(self.test_df_1.x, self.test_df_1.y)
        pd.testing.assert_frame_equal(wyn.convert_dtypes(), wyn_ref)


    def test_bckt_stat_simple_discr(self):
        """ Test zmiennej dyskretnej, numerycznej"""
        wyn_array = np.array(
            [
                (1, '1', 1, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, 2, 3, 0.66666667, 0.42857143),
                (2, '2', 2, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, 1, 2, 0.5       , 0.28571429),
                (3, '3', 3, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, 1, 1, 1.        , 0.14285714),
                (4, '8', 8, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, 0, 1, 0.        , 0.14285714),
                (5, 'TOTAL', pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, 4, 7, 0.57142857, 1.        )]
            )
        wyn_ref = self.df_from_array(wyn_array, index = [ '1','2','3','8','TOTAL'], 
                                     discr_type='Int64')

        wyn = bckt.bckt_stats(self.test_df_2.x, self.test_df_2.y)

        pd.testing.assert_frame_equal(wyn.convert_dtypes(), wyn_ref)


    def test_bckt_stat_sort_avg_target(self):
        """ Test sortowania po zmiennej avg_target"""
        # discrete zachowuje naturalny typ wejścia (Int64), bin = "8" nie "8.0"
        wyn_array = np.array(
            [(1, '<NA>',   pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, 1, 1, 1.        , 0.125),
            (2, '8',          8, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, 0, 1, 0.        , 0.125),
            (3, '2',          2, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, 1, 2, 0.5       , 0.25 ),
            (4, '1',          1, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, 2, 3, 0.66666667, 0.375),
            (5, '3',          3, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, 1, 1, 1.        , 0.125),
            (6, 'TOTAL',   pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, 5, 8, 0.625     , 1.   )]
            )
        wyn_ref = self.df_from_array(wyn_array, index = ['<NA>', '8','2','1','3','TOTAL'],
                                     discr_type='Int64')

        wyn = bckt.bckt_stats(self.test_df_3.x, self.test_df_3.y, sort_by = 'avg_target')

        pd.testing.assert_frame_equal(wyn, wyn_ref)

    def test_bckt_stat_sort_avg_target_desc(self):
        """ Test sortowania po zmiennej avg_target malejąco"""
        wyn_array = np.array(
            [                
                (1, '3', 3, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, 1, 1, 1.        , 0.14285714),
                (2, '1', 1, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, 2, 3, 0.66666667, 0.42857143),
                (3, '2', 2, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, 1, 2, 0.5       , 0.28571429),
                (4, '8', 8, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, 0, 1, 0.        , 0.14285714),                
                (5, 'TOTAL', pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, 4, 7, 0.57142857, 1.)                
            ]
            )
        wyn_ref = self.df_from_array(wyn_array, index = ['3','1','2','8','TOTAL'],
                                     discr_type='Int64')

        wyn = bckt.bckt_stats(self.test_df_2.x, self.test_df_2.y, sort_by = 'avg_target', ascending=False)
        pd.testing.assert_frame_equal(wyn, wyn_ref)

    def test_bckt_stat_filetered(self):
        """ Test, czy nie wywali błędu, gdy mam ramkę pandas z usuniętymi wierszami,
        co skutkuje indeksem który ma w sobie dziury
        """
        test = self.test_df_1
        test = test[test['x']!='b']
        try:
            bckt.bckt_stats(test.x, test.y)
        except TypeError:
            self.fail("TypeError został rzucony!")

    def test_bckt_cut_stat_simple(self):
        """ Test statystyk dla zmiennej ciągłej"""
        wyn_array = np.array(
                [(1,          '<NA>', pd.NA, pd.NA, pd.NA,     pd.NA,     pd.NA,   pd.NA,   1,     1,    1., 0.125),
                ( 2,  '(0.999, 2.0]', pd.NA,     1.,    1.5,         2.,        1.4,        1,   3,     5,   0.6, 0.625),
                ( 3,    '(2.0, 8.0]', pd.NA,     2.,      5,         8.,        5.5,      5.5,   1,     2,   0.5, 0.25 ),                
                ( 4,         'TOTAL', pd.NA, pd.NA, pd.NA,     pd.NA, 2.57142857,       2.,   5,     8, 0.625, 1.   )] 
            )

        wyn_ref = self.df_from_array(wyn_array, index = ['<NA>','(0.999, 2.0]','(2.0, 8.0]', 'TOTAL'], discr_type='Int64')
        wyn = bckt.bckt_cut_stats(self.test_df_3.x, self.test_df_3.y, bins=2)
        print('ref:')
        print(wyn_ref)
        print('wyn:')
        print(wyn)
        print(wyn.dtypes)
        pd.testing.assert_frame_equal(wyn_ref.convert_dtypes(), wyn.convert_dtypes(), check_dtype=False)
		

    def test_bckt_cut_stat_sort_avg_target_desc(self):
        """ Test sortowania po zmiennej avg_target malejąco, dla zmiennej ciągłej"""
        wyn_array = np.array(
                [(1,          '<NA>', pd.NA, pd.NA, pd.NA,     pd.NA,     pd.NA,   pd.NA,   1,     1,    1., 0.125),
                ( 2,    '(2.0, 8.0]', pd.NA,     1.,    1.5,         2.,        5.5,      5.5,   1,     2,   0.5, 0.25 ),
                ( 3,  '(0.999, 2.0]', pd.NA,     2.,     5.,         8.,        1.4,        1,   3,     5,   0.6, 0.625),
                ( 4,         'TOTAL', pd.NA, pd.NA, pd.NA,     pd.NA, 2.57142857,       2.,   5,     8, 0.625, 1.   )] 
            )

        wyn = bckt.bckt_cut_stats(self.test_df_3.x, self.test_df_3.y, bins=2, sort_by = 'avg_target')
        print(wyn)
        print(wyn.dtypes)

        print("XXXXXXXXXXXXXXXXXXXXX")
        wyn_ref = self.df_from_array(wyn_array, discr_type='float64',
                                     index = ['<NA>','(2.0, 8.0]','(0.999, 2.0]', 'TOTAL'])
        print(wyn_ref)
        print(wyn_ref.dtypes)
        print(wyn)
        print(wyn.dtypes)

        pd.testing.assert_frame_equal(wyn.convert_dtypes(), wyn_ref.convert_dtypes())

    
    def test_bckt_cut_stat_duplicates(self):
        """ Test, czy poprawnie obsłużone jest wielokrotne wystąpienie tego samego kwantyla.
            Jak nie, to po prostu się wywali. Również test na puste kwantyle        
        """
        bckt.bckt_cut_stats(self.test_df_2.x, self.test_df_2.y, bins=10, sort_by = 'avg_target')

    def test_bckt_stat_wagi(self):
        """ Test wag, bez predykcji jeszcze"""
        wyn_array = np.array(
            [(1, 'a', 'a', pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, 3., 4., 0.75, 0.5),
            (2, 'b', 'b', pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, 1., 2., 0.5       , 0.25),
            (3, 'c', 'c', pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, 1., 1., 1.        , 0.125),
            (4, 'd', 'd', pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, 0., 1., 0.        , 0.125),
            (5, 'TOTAL', 'TOTAL', pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, 5, 8, 0.625, 1.        )]
            )
        wyn_ref = self.df_from_array(wyn_array, index = ['a','b','c','d','TOTAL'])
        
        wyn = bckt.bckt_stats(self.test_df_1.x, self.test_df_1.y, weights=self.test_df_1.w)
        pd.testing.assert_frame_equal(wyn, wyn_ref)

    def test_bckt_stat_min_info(self):
        """ min_info"""
        wyn_array = np.array(
            [(1, 'a', 'a', pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, 3, 4, 0.75, 0.5),
            (2, 'b', 'b', pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, 1, 2, 0.5       , 0.25),
            (3, 'c', 'c', pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, 1, 1, 1        , 0.125),
            (4, 'd', 'd', pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, 0, 1, 0        , 0.125),
            (5, 'TOTAL', 'TOTAL', pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, 5, 8, 0.625, 1.        )]
            )
        wyn_ref = self.df_from_array(wyn_array, index = wyn_array[:,1])
        
        wyn = bckt.bckt_stats(self.test_df_1.x, self.test_df_1.y, weights=self.test_df_1.w, min_info = True)
        pd.testing.assert_frame_equal(wyn, wyn_ref[['sum_target','n_obs','avg_target','pct_obs']])
		
    def test_bckt_stat_test(self):
        """ min_info"""
        
        wyn = bckt.bckt_stats(self.test_df_1.x, self.test_df_1.y, weights=self.test_df_1.w)
        odczyt = pd.DataFrame.from_dict(
            {
            'a': {'nr': 1, 'bin': 'a', 'discrete': 'a', 'od': pd.NA, 'srodek': pd.NA, 'do': pd.NA, 'mean': pd.NA, 'median': pd.NA, 'sum_target': 3.0, 'n_obs': 4.0, 'avg_target': 0.75, 'pct_obs': 0.5}, 
            'b': {'nr': 2, 'bin': 'b', 'discrete': 'b', 'od': pd.NA, 'srodek': pd.NA, 'do': pd.NA, 'mean': pd.NA, 'median': pd.NA, 'sum_target': 1.0, 'n_obs': 2.0, 'avg_target': 0.5, 'pct_obs': 0.25},
            'c': {'nr': 3, 'bin': 'c', 'discrete': 'c', 'od': pd.NA, 'srodek': pd.NA, 'do': pd.NA, 'mean': pd.NA, 'median': pd.NA, 'sum_target': 1.0, 'n_obs': 1.0, 'avg_target': 1.0, 'pct_obs': 0.125},
            'd': {'nr': 4, 'bin': 'd', 'discrete': 'd', 'od': pd.NA, 'srodek': pd.NA, 'do': pd.NA, 'mean': pd.NA, 'median': pd.NA, 'sum_target': 0.0, 'n_obs': 1.0, 'avg_target': 0.0, 'pct_obs': 0.125},
            'TOTAL': {'nr': 5, 'bin': 'TOTAL', 'discrete': 'TOTAL', 'od': pd.NA, 'srodek': pd.NA, 'do': pd.NA, 'mean': pd.NA, 'median': pd.NA, 'sum_target': 5.0, 'n_obs': 8.0, 'avg_target': 0.625, 'pct_obs': 1.0}},
            orient='index'
        )
        # Triage (spec sekcja 9): from_dict daje surowe numpy dtypes — dociągamy
        # referencję do kontraktu typów jawnie, zamiast polegać na inferencji.
        float_cols = ['od', 'srodek', 'do', 'mean', 'median', 'avg_target', 'pct_obs']
        for col in float_cols:
            odczyt[col] = pd.to_numeric(odczyt[col], errors='coerce').astype('Float64')
        odczyt = odczyt.astype({
            'nr': 'Int64', 'bin': 'string', 'discrete': 'string',
            'sum_target': 'Int64', 'n_obs': 'Int64',
        })
        pd.testing.assert_frame_equal(wyn, odczyt)

    def test_bckt_cut_filtered(self):
        """ Test, czy nie wywali błędu, gdy mam ramkę pandas z usuniętymi wierszami,
        co skutkuje indeksem który ma w sobie dziury
        """
        test = self.test_df_3
        test = test[test['x'] != 2]
        try:
            bckt.bckt_cut_stats(test.x, test.y, bins=2)
        except ValueError:
            self.fail("Niespodziewanie rzucony wyjątek ValueError!")

    def test_bckt_stats_over_time_basic(self):
        # Przygotowanie przykładowych danych
        df = pd.DataFrame({
            "czas": ["2024-01", "2024-01", "2024-01", "2024-02", "2024-02", "2024-02", "2024-03", "2024-03"],
            "var": ["A", "B", "A", "A", "B", "C", "A", "C"],
            "weights": [1, 2, 1, 3, 1, 2, 2, 1]
        })

        # Oczekiwany wynik
        expected = pd.DataFrame(
            {
                "A": [1/2, 1/2, 2/3],
                "B": [1/2, 1/6, 0.0],
                "C": [0.0, 1/3, 1/3]
            },
            index=["2024-01", "2024-02", "2024-03"]
        )
        expected.index.name = "czas"
        expected.columns.name = "var"

        # Wywołanie funkcji
        result = bckt.bckt_stats_over_time(
            czas=df["czas"],
            var=df["var"],
            target=pd.Series([0]*len(df)),  # target nie jest używany w tej funkcji
            weights=df["weights"]
        )

        # Porównanie wyników
        pd.testing.assert_frame_equal(result, expected, check_dtype=False, atol=1e-8)    

    def test_bckt_stats_over_time_basic_bez_wag(self):
        # Przygotowanie przykładowych danych
        df = pd.DataFrame({
            "czas": ["2024-01", "2024-01", "2024-01", "2024-02", "2024-02", "2024-02", "2024-03", "2024-03"],
            "var": ["A", "B", "A", "A", "B", "C", "A", "C"],
        })

        # Oczekiwany wynik
        expected = pd.DataFrame(
            {
                "A": [2/3, 1/3, 1/3],
                "B": [1/3, 1/3, 0.0],
                "C": [0.0, 1/3, 1/3]
            },
            index=["2024-01", "2024-02", "2024-03"]
        )
        expected.index.name = "czas"
        expected.columns.name = "var"

        # Wywołanie funkcji
        result = bckt.bckt_stats_over_time(
            czas=df["czas"],
            var=df["var"],
            target=pd.Series([0]*len(df)),  # target nie jest używany w tej funkcji
            weights=df["weights"]
        )

        # Porównanie wyników
        pd.testing.assert_frame_equal(result, expected, check_dtype=False, atol=1e-8)    

    def test_bckt_stats_over_time_brak_var_w_okresie(self):
        """
        Gdy zmienna var nie przyjmuje danej wartości w jakimś okresie,
        pivot_target nie powinien zawierać np.nan (0/0) dla tej kombinacji.
        """
        df = pd.DataFrame({
            "czas":   ["2024-01", "2024-01", "2024-02", "2024-02"],
            "var":    ["A",       "B",        "A",        pd.NA],
            "target": [0,          1,          0,          0],
        })

        result = bckt.bckt_stats_over_time(
            czas=df["czas"],
            var=df["var"],
            target=df["target"],
        )

        pivot_target = result[2]
        # Float64 dtype: isna() nie wykrywa np.nan, tylko pd.NA — dlatego isin
        self.assertFalse(
            pivot_target.isin([np.nan]).any().any(),
            "pivot_target zawiera np.nan (wynik 0/0) dla kombinacji okres–var bez obserwacji",
        )

    if __name__ == '__main__':
        unittest.main()
