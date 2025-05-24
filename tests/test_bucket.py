import unittest
import pandas as pd
import numpy as np
#from pandas.util.testing import assert_frame_equal # <-- for testing dataframes
import buckets as bckt
#target = __import__("buckets.py")

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
    test_df_3 = pd.DataFrame({'x':[1,1,1,8,2,2,3,np.nan],
                              'y':[1,1,0,0,1,0,1,1]
                              })


    def df_from_array(self, x, index):
        """
            Funkcja tworzy DataFrame'a z listy array-ów jako rekordy. Ustawia nazwy 
            zmiennych oraz ich typy. Nakłada przekazany indeks.
        """
        wyn = pd.DataFrame.from_records(x, index = index,
            columns = ['nr', 'bin', 'discrete', 'od', 'srodek', 'do', 'mean', 'median',
                'sum_target', 'n_obs', 'avg_target', 'pct_obs']
                )
        wyn=wyn.astype({'nr':            'int64',
                'bin':           'object',
                'discrete':      'object',
                'od':            'float64',
                'srodek':        'float64',
                'do':            'float64',
                'mean':          'float64',
                'median':        'float64',
                'sum_target':    'float64',
                'n_obs':         'float64',
                'avg_target':    'float64',
                'pct_obs':       'float64'})        
        return wyn

    def test_bckt_stat_simple_cat(self):
        """ Test zmiennej kategorycznej"""
        wyn_array = np.array(
            [(1, 'a', 'a', np.nan, np.nan, np.nan, np.nan, np.nan, 2., 3., 0.66666667, 0.42857143),
            (2, 'b', 'b', np.nan, np.nan, np.nan, np.nan, np.nan, 1., 2., 0.5       , 0.28571429),
            (3, 'c', 'c', np.nan, np.nan, np.nan, np.nan, np.nan, 1., 1., 1.        , 0.14285714),
            (4, 'd', 'd', np.nan, np.nan, np.nan, np.nan, np.nan, 0., 1., 0.        , 0.14285714),
            (5, 'TOTAL', 'TOTAL', np.nan, np.nan, np.nan, np.nan, np.nan, 4, 7, 0.57142857, 1.        )]
            )
        wyn_ref = self.df_from_array(wyn_array, index = ['a','b','c','d','TOTAL'])
        
        wyn = bckt.bckt_stats(self.test_df_1.x, self.test_df_1.y)
        pd.testing.assert_frame_equal(wyn, wyn_ref)


    def test_bckt_stat_simple_discr(self):
        """ Test zmiennej dyskretnej, numerycznej"""
        wyn_array = np.array(
            [
                (1, '1', 1, np.nan, np.nan, np.nan, np.nan, np.nan, 2, 3, 0.66666667, 0.42857143),
                (2, '2', 2, np.nan, np.nan, np.nan, np.nan, np.nan, 1, 2, 0.5       , 0.28571429),
                (3, '3', 3, np.nan, np.nan, np.nan, np.nan, np.nan, 1, 1, 1.        , 0.14285714),
                (4, '8', 8, np.nan, np.nan, np.nan, np.nan, np.nan, 0, 1, 0.        , 0.14285714),
                (5, 'TOTAL', np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 4, 7, 0.57142857, 1.        )]
            )
        wyn_ref = self.df_from_array(wyn_array, index = [ '1','2','3','8','TOTAL'])
        wyn_ref['discrete'] = wyn_ref.discrete.astype('float64')

        wyn = bckt.bckt_stats(self.test_df_2.x, self.test_df_2.y)

        pd.testing.assert_frame_equal(wyn, wyn_ref)


    def test_bckt_stat_sort_avg_target(self):
        """ Test sortowania po zmiennej avg_target"""
        wyn_array = np.array(
            [(1, '<NA>',   np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1, 1, 1.        , 0.125),
            (2, '8.0',        8.0, np.nan, np.nan, np.nan, np.nan, np.nan, 0, 1, 0.        , 0.125),
            (3, '2.0',        2.0, np.nan, np.nan, np.nan, np.nan, np.nan, 1, 2, 0.5       , 0.25 ),
            (4, '1.0',        1.0, np.nan, np.nan, np.nan, np.nan, np.nan, 2, 3, 0.66666667, 0.375),
            (5, '3.0',        3.0, np.nan, np.nan, np.nan, np.nan, np.nan, 1, 1, 1.        , 0.125),
            (6, 'TOTAL',   np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 5, 8, 0.625     , 1.   )]
            )
        wyn_ref = self.df_from_array(wyn_array, index = ['<NA>', '8.0','2.0','1.0','3.0','TOTAL'])
        wyn_ref['discrete'] = wyn_ref.discrete.astype('float64')
        #print(wyn_ref)

        wyn = bckt.bckt_stats(self.test_df_3.x, self.test_df_3.y, sort_by = 'avg_target')
        #print(wyn)

        pd.testing.assert_frame_equal(wyn, wyn_ref)

    def test_bckt_stat_sort_avg_target_desc(self):
        """ Test sortowania po zmiennej avg_target malejąco"""
        wyn_array = np.array(
            [                
                (1, '3', 3, np.nan, np.nan, np.nan, np.nan, np.nan, 1, 1, 1.        , 0.14285714),
                (2, '1', 1, np.nan, np.nan, np.nan, np.nan, np.nan, 2, 3, 0.66666667, 0.42857143),
                (3, '2', 2, np.nan, np.nan, np.nan, np.nan, np.nan, 1, 2, 0.5       , 0.28571429),
                (4, '8', 8, np.nan, np.nan, np.nan, np.nan, np.nan, 0, 1, 0.        , 0.14285714),                
                (5, 'TOTAL', np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 4, 7, 0.57142857, 1.)                
            ]
            )
        wyn_ref = self.df_from_array(wyn_array, index = ['3','1','2','8','TOTAL'])
        wyn_ref['discrete'] = wyn_ref.discrete.astype('float64')

        wyn = bckt.bckt_stats(self.test_df_2.x, self.test_df_2.y, sort_by = 'avg_target', ascending=False)
        pd.testing.assert_frame_equal(wyn, wyn_ref)

    def test_bckt_stat_filetered(self):
        """ Test, czy nie wywali błędu, gdy mam ramkę pandas z usuniętymi wierszami,
        co skutkuje indeksem który ma w sobie dziury
        """
        test = self.test_df_1
        test = test[test['x']!='b']
        print(test)
        try:
            bckt.bckt_stats(test.x, test.y)
        except TypeError:
            self.fail("TypeError został rzucony!")

    def test_bckt_cut_stat_simple(self):
        """ Test statystyk dla zmiennej ciągłej"""
        wyn_array = np.array(
                [(1,          '<NA>', np.nan, np.nan, np.nan,     np.nan,     np.nan,   np.nan,   1.,     1.,    1., 0.125),
                ( 2,  '(0.999, 2.0]', np.nan,     1.,    1.5,         2.,        1.4,        1,   3.,     5.,   0.6, 0.625),
                ( 3,    '(2.0, 8.0]', np.nan,     2.,      5,         8.,        5.5,      5.5,   1.,     2.,   0.5, 0.25 ),                
                ( 4,         'TOTAL', np.nan, np.nan, np.nan,     np.nan, 2.57142857,       2.,   5.,     8., 0.625, 1.   )] 
            )

        wyn_ref = self.df_from_array(wyn_array, index = ['<NA>','(0.999, 2.0]','(2.0, 8.0]', 'TOTAL'])
        wyn_ref['discrete'] = wyn_ref.discrete.astype('float64')
        wyn = bckt.bckt_cut_stats(self.test_df_3.x, self.test_df_3.y, bins=2)
        pd.testing.assert_frame_equal(wyn, wyn_ref)
		

    def test_bckt_cut_stat_sort_avg_target_desc(self):
        """ Test sortowania po zmiennej avg_target malejąco, dla zmiennej ciągłej"""
        wyn_array = np.array(
                [(1,          '<NA>', np.nan, np.nan, np.nan,     np.nan,     np.nan,   np.nan,   1,     1,    1., 0.125),
                ( 2,    '(2.0, 8.0]', np.nan,     1.,    1.5,         2.,        5.5,      5.5,   1,     2,   0.5, 0.25 ),
                ( 3,  '(0.999, 2.0]', np.nan,     2.,     5.,         8.,        1.4,        1,   3,     5,   0.6, 0.625),
                ( 4,         'TOTAL', np.nan, np.nan, np.nan,     np.nan, 2.57142857,       2.,   5,     8, 0.625, 1.   )] 
            )

        wyn_ref = self.df_from_array(wyn_array, index = ['<NA>','(2.0, 8.0]','(0.999, 2.0]', 'TOTAL'])
        wyn_ref['discrete'] = wyn_ref.discrete.astype('float64')

        wyn = bckt.bckt_cut_stats(self.test_df_3.x, self.test_df_3.y, bins=2, sort_by = 'avg_target')
        pd.testing.assert_frame_equal(wyn, wyn_ref)

    
    def test_bckt_cut_stat_duplicates(self):
        """ Test, czy poprawnie obsłużone jest wielokrotne wystąpienie tego samego kwantyla.
            Jak nie, to po prostu się wywali. Również test na puste kwantyle        
        """
        bckt.bckt_cut_stats(self.test_df_2.x, self.test_df_2.y, bins=10, sort_by = 'avg_target')

    def test_bckt_stat_wagi(self):
        """ Test wag, bez predykcji jeszcze"""
        wyn_array = np.array(
            [(1, 'a', 'a', np.nan, np.nan, np.nan, np.nan, np.nan, 3., 4., 0.75, 0.5),
            (2, 'b', 'b', np.nan, np.nan, np.nan, np.nan, np.nan, 1., 2., 0.5       , 0.25),
            (3, 'c', 'c', np.nan, np.nan, np.nan, np.nan, np.nan, 1., 1., 1.        , 0.125),
            (4, 'd', 'd', np.nan, np.nan, np.nan, np.nan, np.nan, 0., 1., 0.        , 0.125),
            (5, 'TOTAL', 'TOTAL', np.nan, np.nan, np.nan, np.nan, np.nan, 5, 8, 0.625, 1.        )]
            )
        wyn_ref = self.df_from_array(wyn_array, index = ['a','b','c','d','TOTAL'])
        
        wyn = bckt.bckt_stats(self.test_df_1.x, self.test_df_1.y, weights=self.test_df_1.w)
        pd.testing.assert_frame_equal(wyn, wyn_ref)

    def test_bckt_stat_min_info(self):
        """ min_info"""
        wyn_array = np.array(
            [(1, 'a', 'a', np.nan, np.nan, np.nan, np.nan, np.nan, 3., 4., 0.75, 0.5),
            (2, 'b', 'b', np.nan, np.nan, np.nan, np.nan, np.nan, 1., 2., 0.5       , 0.25),
            (3, 'c', 'c', np.nan, np.nan, np.nan, np.nan, np.nan, 1., 1., 1.        , 0.125),
            (4, 'd', 'd', np.nan, np.nan, np.nan, np.nan, np.nan, 0., 1., 0.        , 0.125),
            (5, 'TOTAL', 'TOTAL', np.nan, np.nan, np.nan, np.nan, np.nan, 5, 8, 0.625, 1.        )]
            )
        wyn_ref = self.df_from_array(wyn_array, index = wyn_array[:,1])
        
        wyn = bckt.bckt_stats(self.test_df_1.x, self.test_df_1.y, weights=self.test_df_1.w, min_info = True)
        pd.testing.assert_frame_equal(wyn, wyn_ref[['sum_target','n_obs','avg_target','pct_obs']])
		
    def test_bckt_stat_test(self):
        """ min_info"""
        
        wyn = bckt.bckt_stats(self.test_df_1.x, self.test_df_1.y, weights=self.test_df_1.w)
        odczyt = pd.DataFrame.from_dict(
            {
            'a': {'nr': 1, 'bin': 'a', 'discrete': 'a', 'od': np.nan, 'srodek': np.nan, 'do': np.nan, 'mean': np.nan, 'median': np.nan, 'sum_target': 3.0, 'n_obs': 4.0, 'avg_target': 0.75, 'pct_obs': 0.5}, 
            'b': {'nr': 2, 'bin': 'b', 'discrete': 'b', 'od': np.nan, 'srodek': np.nan, 'do': np.nan, 'mean': np.nan, 'median': np.nan, 'sum_target': 1.0, 'n_obs': 2.0, 'avg_target': 0.5, 'pct_obs': 0.25},
            'c': {'nr': 3, 'bin': 'c', 'discrete': 'c', 'od': np.nan, 'srodek': np.nan, 'do': np.nan, 'mean': np.nan, 'median': np.nan, 'sum_target': 1.0, 'n_obs': 1.0, 'avg_target': 1.0, 'pct_obs': 0.125},
            'd': {'nr': 4, 'bin': 'd', 'discrete': 'd', 'od': np.nan, 'srodek': np.nan, 'do': np.nan, 'mean': np.nan, 'median': np.nan, 'sum_target': 0.0, 'n_obs': 1.0, 'avg_target': 0.0, 'pct_obs': 0.125},
            'TOTAL': {'nr': 5, 'bin': 'TOTAL', 'discrete': 'TOTAL', 'od': np.nan, 'srodek': np.nan, 'do': np.nan, 'mean': np.nan, 'median': np.nan, 'sum_target': 5.0, 'n_obs': 8.0, 'avg_target': 0.625, 'pct_obs': 1.0}},
            orient='index'
        )
        print(wyn.to_dict(orient='index') )
        print(wyn.to_string())
        print(wyn.dtypes)
        pd.testing.assert_frame_equal(wyn,odczyt)
        print(wyn.to_markdown())

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

if __name__ == '__main__':
    unittest.main()  
     