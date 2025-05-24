from buckets import report_html, buck, column_types as ct
import numpy as np
import pandas as pd

dane2 = pd.read_parquet('data/sample.parquet')
dane2['phone_kws'] = dane2['phone_kws'].astype('float64')

dane2 = dane2[['target',  'raty_liczba']]

types = ct.ColumnTypes(dane2)
temp = types.types['analytical_type']
temp['phone_kws'] = 'continuous' 
types.types['analytical_type'] = pd.Series(temp,
                                           index=types.types.index)

                    

raport_ob = buck.gen_report_objects(dane2, types)                                           
raport_h = report_html.generate_report(raport_ob)
report_html.save(raport_h, 'result/raport.html')
