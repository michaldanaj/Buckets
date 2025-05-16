import os 
print(os.getcwd())
import sys
print(sys.path)

import buckets.buck as bkt
import buckets.column_types as ct
import pandas as pd
import buckets.var_report as vr

# oczytuję csv z katalogu data
def test_read_csv() -> pd.DataFrame:
    # given
    path = "data/default of credit card clients.csv"
    # when
    df = pd.read_csv(path)
    # then
    assert df is not None
    assert len(df) == 30000
    return df

# test_read_csv()
dane = test_read_csv()
kolumny = ct.ColumnTypes(dane)
kolumny.set('default payment next month', 'target')
statsy = bkt.gen_buckets(dane, kolumny)
bkt.plot(statsy['AGE'])
bkt.plot(statsy['MARRIAGE'])

######    do testów to wrzucić, wartość z granicy   ########
z_drzewka = bkt.bckt_tree(dane, 'AGE', 'default payment next month', min_samples_split=100)
dane['xxx'] = bkt.assign(dane, 'AGE', z_drzewka, 'avg_target')    
z = bkt.assign(pd.DataFrame({'AGE':[79]}), 'AGE', z_drzewka, 'avg_target')    
print("z: ", z)
######    do testów to wrzucić   ########

bkt.plot(statsy['PAY_AMT1'])
statsy['PAY_AMT1']

# Generowanie raportu
wyn = bkt.gen_report_objects(dane, kolumny)
html = vr.generate_report(wyn)

# Zapis raportu
#os.makedirs("result", exist_ok=True)
vr.save(html, "./result/report.html")

print("Raport został zapisany jako 'result/report.html'.")