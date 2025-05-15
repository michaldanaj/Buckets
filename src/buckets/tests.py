import buckets.buckets as bkt
import pandas as pd
import buckets.var_report as vr
import os 

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
kolumny = bkt.ColumnTypes(dane)
kolumny.set('default payment next month', 'target')
statsy = bkt.gen_buckets(dane, kolumny)
bkt.plot(statsy['AGE'])
bkt.plot(statsy['MARRIAGE'])


bkt.plot(statsy['PAY_AMT1'])
statsy['PAY_AMT1']

# Generowanie raportu
wyn = bkt.gen_report_objects(dane, kolumny)
html = vr.generate_report(wyn)

# Zapis raportu
#os.makedirs("result", exist_ok=True)
vr.save(html, "./result/report.html")

print("Raport został zapisany jako 'result/report.html'.")