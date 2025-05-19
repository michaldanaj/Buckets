import buckets.buck as bkt
import buckets.column_types as ct
import pandas as pd
import buckets.report_html as vr


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
kolumny.set("default payment next month", "target")

# Generowanie raportu
wyn = bkt.gen_report_objects(dane, kolumny, max_levels=10)
html = vr.generate_report(wyn)

# Zapis raportu
# os.makedirs("result", exist_ok=True)
vr.save(html, "./result/report.html")

print("Raport został zapisany jako 'result/report.html'.")
