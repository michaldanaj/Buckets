import os

import pandas as pd

from buckets import buck, column_types as ct, report_html

# Tabela "amend": long + dodatkowe kolumny testowe
#   disc_num — zmienna liczbowa dyskretna (1, 3, 8, 10, 11, 18)
#   litera   — zmienna stringowa ('a', 'b', 'c')
#   rozmiar  — uporządkowana Category ('mały' < 'średni' < 'duży')
dane_long = pd.read_parquet("data/default_credit_card_long_amend.parquet")

# wprowadzam braki danych (jak w raporcie bazowym)
idx = dane_long.sample(frac=0.1, random_state=42).index
dane_long.loc[idx, "pay_status"] = pd.NA
dane_long.loc[idx, "credit_limit"] = pd.NA

dane_long["miesiac"] = dane_long["date"].dt.to_period("M")

types = ct.ColumnTypes(dane_long.drop(columns=["ID"]))
types.time_col = "miesiac"

print("Wykryte typy kolumn:")
print(types.types[["analytical_type", "role"]])

raport_ob = buck.gen_report_objects(dane_long.drop(columns=["ID"]), types)
raport_h = report_html.generate_report(raport_ob)

os.makedirs("result", exist_ok=True)
report_html.save(raport_h, "result/raport_default_amend.html")

print("Raport zapisany do result/raport_default_amend.html")
