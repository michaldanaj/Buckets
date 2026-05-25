from buckets import report_html, buck, column_types as ct, statitics as st
import pandas as pd

dane_long = pd.read_parquet("data/default_credit_card_long.parquet")

# wprowadzam braki danych
idx = dane_long.sample(frac=0.1, random_state=42).index
dane_long.loc[idx,'pay_status'] = pd.NA
dane_long.loc[idx,'credit_limit'] = pd.NA

dane_long["miesiac"] = dane_long["date"].dt.to_period("M")

types = ct.ColumnTypes(dane_long.drop(columns=["ID"]))

types.time_col = "miesiac"

raport_ob = buck.gen_report_objects(dane_long.drop(columns=["ID"]), types)
raport_h = report_html.generate_report(raport_ob)
report_html.save(raport_h, "result/raport_default.html")

print("Raport zapisany do result/raport_default.html")


# ----------------------

# wyn = buck.bckt_stats_over_time(dane2["czas"], dane2["raty_liczba"], dane2["target"])
wyn = buck.bckt_stats_over_time(
    dane_long["date"], dane_long["pay_status"], dane_long["target"]
)

miesiac = dane_long["date"].dt.to_period("M")

gini_over_time = st.gini(dane_long["pay_status"], dane_long["target"], by=miesiac)
print(gini_over_time)

wyn_tree = buck.bckt_tree(dane_long, "credit_limit", "target")
print(wyn_tree)
