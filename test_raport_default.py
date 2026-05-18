from buckets import report_html, buck, column_types as ct, statitics as st
import pandas as pd

dane_long = pd.read_parquet("data/default_credit_card_long.parquet")

types = ct.ColumnTypes(dane_long.drop(columns=["ID"]))

types.types.loc[
    types.types["column_name"].isin(
        ["credit_limit", "age", "bill_amount", "pay_amount"]
    ),
    "analytical_type",
] = "continuous"

raport_ob = buck.gen_report_objects(dane_long.drop(columns=["ID"]), types)
raport_h = report_html.generate_report(raport_ob)
report_html.save(raport_h, "result/raport_default.html")

print("Raport zapisany do result/raport_default.html")


# ----------------------

# wyn = buck.bckt_stats_over_time(dane2["czas"], dane2["raty_liczba"], dane2["target"])
wyn = buck.bckt_stats_over_time(
    dane_long["date"], dane_long["pay_status"], dane_long["target"]
)

gini_over_time = st.gini(dane_long["pay_status"], dane_long["target"], by=dane_long["date"])
print(gini_over_time)
