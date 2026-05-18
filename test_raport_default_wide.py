from buckets import report_html, buck, column_types as ct
import pandas as pd

dane = pd.read_parquet("data/default_credit_card.parquet")

types = ct.ColumnTypes(dane.drop(columns=["ID"]))

types.types.loc[
    types.types["column_name"].isin(
        ["credit_limit", "age"]
        + [f"BILL_AMT{i}" for i in range(1, 7)]
        + [f"PAY_AMT{i}" for i in range(1, 7)]
    ),
    "analytical_type",
] = "continuous"

raport_ob = buck.gen_report_objects(dane.drop(columns=["ID"]), types)
raport_h = report_html.generate_report(raport_ob)
report_html.save(raport_h, "result/raport_default_wide.html")

print("Raport zapisany do result/raport_default_wide.html")
