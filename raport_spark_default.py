# coding: utf-8
"""
Sparkowy odpowiednik test_raport_default.py: raport dla danych long
liczony z ramki Spark (agregacja w Sparku, analiza w pandas —
spec/raport-spark.md).

Wymaga ekstrasa spark i JVM. Uruchomienie:
    uv run --extra spark python raport_spark_default.py
"""

import os

import pyspark.sql.functions as F
from pyspark.sql import SparkSession

import buckets.spark as sp
import buckets.statitics as st
from buckets import buck, report_html
from buckets.report import DatasetReport
from buckets.spark import to_pseudo_obs

spark = (
    SparkSession.builder.master("local[*]")
    .appName("buckets-raport-default")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)

dane_long = spark.read.parquet("data/default_credit_card_long.parquet")

# wprowadzam braki danych (analogicznie do wersji pandas: ~10% wierszy,
# te same wiersze dla obu kolumn — wspólna kolumna losująca)
dane_long = dane_long.withColumn("_r", F.rand(seed=42))
for kol in ("pay_status", "credit_limit"):
    dane_long = dane_long.withColumn(kol, F.when(F.col("_r") >= 0.1, F.col(kol)))
dane_long = dane_long.drop("_r")

# odpowiednik dt.to_period("M")
dane_long = dane_long.withColumn("miesiac", F.date_format("date", "yyyy-MM"))

dane_long = dane_long.drop("ID").persist()  # jeden groupBy na zmienną

types = sp.column_types_from_spark(dane_long)
types.time_col = "miesiac"

source = sp.SparkSource(dane_long, types)
raport_h = DatasetReport(source, source.types).to_html()

os.makedirs("result", exist_ok=True)
report_html.save(raport_h, "result/raport_spark_default.html")

print("Raport zapisany do result/raport_spark_default.html")


# ----------------------
# analogi dalszych demonstracji z wersji pandas — wszystkie na
# pseudo-obserwacjach z kanonicznego agregatu (dane nie schodzą
# z klastra wierszowo)

# rozkład pay_status w czasie (odpowiednik bckt_stats_over_time)
agg = sp.aggregate_variable(dane_long, "pay_status", "target", time_col="miesiac")
pobs = to_pseudo_obs(agg)
wyn = buck.bckt_stats_over_time(
    pobs["miesiac"], pobs["pay_status"], pobs["target"],
    weights=pobs["weights"].astype(float),
)
print(wyn.distribution())

# gini pay_status w czasie
gini_over_time = st.gini(
    pobs["pay_status"], pobs["target"], by=pobs["miesiac"], weights=pobs["weights"]
)
print(gini_over_time)

# dyskretyzacja credit_limit drzewem
agg_cl = sp.aggregate_variable(dane_long, "credit_limit", "target")
pobs_cl = to_pseudo_obs(agg_cl)
wyn_tree = buck.bckt_tree_stats(pobs_cl, "credit_limit", "target", weights="weights")
print(wyn_tree)

spark.stop()
