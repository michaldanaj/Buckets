# coding: utf-8
"""
Sparkowy odpowiednik test_raport_default_wide.py: raport dla danych wide
z ręcznym nadpisaniem typów analitycznych.

Wymaga ekstrasa spark i JVM. Uruchomienie:
    uv run --extra spark python raport_spark_default_wide.py
"""

import os

from pyspark.sql import SparkSession

import buckets.spark as sp
from buckets import report_html
from buckets.report import DatasetReport

spark = (
    SparkSession.builder.master("local[*]")
    .appName("buckets-raport-wide")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)

dane = spark.read.parquet("data/default_credit_card.parquet").drop("ID").persist()

types = sp.column_types_from_spark(dane)

# nadpisanie typów — types.types to zwykła ramka pandas, identycznie jak
# w ścieżce pandasowej
types.types.loc[
    types.types["column_name"].isin(
        ["credit_limit", "age"]
        + [f"BILL_AMT{i}" for i in range(1, 7)]
        + [f"PAY_AMT{i}" for i in range(1, 7)]
    ),
    "analytical_type",
] = "continuous"

source = sp.SparkSource(dane, types)
raport_h = DatasetReport(source, source.types).to_html()

os.makedirs("result", exist_ok=True)
report_html.save(raport_h, "result/raport_spark_default_wide.html")

print("Raport zapisany do result/raport_spark_default_wide.html")

spark.stop()
