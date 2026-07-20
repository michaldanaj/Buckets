# coding: utf-8
"""
Sparkowy odpowiednik test_raport_default_amend.py: raport dla tabeli
"amend" (long + kolumny testowe disc_num / litera / rozmiar).

Uwaga: kolumna `rozmiar` jest w parquecie pandasową kategorią uporządkowaną
('mały' < 'średni' < 'duży') — Spark czyta ją jako zwykły string, więc
porządek kategorii w raporcie będzie alfabetyczny (to samo ograniczenie
opisuje pkt 7 backlogu dla ścieżki pandas).

Wymaga ekstrasa spark i JVM. Uruchomienie:
    uv run --extra spark python raport_spark_default_amend.py
"""

import os
os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-21-temurin-jdk/'

import pyspark.sql.functions as F
from pyspark.sql import SparkSession

import buckets.spark as sp
from buckets import report_html
from buckets.report import DatasetReport



spark = (
    SparkSession.builder.master("local[*]")
    .appName("buckets-raport-amend")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)

dane_long = spark.read.parquet("data/default_credit_card_long_amend.parquet")

# wprowadzam braki danych (jak w raporcie bazowym)
dane_long = dane_long.withColumn("_r", F.rand(seed=42))
for kol in ("pay_status", "credit_limit"):
    dane_long = dane_long.withColumn(kol, F.when(F.col("_r") >= 0.1, F.col(kol)))
dane_long = dane_long.drop("_r")

dane_long = dane_long.withColumn("miesiac", F.date_format("date", "yyyy-MM"))

dane_long = dane_long.drop("ID").persist()

types = sp.column_types_from_spark(dane_long)
types.time_col = "miesiac"

print("Wykryte typy kolumn:")
print(types.types[["analytical_type", "role"]])

source = sp.SparkSource(dane_long, types)
raport_h = DatasetReport(source, source.types).to_html()

os.makedirs("result", exist_ok=True)
report_html.save(raport_h, "result/raport_spark_default_amend.html")

print("Raport zapisany do result/raport_spark_default_amend.html")

spark.stop()
