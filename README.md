# buckets

Statystyki targetu na przedziałach i wartościach zmiennych objaśniających —
narzędzie do eksploracji zmiennych w modelowaniu (target binarny 0/1,
opcjonalnie predykcja i wagi). Generuje tabele bucketów, dyskretyzację
drzewem, współczynniki GINI (także w czasie) oraz zbiorczy raport HTML.

Dane wejściowe: ramki **pandas** albo — po agregacji po stronie klastra —
ramki **Spark** (patrz [Raport z danych Spark](#raport-z-danych-spark)).

## Instalacja

```bash
pip install buckets            # rdzeń (pandas)
pip install "buckets[spark]"   # + obsługa ramek Spark (wymaga JVM)
```

Praca nad źródłami: `uv sync` w katalogu repozytorium.

## Szybki start

### Zmienna dyskretna / kategoryczna — `bckt_stats`

```python
import pandas as pd
import buckets.buck as buck

df = pd.DataFrame({
    "segment": ["a", "a", "a", "d", "b", "b", "c"],
    "target":  [1, 1, 0, 0, 1, 0, 1],
})

buck.bckt_stats(df["segment"], df["target"])
```

Wynik — jeden wiersz na wartość zmiennej + wiersz `TOTAL`:

```text
       nr    bin discrete  ...  sum_target  n_obs  avg_target  pct_obs
a       1      a        a  ...           2      3    0.666667  0.428571
b       2      b        b  ...           1      2    0.500000  0.285714
c       3      c        c  ...           1      1    1.000000  0.142857
d       4      d        d  ...           0      1    0.000000  0.142857
TOTAL   5  TOTAL    TOTAL  ...           4      7    0.571429  1.000000
```

Braki danych zmiennej trafiają do osobnego bina `<NA>` (zawsze pierwszy
wiersz). Kontrakt kolumn i typów: [spec/typy-danych.md](spec/typy-danych.md).

### Zmienna ciągła — `bckt_cut_stats`

```python
# binowanie kwantylowe (bins=int) albo po jawnych granicach (bins=lista)
buck.bckt_cut_stats(df_kredyty["dochod"], df_kredyty["target"], bins=10)
buck.bckt_cut_stats(df_kredyty["dochod"], df_kredyty["target"], bins=[0, 5_000, 10_000, 1e9])
```

Dla zmiennej ciągłej tabela ma dodatkowo granice przedziałów (`od`,
`srodek`, `do`) oraz `mean`/`median` zmiennej w binie.

### Dyskretyzacja drzewem — `bckt_tree_stats`

```python
buck.bckt_tree_stats(df_kredyty, "dochod", "target",
                     max_depth=3, min_samples_split=100)
```

Granice binów wyznacza drzewo decyzyjne (sklearn) — biny różnicują target.

### GINI

```python
import buckets.statitics as st

st.gini(df["dochod"], df["target"])                      # pojedyncza wartość
st.gini(df["dochod"], df["target"], by=df["miesiac"])    # osobno per okres
```

## Klasa `BucketTable`

Funkcje `bckt_*` to cienkie nakładki na `BucketTable` — rdzeniową,
otypowaną tabelę agregatów per bin (bez wiersza `TOTAL`; ten powstaje
dopiero przy materializacji). Bezpośrednie użycie daje dostęp do operacji
analitycznych:

```python
from buckets.bucket_table import BucketTable

bt = BucketTable.from_quantiles(df["dochod"], df["target"], n_bins=20)
# pozostałe fabryki: from_discrete, from_bins, from_tree, from_auto

bt.to_frame()                    # DataFrame z TOTAL, sortowaniem, numeracją
bt.to_frame(min_info=True)       # tylko kluczowe kolumny
bt.score(df, "dochod")           # mapowanie wartości na avg_target bina
bt.plot(title="dochód")          # wykres avg_target po binach
bt.gini_discrete()               # gini liczone wprost z agregatów
```

## Raport HTML dla całej ramki

Role i typy kolumn opisuje `ColumnTypes` (typy analityczne wykrywane
automatycznie, role — heurystycznie: kolumna `target` → target,
`id*`/`*date*` → pomijane; wszystko można nadpisać):

```python
import buckets.column_types as ct
from buckets.report import DatasetReport
import buckets.report_html as report_html

types = ct.ColumnTypes(df)
types.time_col = "miesiac"       # opcjonalnie: gini w czasie
types.weights_col = "waga"       # opcjonalnie: wagi obserwacji

report = DatasetReport(df, types)
report_html.save(report.to_html(), "result/raport.html")
```

Raport zawiera dla każdej zmiennej objaśniającej: GINI (pełne i po
dyskretyzacji), GINI w czasie, tabelę dyskretyzacji i wykres bucketów.

### Wagi obserwacji

Wagi są traktowane jako **krotność obserwacji**: dla wag całkowitych każdy
wynik (buckety, mean/median, kwantyle, gini, drzewo) jest identyczny
z wynikiem na danych zreplikowanych wierszowo. Wagi przyjmują wszystkie
funkcje `bckt_*` (parametr `weights`), `st.gini` oraz raport (rola
`WEIGHTS` przez `types.weights_col` — wyłącznie jawnie, bez zgadywania
po nazwie kolumny).

### Rozkład zmiennej w czasie

```python
dot = buck.bckt_stats_over_time(df["miesiac"], df["segment"], df["target"])
dot.counts()        # liczności (sumy wag) czas × wartość
dot.distribution()  # udziały w obrębie okresu (suma = 1)
dot.avg_target()    # średni target w przecięciach
```

## Raport z danych Spark

Dane wierszowe nie są ściągane na drivera: Spark liczy per zmienna mały
agregat (`groupBy` + sumy wag), który po `toPandas()` wchodzi w standardowy
pipeline jako **ważone pseudo-obserwacje** — wyniki są identyczne jak na
danych wierszowych (przybliżenie pojawia się wyłącznie przy mikro-binowaniu
zmiennych o ekstremalnej liczbie unikalnych wartości, próg `max_levels`).
Projekt: [spec/raport-spark.md](spec/raport-spark.md).

```python
import buckets.spark as sp
from buckets.report import DatasetReport
import buckets.report_html as report_html

sdf = spark.read.parquet("dane.parquet")
sdf.persist()                    # zalecane: jeden groupBy na zmienną

types = sp.column_types_from_spark(sdf)
types.time_col = "miesiac"
types.weights_col = "waga"       # jeśli dane mają wagi

source = sp.SparkSource(sdf, types)
report = DatasetReport(source, source.types)   # uwaga: source.types
report_html.save(report.to_html(), "result/raport.html")
```

`pyspark` jest zależnością opcjonalną (`buckets[spark]`) i importowaną
leniwie — rdzeń pakietu działa bez Sparka.

Przykładowe skrypty end-to-end: `raport_spark_default*.py` w korzeniu
repozytorium (odpowiedniki pandasowych `test_raport_default*.py`).

> **Java:** Spark 4.x wymaga JVM 17 lub 21. Na nowszej Javie (24+) odczyt
> parquet pada z `getSubject is not supported` — wtedy wskaż starszą, np.:
> `JAVA_HOME=/usr/lib/jvm/temurin-21-jdk uv run python raport_spark_default.py`

## Rozwój

```bash
uv sync
uv run pytest tests/           # testy jednostkowe
uv run pytest tests/ -m spark  # testy sparkowe (wymagają pyspark + JVM)
uv run mkdocs serve            # dokumentacja API (mkdocstrings)
```

Dokumenty projektowe w [spec/](spec/): kontrakt typów
([typy-danych.md](spec/typy-danych.md)), architektura klas
([buck-refaktor-klasy.md](spec/buck-refaktor-klasy.md)), ścieżka Spark
([raport-spark.md](spec/raport-spark.md)), odłożone pomysły
([backlog.md](spec/backlog.md)).
