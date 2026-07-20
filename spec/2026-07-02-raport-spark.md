# Specyfikacja: raport z danych Spark (agregacja w Spark, analiza w pandas)

Cel: możliwość wygenerowania standardowego raportu (`DatasetReport` → HTML)
z danych trzymanych w ramce **Spark**, bez ściągania danych wierszowych na
drivera. Dane są **agregowane po stronie Sparka**, mały agregat trafia przez
`toPandas()` do pandas, a dalej pracują **istniejące mechanizmy** pakietu.

> Zakres: dokument opisuje docelowy kształt rozwiązania i wymagane zmiany
> w istniejącym kodzie. Sygnatury w blokach kodu są poglądowe.

Zasady projektowe (wprost z wymagania):

1. **Jak najmniej zmian** — istniejące klasy (`BucketTable`,
   `DistributionOverTime`, `VariableAnalysis`, `DatasetReport`) pozostają
   rdzeniem; nie powstaje równoległa "sparkowa" wersja raportu.
2. **Jak najmniej dublowania** — po stronie Sparka żyje wyłącznie kod
   agregujący (jeden mały moduł); cała analityka, typy, prezentacja i HTML
   zostają w jednym egzemplarzu, wspólnym dla obu źródeł danych.

---

## 1. Zasada przewodnia: agregat + ważone pseudo-obserwacje

Kluczowa obserwacja: **pipeline pandas jest już (prawie w całości) ważony**.
`_aggregate` ([bucket_table.py:60](../src/buckets/bucket_table.py#L60)) liczy
`sum(weights)`, `sum(weights*target)`, `sum(weights*pred)`;
`DistributionOverTime` ([over_time.py](../src/buckets/over_time.py)) pivotuje
ważone sumy. To znaczy, że **zagregowane dane ze Sparka można przedstawić jako
małą ramkę pandas ważonych pseudo-obserwacji** i wpuścić w istniejący kod bez
żadnych zmian w jego logice.

### Kanoniczny agregat (kontrakt)

Dla każdej analizowanej zmiennej Spark liczy **jeden** agregat:

```
groupBy(value [, time_period]) →
    n_obs      = sum(weights)            # weights = lit(1.0), gdy brak wag
    sum_target = sum(weights * target)
    sum_pred   = sum(weights * pred)     # tylko gdy pred zdefiniowane
```

gdzie `value` to:
- dla zmiennej **dyskretnej / kategorycznej** — surowa wartość zmiennej
  (null jako osobna grupa, odpowiednik `dropna=False`),
- dla zmiennej **ciągłej** — również surowa (unikalna) wartość, o ile liczba
  unikalnych wartości nie przekracza progu (sekcja 5.2); powyżej progu —
  środek mikro-binu kwantylowego.

**Uwaga — binowanie raportowe NIE zachodzi w Sparku.** Agregat dla zmiennej
ciągłej to drobnoziarnisty stan pośredni (jedna grupa na unikalną wartość),
a właściwe biny raportu (kwantylowe z `from_quantiles`, drzewiaste
z `from_tree`) wyznacza dopiero **pandas na pseudo-obserwacjach** — ważone
kwantyle po unikalnych wartościach dają dokładnie te same granice, co kwantyle
po surowych wierszach (stąd wymóg 3.1), a drzewo na wagach te same splity
(stąd 3.2). To świadoma decyzja: logika binowania istnieje w jednym
egzemplarzu, po stronie pandas; Spark umie wyłącznie sumować. Alternatywa
(binowanie w Sparku: `approxQuantile` na granice + `Bucketizer` +
`percentile_approx` na mediany) dublowałaby logikę binowania w dwóch silnikach
i była z natury przybliżona. Mikro-binowanie z sekcji 5.2 nie jest binowaniem
raportowym — to tylko bezpiecznik kardynalności stanu pośredniego.

Wymiar `time_period` jest dokładany, gdy `ColumnTypes.time_col` jest
zdefiniowane — wtedy z **tego samego** agregatu wychodzą buckety, gini,
gini w czasie i rozkład w czasie.

### Konwersja na pseudo-obserwacje

Każdy wiersz agregatu rozkładamy na **dwa** wiersze pandas:

```
(value, time, target=1, weights=sum_target,         pred=sum_pred/n_obs)
(value, time, target=0, weights=n_obs - sum_target, pred=sum_pred/n_obs)
```

(wiersze z wagą 0 można pominąć). Taka ramka wchodzi w istniejące funkcje jako
zwykłe dane z wagami.

### Dlaczego wyniki są identyczne (nie przybliżone)

| Operacja w pipeline | Dlaczego pseudo-obserwacje dają dokładnie ten sam wynik |
|---|---|
| `BucketTable.from_discrete` → `_aggregate` | `sum(w*t)` po pseudo-wierszach = `sum_target`; `sum(w)` = `n_obs` — algebra sum |
| `avg_target`, `pct_obs`, wiersz `TOTAL` | pochodne powyższych sum |
| `avg_pred` | `sum(w*pred)` = `n_obs * (sum_pred/n_obs)` = `sum_pred` |
| gini (`roc_auc_score`) | AUC zależy tylko od rozkładu (value → liczba 1 i 0); remisy na `value` obsługuje standardowa korekta 0.5 — identycznie jak na danych wierszowych; wymaga przekazania `sample_weight` (sekcja 3.3) |
| gini w czasie | jak wyżej, per grupa `time_period` |
| `DistributionOverTime` | wszystkie pivoty to ważone sumy po (czas × value) |
| drzewo decyzyjne | wewnątrz grupy o tej samej `value` żaden split nie rozdziela obserwacji; kryterium impurity na dwóch pseudo-wierszach (klasa 1 z wagą `sum_target`, klasa 0 z resztą) jest tożsame z surowymi danymi — wymaga `sample_weight` (sekcja 3.2) |
| `mean`/`median` per bin (ciągła) | ważona średnia / ważona mediana po unikalnych wartościach = dokładna średnia / mediana surowych danych (sekcja 3.1) |

Jedyne odstępstwo od dokładności pojawia się przy **mikro-binowaniu** bardzo
licznych zmiennych ciągłych (sekcja 5.2) — i jest to świadome, kontrolowane
przybliżenie.

---

## 2. Gdzie dziś kod dotyka danych wierszowych (inwentaryzacja)

| Miejsce | Operacja wierszowa | Pokrycie przez pseudo-obserwacje |
|---|---|---|
| [bucket_table.py:60](../src/buckets/bucket_table.py#L60) `_aggregate` | `groupby(var)` + ważone sumy | ✅ bez zmian |
| [bucket_table.py:184](../src/buckets/bucket_table.py#L184) `from_bins` | `pd.cut` + `mean`/`median` per bin | ⚠️ wymaga uwzględnienia wag w `mean`/`median` (3.1) |
| [bucket_table.py:238](../src/buckets/bucket_table.py#L238) `from_quantiles` | `variable.quantile(...)` | ⚠️ wymaga kwantyli ważonych (3.1) |
| [bucket_table.py:250](../src/buckets/bucket_table.py#L250) `from_tree` | sklearn na wierszach | ⚠️ wymaga `sample_weight` (3.2) |
| [statitics.py:6](../src/buckets/statitics.py#L6) `gini` | `roc_auc_score` na wierszach | ⚠️ wymaga parametru `weights` (3.3) |
| [report.py:56-87](../src/buckets/report.py#L56-L87) `VariableAnalysis.build` | `df[variable]`, `assign`, gini w czasie | ⚠️ musi dostawać ramkę per zmienna i przekazywać wagi (4.2) |
| [report.py:122](../src/buckets/report.py#L122) `nunique()` guard | liczba poziomów | ✅ = liczba wierszy agregatu |
| [column_types.py:137](../src/buckets/column_types.py#L137) `determine_column_types` | `dtype`, `nunique()` | ⚠️ wariant ze schematu Spark (4.3) |
| [over_time.py](../src/buckets/over_time.py) `DistributionOverTime` | pivoty ważone | ✅ bez zmian |
| [buck.py:249](../src/buckets/buck.py#L249) `assign` (scoring) | mapowanie wierszy | ⛔ poza zakresem raportu (sekcja 7) |

Wniosek: większość rdzenia działa od razu; do domknięcia są **luki w obsłudze
wag**, które są zresztą ukrytymi niespójnościami także w czystym pandas
(np. `from_bins` przyjmuje `weights`, ale liczy `mean`/`median` bez wag).

---

## 3. Zmiany w istniejącym kodzie: domknięcie obsługi wag

Wszystkie zmiany z tej sekcji są **backward-compatible** (przy `weights=None`
wersja ważona redukuje się do dzisiejszej) i poprawiają spójność również dla
użytkowników czysto pandasowych.

### 3.1 `BucketTable.from_bins` / `from_quantiles` — ważone statystyki

- `mean` per bin: `sum(w*x)/sum(w)` zamiast `mean=("variable", "mean")`
  ([bucket_table.py:191](../src/buckets/bucket_table.py#L191)).
- `median` per bin: **ważona mediana** (helper `weighted_median(x, w)` —
  sortowanie po `x`, skumulowana waga, punkt 50%).
- `total_mean` / `total_median`
  ([bucket_table.py:223-224](../src/buckets/bucket_table.py#L223-L224)):
  wersje ważone.
- `from_quantiles`: granice z **ważonych kwantyli** (helper
  `weighted_quantile(x, w, q)`), zamiast `variable.quantile`.

Helpery lądują w `statitics.py` (są ogólne, przydadzą się poza `BucketTable`).

### 3.2 `tree.make_tree` — `sample_weight`

Realizuje istniejące TODO ([tree.py:8](../src/buckets/tree.py#L8)):

```python
def make_tree(df, var, target, *, weights=None, max_depth=3, min_samples_leaf=50):
    tree.fit(X, y, sample_weight=weights)
```

**Ważny szczegół semantyczny:** `min_samples_leaf` w sklearn liczy *wiersze*,
nie sumę wag. Dla pseudo-obserwacji (mało wierszy o dużych wagach) należy
zamiast tego użyć `min_weight_fraction_leaf = min_samples / suma_wag` — inaczej
warunek "min. 100 obserwacji w liściu" przestaje cokolwiek znaczyć. `make_tree`
przy podanych `weights` ma przeliczać próg samodzielnie, żeby wywołujący
(`VariableAnalysis.build`) nie musiał znać tej subtelności.

### 3.3 `statitics.gini` — parametr `weights`

`roc_auc_score` wspiera `sample_weight` natywnie:

```python
def gini(var, target, by=None, weights=None, skipna=True): ...
    return 2 * roc_auc_score(t, v, sample_weight=w) - 1
```

Przy okazji można zrealizować pkt 5.1 backlogu (`include_groups=False`).

### 3.4 `ColumnTypes` — rola `WEIGHTS`

Żeby `DatasetReport` wiedział, którą kolumnę traktować jako wagi (dziś w ogóle
ich nie przekazuje), do `Role` dochodzi `WEIGHTS = "weights"`, a
`ColumnTypes` dostaje property `weights_col` (analogicznie do `time_col`).
Ramki pseudo-obserwacji zawsze niosą kolumnę wag — to jedyny sposób, żeby
raport ze Sparka przepływał przez `VariableAnalysis.build` bez specjalnych
gałęzi "if spark".

---

## 4. Nowe elementy

### 4.1 Moduł `buckets/spark.py` — jedyne miejsce znające PySpark

Import `pyspark` wyłącznie tutaj i wyłącznie leniwie (pakiet nie zyskuje
twardej zależności; `pyspark` jako *optional dependency* / extra
`buckets[spark]` w `pyproject.toml`).

```python
def aggregate_variable(
    sdf,                      # pyspark.sql.DataFrame
    var: str,
    target: str,
    pred: str | None = None,
    weights: str | None = None,
    time_col: str | None = None,
    max_levels: int = 100_000,   # próg mikro-binowania (5.2)
) -> pd.DataFrame:
    """Kanoniczny agregat z sekcji 1: groupBy(var[, time]) → toPandas();
    kolumny: value[, time], n_obs, sum_target[, sum_pred].

    Zmienna ciągła: grupowanie po SUROWEJ wartości — biny raportowe
    wyznaczy pandas na pseudo-obserwacjach (sekcja 1). Dopiero gdy
    approx_count_distinct(var) > max_levels, wartości są wcześniej
    zwijane do środków mikro-binów kwantylowych:
        edges = sdf.approxQuantile(var, [i/max_levels ...], eps)
        Bucketizer(splits=edges) → value = (od+do)/2 mikro-binu
    (null omija Bucketizer i zostaje osobną grupą <NA>)."""

def to_pseudo_obs(agg: pd.DataFrame, var: str) -> pd.DataFrame:
    """Agregat → ramka pseudo-obserwacji: kolumny [var, target, weights
    [, pred][, time]]. Czysty pandas — funkcja żyje tu tylko dlatego,
    że jest drugą połową kontraktu agregatu."""

def column_types_from_spark(sdf, discrete_threshold: int = 20) -> ct.ColumnTypes:
    """Buduje ColumnTypes ze schematu Spark: dtype z sdf.dtypes,
    nunique z approx_count_distinct (jeden job na całą ramkę),
    role heurystycznie jak w determine_column_types."""
```

To jest **cały** kod sparkowy — trzy funkcje. Żadna logika analityczna nie jest
tu powielana: moduł tylko liczy sumy i przestawia wynik do umówionego kształtu.

### 4.2 Seam w `DatasetReport`: źródło danych per zmienna

Dziś `DatasetReport` i `VariableAnalysis.build` sięgają wprost do `self.df`.
Wprowadzamy minimalny szew — **dostawcę ramki per zmienna**:

```python
class PandasSource:
    """Dziś zachowanie: wycina kolumny [var, target, time?, weights?] z df."""
    def frame_for(self, var: str) -> pd.DataFrame: ...
    def n_levels(self, var: str) -> int: ...          # nunique

class SparkSource:
    """aggregate_variable → to_pseudo_obs; cache agregatu per zmienna."""
    def frame_for(self, var: str) -> pd.DataFrame: ...
    def n_levels(self, var: str) -> int: ...          # len(agg)
```

`DatasetReport.__init__` przyjmuje `df: pd.DataFrame | SparkSource` (albo
fabryka `DatasetReport.from_spark(sdf, types)`) i wszędzie tam, gdzie dziś
stoi `self.df[column_name]` / `self.df[target]`, pracuje na
`source.frame_for(var)`. `VariableAnalysis.build` zamiast pełnego `df`
dostaje tę małą ramkę (i tak używa wyłącznie kolumn `variable`, `target`,
`time_col`) **plus wagi** — jedyna realna zmiana w jego treści to przekazanie
`weights` do `bckt_tree_stats`, `st.gini` i fabryk `BucketTable` (możliwe po
sekcji 3).

Ścieżka pandas przechodzi przez ten sam szew (z `weights=None`), więc **kod
raportu pozostaje jeden** — różni się tylko dostawca ramek.

### 4.3 `ColumnTypes` ze Sparka

`column_types_from_spark` (4.1) zwraca zwykły obiekt `ColumnTypes` — dalsza
konfiguracja (`time_col`, `set(...)`, role) działa identycznie jak dziś.
Mapowanie typów: typy numeryczne Sparka → logika `discrete`/`continuous` po
`approx_count_distinct`; `string`/`boolean` → `categorical`.

---

## 5. Dokładność i przypadki brzegowe

### 5.1 Typy `Int64` w `sum_target`/`n_obs`

Kanonizacja ([bucket_table.py:339-344](../src/buckets/bucket_table.py#L339-L344))
wybiera `Int64`, gdy wartości są całkowite. Pseudo-obserwacje mają
`target ∈ {0,1}` i wagi całkowite (gdy oryginalne wagi były całkowite/None),
więc sumy pozostają całkowite — **kontrakt typów z
[2026-05-31-typy-danych.md](2026-05-31-typy-danych.md) jest zachowany bez wyjątków**. To był główny
powód wyboru pseudo-obserwacji z rozbiciem 0/1 zamiast wariantu
`(value, target=avg_target, weights=n_obs)`.

### 5.2 Zmienna ciągła o ekstremalnej liczbie unikalnych wartości

Agregat po unikalnych wartościach jest dokładny, ale dla np. surowych floatów
może być wielki. Guard: gdy `approx_count_distinct(var) > max_levels`
(domyślnie 100 000), Spark najpierw binuje zmienną do `max_levels`
mikro-kwantyli (`approxQuantile` + `Bucketizer`, jako `value` bierzemy środek
mikro-binu) i dopiero to agreguje. Skutek: `median`/`mean`/granice binów/gini
stają się przybliżone — z błędem pomijalnym przy 100k mikro-binów. Próg jest
parametrem użytkownika.

### 5.3 Braki danych

Spark `groupBy` trzyma `null` jako osobną grupę → w pseudo-obserwacjach
`value = None` → pandas `NaN`/`pd.NA` → istniejąca obsługa bina `<NA>`
działa bez zmian. Walidacja "target bez braków" musi zostać wykonana po
stronie Sparka (tani `filter(target.isNull()).limit(1)`), bo do pandas braki
targetu już nie dotrą (zgubiłyby się w sumach).

### 5.4 Liczba jobów Spark

V1: jeden `groupBy` na zmienną (+ jeden `approx_count_distinct` zbiorczo dla
`ColumnTypes`). Zalecenie w docstringu: `sdf.persist()` przed raportem.
Batching wielu zmiennych w jeden job (melt/`grouping sets`) — świadomie
odłożony do backlogu (sekcja 7).

---

## 6. Plan wdrożenia (etapy niezależnie testowalne)

1. **Wagi w rdzeniu** (sekcja 3.1–3.3): `weighted_median`/`weighted_quantile`
   w `statitics.py`, ważone `mean`/`median`/kwantyle w `BucketTable`,
   `sample_weight` w `make_tree`, `weights` w `gini`. Testy: dla danych
   z wagami całkowitymi wynik = wynik na danych zreplikowanych wierszowo.
2. **Pseudo-obserwacje** (`to_pseudo_obs` + testy równoważności): raport
   z pseudo-obserwacji ≡ raport z surowych danych pandas (to test całego
   mechanizmu **bez** Sparka — kluczowy test tej specyfikacji).
3. **Seam `PandasSource`/`SparkSource` + rola `WEIGHTS`** (sekcje 3.4, 4.2):
   refaktor `DatasetReport`/`VariableAnalysis.build` na dostawcę ramek;
   istniejące testy raportu muszą przejść bez zmian referencji.
4. **Moduł `spark.py`** (sekcja 4.1) + extra `buckets[spark]`; testy na
   lokalnym `SparkSession` (marker `pytest.mark.spark`, pomijane bez pyspark).

Etapy 1–3 nie wymagają Sparka i samodzielnie poprawiają jakość pakietu.

---

## 7. Poza zakresem (kandydaci do backlogu)

- **Scoring ramki Spark** (`assign`/`score` po stronie Sparka): `BucketTable`
  mógłby generować wyrażenie `CASE WHEN` / `Bucketizer` z granic binów.
  Naturalne rozszerzenie, ale niezależne od raportu.
- **Batching agregacji** wielu zmiennych w jeden job Spark (melt do formatu
  długiego `(variable, value, target)` + jeden `groupBy`).
- **Inne silniki** (DuckDB, Polars, SQL): kontrakt agregat→pseudo-obserwacje
  jest silnikowo-agnostyczny — każdy backend to jedna funkcja
  `aggregate_variable`. Warto trzymać kontrakt w jednym miejscu (docstring
  `to_pseudo_obs`) właśnie pod ten kierunek.
