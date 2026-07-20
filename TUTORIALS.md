# Tutoriale — buckets krok po kroku

Seria ćwiczeń w formie list TODO: każdy tutorial przechodzi przez jeden
element pakietu, od podstawowych statystyk bucketów po raport z danych
Spark. Odhaczaj kroki po kolei — kod jest uruchamialny w REPL-u,
notebooku albo skrypcie.

Wszystkie tutoriale używają **nowego API klasowego**:

| Klasa | Moduł | Rola |
|---|---|---|
| `BucketTable` | `buckets.bucket_table` | tabela agregatów per bin (fabryki `from_*`) |
| `DistributionOverTime` | `buckets.over_time` | rozkład zmiennej i targetu w czasie |
| `ColumnTypes` | `buckets.column_types` | typy analityczne i role kolumn |
| `DatasetReport` | `buckets.report` | raport HTML dla całej ramki |
| `SparkSource` | `buckets.spark` | źródło danych per zmienna dla raportu ze Sparka |

Funkcje `buck.bckt_*` to cienkie nakładki na te klasy — tu używamy klas
bezpośrednio.

## Spis treści

- [Tutorial 0 — przygotowanie środowiska i danych](#tutorial-0)
- [Tutorial 1 — pierwsze buckety: zmienna dyskretna](#tutorial-1)
- [Tutorial 2 — zmienna typu `Categorical`: porządek kategorii](#tutorial-2)
- [Tutorial 3 — zmienna ciągła: kwantyle i jawne granice](#tutorial-3)
- [Tutorial 4 — dyskretyzacja drzewem](#tutorial-4)
- [Tutorial 5 — automat: `from_auto` i typy analityczne](#tutorial-5)
- [Tutorial 6 — GINI i wagi obserwacji](#tutorial-6)
- [Tutorial 7 — rozkład zmiennej w czasie: `DistributionOverTime`](#tutorial-7)
- [Tutorial 8 — raport HTML dla całej ramki](#tutorial-8)
- [Tutorial 9 (Spark) — agregat kanoniczny i pseudo-obserwacje](#tutorial-9)
- [Tutorial 10 (Spark) — analizy na danych rzeczywistych](#tutorial-10)
- [Tutorial 11 (Spark) — pełny raport HTML ze Sparka](#tutorial-11)
- [Co dalej](#co-dalej)

---

<a id="tutorial-0" name="tutorial-0"></a>

## Tutorial 0 — przygotowanie środowiska i danych

**Cel:** działające środowisko + dwie ramki danych, na których pracują
pozostałe tutoriale.

- [ ] Zsynchronizuj środowisko:

  ```bash
  uv sync                      # rdzeń (pandas) — tutoriale 1–8
  uv run --extra spark python  # tutoriale 9–11 (wymaga JVM 17/21)
  ```

- [ ] (Spark) Sprawdź Javę — Spark 4.x wymaga JVM 17 lub 21; na nowszej
  (24+) odczyt parquet pada z `getSubject is not supported`:

  ```bash
  JAVA_HOME=/usr/lib/jvm/java-21-temurin-jdk uv run --extra spark python ...
  ```

- [ ] Wygeneruj **dane syntetyczne** — samowystarczalne, używane w
  tutorialach 1–9. Zapisz poniższy fragment np. jako `dane_tut.py`
  i importuj z niego `df`:

  ```python
  import numpy as np
  import pandas as pd

  rng = np.random.default_rng(42)
  n = 10_000

  df = pd.DataFrame({
      "miesiac": rng.choice(
          ["2005-04", "2005-05", "2005-06", "2005-07", "2005-08", "2005-09"], n
      ),
      "segment": rng.choice(list("ABCD"), n, p=[0.4, 0.3, 0.2, 0.1]),
      "dochod": rng.lognormal(8.5, 0.6, n).round(2),
      "liczba_dzieci": rng.poisson(1.2, n),
  })

  # target zależy od dochodu i segmentu — jest co dyskretyzować
  logit = (
      1.0
      - 0.0004 * df["dochod"]
      + df["segment"].map({"A": -0.5, "B": 0.0, "C": 0.4, "D": 0.8})
  )
  df["pred"] = 1 / (1 + np.exp(-logit))            # "model idealny"
  df["target"] = (rng.random(n) < df["pred"]).astype(int)

  # braki danych w dochodzie (~10%) — pakiet traktuje je jako osobny bin
  df.loc[rng.random(n) < 0.1, "dochod"] = np.nan
  ```

- [ ] Przywróć **dane rzeczywiste** (default of credit card clients,
  UCI) — używane w tutorialach 10–11; są wersjonowane w repo, więc jeśli
  nie ma ich w katalogu `data/`:

  ```bash
  git restore data/
  ```

  Format long (klient × miesiąc) w `data/default_credit_card_long.parquet`
  buduje skrypt `convert_data.py`.

---

<a id="tutorial-1" name="tutorial-1"></a>

## Tutorial 1 — pierwsze buckety: zmienna dyskretna

**Cel:** policzyć statystyki targetu po wartościach zmiennej
kategorycznej klasą `BucketTable`.

- [ ] Zbuduj tabelę bucketów fabryką `from_discrete`:

  ```python
  from buckets.bucket_table import BucketTable

  bt = BucketTable.from_discrete(df["segment"], df["target"])
  bt.to_frame()
  ```

  Oczekuj jednego wiersza na wartość (`A`–`D`) + wiersza `TOTAL`;
  kluczowe kolumny: `n_obs`, `sum_target`, `avg_target`, `pct_obs`.

- [ ] Ogranicz wynik do najważniejszych kolumn i posortuj po ryzyku:

  ```python
  bt.to_frame(min_info=True, sort_by="avg_target", ascending=False)
  ```

- [ ] Sprawdź obsługę braków — wstaw NaN i zobacz bin `<NA>`
  (zawsze pierwszy wiersz):

  ```python
  seg_na = df["segment"].where(df.index % 7 != 0)   # ~14% braków
  BucketTable.from_discrete(seg_na, df["target"]).to_frame()
  ```

- [ ] Dodaj predykcję — dojdą kolumny `avg_pred` (kalibracja per bin):

  ```python
  BucketTable.from_discrete(df["segment"], df["target"], pred=df["pred"]).to_frame()
  ```

- [ ] Narysuj wykres i policz gini z agregatów:

  ```python
  bt.plot(title="segment")
  bt.gini_discrete()
  ```

**Sprawdź się:** `avg_target` powinno rosnąć od segmentu `A` do `D`
(tak skonstruowaliśmy dane), a suma `pct_obs` (bez TOTAL) dawać 1.

---

<a id="tutorial-2" name="tutorial-2"></a>

## Tutorial 2 — zmienna typu `Categorical`: porządek kategorii

**Cel:** zobaczyć, co pandasowa kategoria (zwłaszcza **uporządkowana**)
daje w bucketach — i gdzie są pułapki.

- [ ] Zbuduj uporządkowaną kategorię — `qcut` zwraca ją od razu
  (`mały < średni < duży`); dołóż pusty poziom `XXL`, żeby zobaczyć,
  jak pakiet traktuje kategorie bez obserwacji:

  ```python
  df["rozmiar"] = pd.qcut(
      df["dochod"], q=[0, 0.5, 0.8, 1], labels=["mały", "średni", "duży"]
  )
  df["rozmiar"] = df["rozmiar"].cat.add_categories(["XXL"])
  df["rozmiar"].dtype    # category, ordered=True
  ```

  Braki `dochod` przechodzą na braki `rozmiar` — przydadzą się za chwilę.

- [ ] Sprawdź klasyfikację: `Categorical` (nawet o etykietach
  liczbowych) to zawsze typ `categorical`:

  ```python
  import buckets.column_types as ct

  ct.guess_column_type(df["rozmiar"])   # → categorical
  ```

- [ ] Policz buckety — **kolejność wierszy respektuje porządek
  kategorii**, nie alfabet:

  ```python
  bt = BucketTable.from_discrete(df["rozmiar"], df["target"])
  bt.to_frame()
  ```

  Oczekuj kolejności: `<NA>`, `mały`, `średni`, `duży`, `XXL`, `TOTAL` —
  oraz malejącego `avg_target` (rozmiar rośnie z dochodem). Ta sama
  kolejność obowiązuje na wykresie `bt.plot()`.

- [ ] Zobacz, co tracisz **bez** kategorii — po rzutowaniu na string
  grupy wracają do porządku alfabetycznego (`duży` przed `mały`):

  ```python
  BucketTable.from_discrete(
      df["rozmiar"].astype("string"), df["target"]
  ).to_frame()
  ```

  Uwaga: rzutuj przez `astype("string")`, nie `astype(str)` — to drugie
  zamienia braki na literalny napis `"nan"`, który staje się osobną,
  fałszywą grupą zamiast bina `<NA>`.

- [ ] Droga powrotna — ze **stringa do kategorii**. Tak przywracasz
  porządek zmiennej, która przyszła jako zwykły tekst (z CSV, ze Sparka,
  z bazy):

  ```python
  roz_str = df["rozmiar"].astype("string")     # symulacja danych tekstowych

  typ = pd.CategoricalDtype(["mały", "średni", "duży"], ordered=True)
  roz_cat = roz_str.astype(typ)

  BucketTable.from_discrete(roz_cat, df["target"]).to_frame()
  ```

  Kolejność wierszy znów jest kategorialna. Dwie zasady:

  - wartości spoza listy `categories` (u nas `XXL`) stają się **brakami**
    i wpadają do bina `<NA>` — literówka w liście po cichu "gubi" grupę,
    więc porównaj `n_obs` przed i po konwersji;
  - bez znanej z góry listy poziomów użyj `roz_str.astype("category")` —
    kategorie zbierze z danych, ale porządek będzie alfabetyczny
    (`ordered=False`); listę do `CategoricalDtype` możesz wtedy podać
    ręcznie na podstawie `roz_str.dropna().unique()`.

- [ ] Pusty poziom `XXL` dostaje wiersz z `n_obs=0` i `avg_target=<NA>`
  (groupby z `observed=False`) — kategoria "widziana, ale pusta" jest
  raportowana, a nie ukrywana. Usuń nieużywane poziomy, jeśli tego nie
  chcesz:

  ```python
  BucketTable.from_discrete(
      df["rozmiar"].cat.remove_unused_categories(), df["target"]
  ).to_frame()
  ```

- [ ] **Pułapka (`spec/2026-05-31-backlog.md`, pkt 7):** kolumna `discrete`
  w wyniku jest stringifikowana — porządek kategorii niesie *kolejność
  wierszy*, ale nie typ kolumny. Sortowanie po niej jest więc
  alfabetyczne:

  ```python
  bt.to_frame(sort_by="discrete")   # XXL, duży, mały, średni — NIE rób tak
  bt.to_frame()                     # kolejność kategorialna — zostań przy domyślnej
  ```

- [ ] W czasie: `DistributionOverTime` (szerzej w Tutorialu 7) też
  układa kolumny pivotów wg
  porządku kategorii. Parametr `var_order` przydaje się, gdy dane
  przychodzą jako zwykłe stringi (np. ze Sparka — parquet z pandasową
  kategorią Spark czyta jako string i porządek wraca do alfabetycznego,
  patrz `raport_spark_default_amend.py`):

  ```python
  from buckets.over_time import DistributionOverTime

  dot = DistributionOverTime(
      df["miesiac"], df["rozmiar"].astype("string"), df["target"],
      var_order=["mały", "średni", "duży"],
  )
  dot.distribution()
  ```

**Sprawdź się:** tabela z kategorii ma wiersze w porządku
`mały → duży`, wersja stringowa — alfabetycznie; `XXL` znika po
`remove_unused_categories()`; kolumny `dot.distribution()` idą w
kolejności z `var_order`.

---

<a id="tutorial-3" name="tutorial-3"></a>

## Tutorial 3 — zmienna ciągła: kwantyle i jawne granice

**Cel:** zdyskretyzować zmienną ciągłą i odczytać zależność targetu od
jej poziomu.

- [ ] Binowanie kwantylowe — `from_quantiles`:

  ```python
  bt = BucketTable.from_quantiles(df["dochod"], df["target"], n_bins=10)
  bt.to_frame()
  ```

  Dla zmiennej ciągłej tabela ma dodatkowo granice (`od`, `srodek`,
  `do`) oraz `mean`/`median` zmiennej w binie. Braki `dochod` → bin
  `<NA>`.

- [ ] Binowanie po jawnych granicach — `from_bins` (biznesowe progi):

  ```python
  BucketTable.from_bins(
      df["dochod"], df["target"],
      bins=[0, 2_000, 4_000, 6_000, 10_000, 1e9],
  ).to_frame()
  ```

- [ ] Zobacz zależność na wykresie — malejący `avg_target` po `srodek`:

  ```python
  bt.plot(title="dochód (decyle)")
  ```

**Sprawdź się:** `avg_target` maleje z dochodem (ujemny współczynnik
w logicie z Tutorialu 0); wiersz `TOTAL` ma `mean`/`median` całej próby.

---

<a id="tutorial-4" name="tutorial-4"></a>

## Tutorial 4 — dyskretyzacja drzewem

**Cel:** pozwolić drzewu decyzyjnemu (sklearn) wyznaczyć granice binów
tak, żeby różnicowały target.

- [ ] Zbuduj buckety drzewem — `from_tree` przyjmuje ramkę i **nazwy**
  kolumn:

  ```python
  bt_tree = BucketTable.from_tree(
      df, "dochod", "target",
      max_depth=3, min_samples_split=500,
  )
  bt_tree.to_frame()
  ```

  Wiersze z NaN w zmiennej nie uczestniczą w budowie drzewa
  (`skipna=True`), ale trafiają do statystyk jako bin `<NA>`.

- [ ] Porównaj jakość dyskretyzacji z kwantylami:

  ```python
  bt_q = BucketTable.from_quantiles(df["dochod"], df["target"], n_bins=8)
  print("kwantyle:", bt_q.gini_discrete())
  print("drzewo:  ", bt_tree.gini_discrete())
  ```

- [ ] Użyj bucketów jako **transformacji zmiennej** — `score` mapuje
  wartości na `avg_target` bina (WoE-podobne kodowanie):

  ```python
  dochod_scr = bt_tree.score(df, "dochod")
  pd.concat([df["dochod"], dochod_scr], axis=1).head()
  ```

**Sprawdź się:** drzewo przy tej samej (lub mniejszej) liczbie binów
powinno mieć gini ≥ kwantyli — granice dobiera pod target.

---

<a id="tutorial-5" name="tutorial-5"></a>

## Tutorial 5 — automat: `from_auto` i typy analityczne

**Cel:** nie decydować ręcznie, czy zmienna jest dyskretna, czy ciągła.

- [ ] Zobacz, jak pakiet klasyfikuje zmienne (`guess_column_type`:
  nienumeryczna → `categorical`, numeryczna z `nunique() <
  discrete_threshold` → `discrete`, inaczej `continuous`):

  ```python
  import buckets.column_types as ct

  for col in ["segment", "liczba_dzieci", "dochod"]:
      print(col, "→", ct.guess_column_type(df[col]))
  ```

- [ ] Jedna fabryka dla każdego typu — `from_auto` sam wybiera
  `from_discrete` albo `from_quantiles`/`from_bins`:

  ```python
  BucketTable.from_auto(df["segment"], df["target"]).to_frame()        # categorical
  BucketTable.from_auto(df["liczba_dzieci"], df["target"]).to_frame()  # discrete
  BucketTable.from_auto(df["dochod"], df["target"], bins=10).to_frame()  # continuous
  ```

- [ ] Przesuń próg i zobacz zmianę decyzji — `liczba_dzieci` potraktowana
  jak ciągła:

  ```python
  BucketTable.from_auto(
      df["liczba_dzieci"], df["target"], bins=4, discrete_threshold=3
  ).to_frame()
  ```

**Sprawdź się:** atrybut `bt.kind` mówi, którą ścieżką poszła fabryka
(`Kind.DISCRETE` / `Kind.CONTINUOUS`).

---

<a id="tutorial-6" name="tutorial-6"></a>

## Tutorial 6 — GINI i wagi obserwacji

**Cel:** ocenić moc predykcyjną zmiennych i zrozumieć kontrakt wag
(waga = krotność obserwacji).

- [ ] Policz gini pojedynczej zmiennej i porównaj kilka:

  ```python
  import buckets.statitics as st

  for col in ["dochod", "liczba_dzieci", "pred"]:
      print(col, round(st.gini(df[col], df["target"]), 4))
  ```

  Uwaga: `st.gini` wymaga porządku wartości — dla zmiennej
  kategorycznej (`segment`) licz gini z tabeli bucketów
  (`bt.gini_discrete()`) albo na jej wersji po `score` (Tutorial 4).

- [ ] Gini **w czasie** — parametr `by` zwraca Series per okres:

  ```python
  st.gini(df["dochod"], df["target"], by=df["miesiac"])
  ```

- [ ] Wagi: sprawdź, że wagi całkowite są równoważne replikacji wierszy
  — to kontrakt całego pakietu (buckety, kwantyle, gini, drzewo):

  ```python
  w = pd.Series(rng.integers(1, 5, n), index=df.index).astype(float)
  df_rep = df.loc[df.index.repeat(w.astype(int))]      # dane zreplikowane

  bt_w   = BucketTable.from_discrete(df["segment"], df["target"], weights=w)
  bt_rep = BucketTable.from_discrete(df_rep["segment"], df_rep["target"])

  cols = ["n_obs", "sum_target", "avg_target"]
  assert bt_w.to_frame()[cols].equals(bt_rep.to_frame()[cols])

  assert st.gini(df["dochod"], df["target"], weights=w) == \
         st.gini(df_rep["dochod"], df_rep["target"])
  ```

**Sprawdź się:** `pred` ma najwyższe gini (to "model idealny");
oba asserty przechodzą bez wyjątku.

---

<a id="tutorial-7" name="tutorial-7"></a>

## Tutorial 7 — rozkład zmiennej w czasie: `DistributionOverTime`

**Cel:** zbadać stabilność zmiennej i targetu między okresami.

- [ ] Zbuduj obiekt — konstruktor przyjmuje serie czasu, zmiennej
  i targetu:

  ```python
  from buckets.over_time import DistributionOverTime

  dot = DistributionOverTime(
      df["miesiac"], df["segment"], df["target"], pred=df["pred"]
  )
  ```

- [ ] Odczytaj nazwane akcesory (każdy zwraca pivot czas × wartość):

  ```python
  dot.counts()        # liczności (sumy wag)
  dot.distribution()  # udziały w obrębie okresu (wiersze sumują się do 1)
  dot.avg_target()    # średni target w przecięciach
  dot.avg_pred()      # średnia predykcja (bo podaliśmy pred)
  dot.avg_target_total()  # średni target per okres (PIT)
  ```

- [ ] Narysuj wykresy Trellis (panel per wartość zmiennej) — moduł
  `buckets.trellis`:

  ```python
  import buckets.trellis as tr

  tr.plot_distribution(dot, title="segment — struktura w czasie")
  tr.plot_avg_target_by_bucket(dot, title="segment — target w czasie")
  tr.plot_avg_target_by_period(dot)
  tr.plot_pit_ttc(dot)   # target w czasie vs średnia predykcja
  ```

**Sprawdź się:** dane syntetyczne są stacjonarne, więc linie
`distribution()` powinny być niemal płaskie — na danych rzeczywistych
ten wykres wyłapuje przesunięcia populacji.

---

<a id="tutorial-8" name="tutorial-8"></a>

## Tutorial 8 — raport HTML dla całej ramki

**Cel:** jedna komenda → raport ze wszystkimi zmiennymi: gini (pełne
i po dyskretyzacji), gini w czasie, tabela dyskretyzacji, wykresy.

- [ ] Opisz kolumny obiektem `ColumnTypes` — typy analityczne wykrywane
  automatycznie, role heurystycznie (kolumna `target` → target,
  `id*`/`*date*` → pomijane):

  ```python
  import buckets.column_types as ct

  types = ct.ColumnTypes(df)
  types.types            # podgląd: dtype, analytical_type, role
  ```

- [ ] Skoryguj role: wskaż kolumnę czasu i wyłącz `pred` z analizy
  (nie jest zmienną objaśniającą):

  ```python
  types.time_col = "miesiac"
  types.types.loc["pred", "role"] = ct.Role.SKIPPED
  ```

  Analogicznie działa `types.weights_col = "waga"` (wagi wyłącznie
  jawnie — nigdy zgadywane po nazwie) oraz
  `types.set(["liczba_dzieci"], "continuous")` dla typów analitycznych.

- [ ] Zbuduj raport i zapisz HTML:

  ```python
  from buckets.report import DatasetReport
  import buckets.report_html as report_html

  report = DatasetReport(df, types)
  report_html.save(report.to_html(), "result/raport_tutorial.html")
  ```

- [ ] Zanim otworzysz HTML, podejrzyj składowe w Pythonie:

  ```python
  report.buckets()["dochod"]     # tabela dyskretyzacji per zmienna
  report.analyses()["segment"]   # pełny obiekt VariableAnalysis
  ```

**Sprawdź się:** raport ma sekcję dla `segment`, `dochod`
i `liczba_dzieci` (oraz `rozmiar`, jeśli ramka niesie ją z Tutorialu 2);
`pred` i `miesiac` nie są analizowane jako zmienne.

---

<a id="tutorial-9" name="tutorial-9"></a>

## Tutorial 9 (Spark) — agregat kanoniczny i pseudo-obserwacje

**Cel:** zrozumieć most Spark→pandas: dane wierszowe **nie schodzą**
z klastra — schodzi mały agregat, który wchodzi w standardowy pipeline
jako ważone pseudo-obserwacje z wynikami identycznymi jak na wierszach.

Uruchamiaj przez `uv run --extra spark python` (pamiętaj o `JAVA_HOME`,
Tutorial 0).

- [ ] Wrzuć syntetyczną ramkę do Sparka:

  ```python
  from pyspark.sql import SparkSession

  spark = (
      SparkSession.builder.master("local[*]")
      .appName("buckets-tutorial")
      .config("spark.ui.enabled", "false")
      .getOrCreate()
  )
  sdf = spark.createDataFrame(df)
  ```

- [ ] Policz **agregat kanoniczny** — jeden `groupBy` po stronie Sparka
  (kolumny: zmienna, `n_obs`, `sum_target`):

  ```python
  import buckets.spark as sp

  agg = sp.aggregate_variable(sdf, var="segment", target="target")
  agg
  ```

- [ ] Zamień agregat na **pseudo-obserwacje** — każdy wiersz rozpada się
  na `target=1` (waga `sum_target`) i `target=0` (waga
  `n_obs - sum_target`):

  ```python
  pobs = sp.to_pseudo_obs(agg)
  pobs
  ```

- [ ] Policz buckety na pseudo-obserwacjach i **sprawdź równoważność**
  ze ścieżką czysto pandasową:

  ```python
  bt_spark = BucketTable.from_discrete(
      pobs["segment"], pobs["target"], weights=pobs["weights"].astype(float)
  )
  bt_pandas = BucketTable.from_discrete(df["segment"], df["target"])

  cols = ["n_obs", "sum_target", "avg_target"]
  assert bt_spark.to_frame()[cols].equals(bt_pandas.to_frame()[cols])
  ```

**Sprawdź się:** assert przechodzi — to ta sama równoważność
waga=krotność co w Tutorialu 6; dowód i kontrakt agregatu:
`spec/2026-07-02-raport-spark.md`, sekcja 1.

---

<a id="tutorial-10" name="tutorial-10"></a>

## Tutorial 10 (Spark) — analizy na danych rzeczywistych

**Cel:** na danych credit card (format long, klient × miesiąc)
policzyć rozkład w czasie, gini w czasie i dyskretyzację drzewem —
wszystko z pseudo-obserwacji.

Wymaga `data/default_credit_card_long.parquet` (Tutorial 0). Pełny
skrypt referencyjny: `raport_spark_default.py`.

- [ ] Wczytaj dane i przygotuj kolumnę okresu:

  ```python
  import pyspark.sql.functions as F

  dane = spark.read.parquet("data/default_credit_card_long.parquet")
  dane = dane.withColumn("miesiac", F.date_format("date", "yyyy-MM"))
  dane = dane.drop("ID").persist()   # persist: jeden groupBy na zmienną
  ```

- [ ] Rozkład `pay_status` w czasie — agregat z `time_col` dokłada wymiar
  czasowy, pseudo-obserwacje zasilają `DistributionOverTime`:

  ```python
  agg = sp.aggregate_variable(dane, "pay_status", "target", time_col="miesiac")
  pobs = sp.to_pseudo_obs(agg)

  dot = DistributionOverTime(
      pobs["miesiac"], pobs["pay_status"], pobs["target"],
      weights=pobs["weights"].astype(float),
  )
  dot.distribution()
  ```

- [ ] Gini `pay_status` w czasie:

  ```python
  st.gini(pobs["pay_status"], pobs["target"],
          by=pobs["miesiac"], weights=pobs["weights"])
  ```

- [ ] Dyskretyzacja `credit_limit` drzewem — pseudo-obserwacje niosą
  wagi, więc drzewo widzi pełne krotności:

  ```python
  agg_cl = sp.aggregate_variable(dane, "credit_limit", "target")
  pobs_cl = sp.to_pseudo_obs(agg_cl)

  BucketTable.from_tree(
      pobs_cl, "credit_limit", "target", weights="weights",
      max_depth=3, min_samples_split=100,
  ).to_frame()
  ```

**Sprawdź się:** `dot.distribution()` ma 6 wierszy (2005-04…2005-09);
buckety drzewa dla `credit_limit` mają malejący `avg_target` — wyższy
limit, niższe ryzyko.

---

<a id="tutorial-11" name="tutorial-11"></a>

## Tutorial 11 (Spark) — pełny raport HTML ze Sparka

**Cel:** raport jak w Tutorialu 8, ale źródłem jest ramka Spark —
`SparkSource` liczy per zmienna agregat (jeden job, cache) i podaje
raportowi pseudo-obserwacje.

- [ ] Zbuduj `ColumnTypes` **ze schematu Spark** (bez ściągania danych;
  `approx_count_distinct`, więc typy na granicy progu można skorygować
  przez `types.set(...)`):

  ```python
  types = sp.column_types_from_spark(dane)
  types.time_col = "miesiac"
  print(types.types[["analytical_type", "role"]])
  ```

- [ ] Zbuduj raport przez `SparkSource` — uwaga: do `DatasetReport`
  podawaj `source.types`, nie oryginalny obiekt (źródło pracuje na
  kopii z zarejestrowaną kolumną wag pseudo-obserwacji):

  ```python
  from buckets.report import DatasetReport
  import buckets.report_html as report_html

  source = sp.SparkSource(dane, types)
  report = DatasetReport(source, source.types)
  report_html.save(report.to_html(), "result/raport_spark_tutorial.html")

  spark.stop()
  ```

- [ ] Porównaj z raportem pandasowym z tych samych danych
  (`test_raport_default.py` vs `raport_spark_default.py`) — sekcje
  i liczby są identyczne; różnice mogą pojawić się wyłącznie przy
  mikro-binowaniu zmiennych o ekstremalnej kardynalności
  (`max_levels`, `spec/2026-07-02-raport-spark.md` 5.2).

**Sprawdź się:** raport w `result/` zawiera dla każdej zmiennej gini
(pełne i po dyskretyzacji), gini w czasie, tabelę dyskretyzacji
i wykresy bucketów w czasie.

---

<a id="co-dalej" name="co-dalej"></a>

## Co dalej

- Kontrakt kolumn i typów wyników: `spec/2026-05-31-typy-danych.md`
- Architektura klas: `spec/2026-05-31-buck-refaktor-klasy.md`
- Ścieżka Spark (dowód równoważności pseudo-obserwacji): `spec/2026-07-02-raport-spark.md`
- Sekcje raportu "w czasie": `spec/2026-07-03-raport-w-czasie.md`
- Testy jako dodatkowe przykłady użycia: `tests/`
