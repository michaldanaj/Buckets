# Fasada `Dataset` — przezroczyste API pandas/Spark

Status: propozycja (backlog, pkt 8). Cel: użytkownik raz deklaruje, na czym
pracuje (ramka pandas albo Spark), a potem każdą analizę woła identycznie —
przez nazwy kolumn — nie wiedząc i nie musząc wiedzieć, którą ścieżką liczy
się wynik.

---

## 1. Motywacja: dzisiejsza asymetria

Ścieżka pandas jest **seriowa** — funkcje przyjmują kolumny jako `pd.Series`:

```python
buck.bckt_stats(df["segment"], df["target"])
st.gini(df["dochod"], df["target"], by=df["miesiac"])
```

Ścieżka Spark jest **nazwowa i ręczna** — użytkownik sam wykonuje most
agregat → pseudo-obserwacje i sam pilnuje wag:

```python
agg  = sp.aggregate_variable(sdf, "pay_status", "target", time_col="miesiac")
pobs = sp.to_pseudo_obs(agg)
wyn  = buck.bckt_stats_over_time(
    pobs["miesiac"], pobs["pay_status"], pobs["target"],
    weights=pobs["weights"].astype(float),      # ← łatwo zapomnieć
)
```

Ta wiedza ("najpierw agregat, potem pseudo-obserwacje, wagi rzutuj na float")
jest mechaniczna i powtarzalna — powinna siedzieć w bibliotece, nie w głowie
użytkownika.

**Połowa abstrakcji już istnieje.** `DatasetReport` przyjmuje źródło
o kontrakcie `frame_for(var)` / `n_levels(var)` i ma dwie implementacje:
`report.PandasSource` i `spark.SparkSource` (z cache'em agregatów per
zmienna). Problem: ta warstwa jest schowana **pod raportem** — analizy
ręczne (buckety, gini, drzewo, w czasie) z niej nie korzystają. Fasada
`Dataset` to wyciągnięcie tej warstwy przed nawias i dobudowanie na niej
metod analitycznych.

---

## 2. Docelowe API

```python
import buckets as bk

# jedyne miejsce, gdzie decyduje się pandas/Spark — dispatch po typie ramki
ds = bk.Dataset(df,  target="target", time_col="miesiac")             # pandas
ds = bk.Dataset(sdf, target="target", time_col="miesiac")             # Spark

# od tego momentu WSZĘDZIE ten sam interfejs: nazwy kolumn
ds.discrete("segment")                    # BucketTable
ds.quantiles("dochod", bins=10)           # BucketTable
ds.bins("dochod", edges=[0, 5e3, 1e9])    # BucketTable
ds.tree("dochod", max_depth=3)            # BucketTable
ds.stats("cokolwiek")                     # automat po typie analitycznym
ds.gini("dochod")                         # float
ds.gini("dochod", by="miesiac")           # pd.Series per okres
ds.over_time("segment")                   # DistributionOverTime
ds.report()                               # DatasetReport → .to_html()
```

Zasady:

- `target`, `pred`, `weights`, `time_col` deklarowane **raz** w konstruktorze
  (lub przez przekazany `ColumnTypes`) — nie w każdym wywołaniu,
- wszystkie metody odnoszą się do kolumn **po nazwie** — jedyna forma wspólna
  dla obu silników (dla Sparka serii nie ma),
- metody zwracają istniejące obiekty rdzenia (`BucketTable`,
  `DistributionOverTime`, `DatasetReport`) — fasada niczego nie duplikuje,
- konstruktor: `isinstance`-dispatch po typie ramki; import pyspark leniwy
  (jak dziś w `buckets.spark`) — rdzeń działa bez Sparka.

---

## 3. Mechanika: jeden prywatny punkt przecięcia

Cała różnica silników zamyka się w jednej metodzie:

```python
class Dataset:
    def _frame_for(self, var: str) -> tuple[pd.DataFrame, str | None]:
        """Ramka pandas do analizy zmiennej `var` + nazwa kolumny wag.

        pandas: oryginalny df + ewentualne wagi użytkownika (bez kopii).
        Spark:  cache'owany aggregate_variable(var[, time_col])
                → to_pseudo_obs → wagi pseudo-obserwacji (float).
        """
```

Każda metoda analityczna woła `_frame_for(var)` i deleguje do pandasowego
rdzenia. Dla Sparka to dokładnie dzisiejsza mechanika
`SparkSource._agg`/`frame_for` (memoizacja: jeden `groupBy` na zmienną,
`persist()` zalecany jak dotąd) — rozszerzona z użycia "tylko dla raportu"
na wszystkie analizy. Wyniki pozostają IDENTYCZNE z danymi wierszowymi
(dowód równoważności pseudo-obserwacji: [2026-07-02-raport-spark.md](2026-07-02-raport-spark.md),
sekcja 1); przybliżenie pojawia się wyłącznie przy mikro-binowaniu
(`max_levels`, tamże 5.2).

Relacja do istniejących klas:

- `PandasSource`/`SparkSource` → stają się wewnętrznymi silnikami `Dataset`
  (albo `Dataset` je zastępuje — do decyzji w implementacji; kontrakt
  `frame_for`/`n_levels` zostaje, więc `DatasetReport` nie wymaga zmian),
- `ds.report()` to cukier na `DatasetReport(source, source.types)` — znika
  pułapka "podawaj `source.types`, nie oryginalny `types`", bo fasada robi
  to sama,
- funkcje `buck.bckt_*` i fabryki `BucketTable.from_*` zostają bez zmian —
  fasada to warstwa NA rdzeniu, nie jego przepisanie; stare API dalej działa.

---

## 4. Dlaczego fasada, a nie "te same funkcje przyjmują obie ramki"

Rozważany wariant: `buck.bckt_stats(frame, "var")` z `isinstance` w każdej
funkcji. Odrzucony, bo:

- obecne API pandasowe jest seriowe — dla Sparka i tak trzeba przejść na
  nazwy kolumn, więc zmiana sygnatur jest nieunikniona; lepiej zrobić ją
  raz, w jednym obiekcie, niż w kilkunastu funkcjach,
- dispatch i stan (cache agregatów, kolumna wag, `ColumnTypes`) rozlałyby
  się po wszystkich funkcjach zamiast siedzieć w jednym miejscu,
- fasada nie rusza rdzenia; wariant funkcyjny wymuszałby przebudowę
  wszystkich sygnatur `bckt_*`.

Odrzucono też zewnętrzną warstwę abstrakcji ramek (narwhals/ibis) — ciężka
zależność, a nasz most pseudo-obserwacji i tak jest specyficzny dla domeny
(ważone agregaty targetu binarnego), nie do kupienia z półki.

---

## 5. Świadome przecieki abstrakcji

1. **`score`/`assign`** — zwracają wartość per wiersz, a wiersze w Sparku
   zostają na klastrze. `ds.score(var)` dla silnika Spark musi zwracać
   **ramkę Spark** (mapowanie granic binów przez `Bucketizer`/`when`,
   wykonane na klastrze), nie pandasową Series. Inny typ wyniku niż
   w pandas — zapisać jawnie w kontrakcie metody; ewentualnie osobna nazwa
   (`score_frame`?) zamiast udawania, że to to samo.
2. **Mikro-binowanie** — przy kardynalności > `max_levels` wyniki Spark są
   kontrolowanie przybliżone. Fasada tego nie ukrywa, tylko dokumentuje
   (parametr `max_levels` w konstruktorze, jak dziś w `SparkSource`).
3. **Porządek `Categorical`** — parquet czytany Sparkiem gubi pandasowy
   porządek kategorii (string, porządek alfabetyczny; backlog pkt 7,
   TUTORIALS.md tutorial 2). `Dataset` może przyjmować korektę per kolumna
   (np. `var_order=` / `CategoricalDtype`), przekazywaną do
   `DistributionOverTime` i sortowania bucketów.

---

## 6. Decyzje do podjęcia przed implementacją

- **Nazwa**: `Dataset`? (`Session`, `Frame`, `Analysis` — `Dataset` najbliżej
  istniejącego `DatasetReport`).
- **Zwrotki metod bucketowych**: `BucketTable` (obiekt, spójne z rdzeniem,
  `.to_frame()` na życzenie) czy od razu `DataFrame` (wygoda w REPL)?
  Propozycja: `BucketTable` — jedna konwencja, mniej magii.
- **Los `PandasSource`/`SparkSource`**: wchłonięte przez `Dataset` czy
  zostają jako silniki pod spodem? Propozycja: zostają (małe klasy, czysty
  kontrakt), `Dataset` je opakowuje.
- **`ds.score` dla Sparka**: w pierwszym etapie `NotImplementedError`
  z czytelnym komunikatem czy od razu implementacja na `Bucketizer`?
- **Eksport**: `buckets.Dataset` w `__init__.py` (dziś pusty) — to dobra
  okazja, by zdefiniować publiczne API pakietu.

---

## 7. Plan etapów

1. **Etap 1 — szkielet**: `Dataset` z dispatch pandas/Spark, `_frame_for`
   z cache, metody `discrete`/`quantiles`/`bins`/`tree`/`stats`/`gini`/
   `over_time`; testy równoważności: każda metoda daje na `Dataset(sdf)`
   wynik identyczny jak na `Dataset(df)` (reużycie istniejących testów
   pseudo-obserwacji).
2. **Etap 2 — raport**: `ds.report()`; `DatasetReport` przyjmuje `Dataset`
   (kontrakt `frame_for`/`n_levels` już pasuje); aktualizacja skryptów
   `raport_spark_default*.py` i TUTORIALS.md (tutoriale 9–11 upraszczają
   się do jednej-dwóch linii różnicy względem pandas).
3. **Etap 3 — scoring** (opcjonalny): `ds.score` dla Sparka na klastrze;
   korekta porządku `Categorical` (`var_order` per kolumna).
