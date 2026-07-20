# Specyfikacja: refaktoryzacja `buck.py` na strukturę klasową

Dokument projektuje podział obecnego, proceduralnego [src/buckets/buck.py](../src/buckets/buck.py)
na klasy i metody. Projekt od początku uwzględnia problemy z typami danych
opisane w [spec/2026-05-31-typy-danych.md](2026-05-31-typy-danych.md) — to one wyznaczają główną
decyzję architektoniczną.

> Zakres: dokument opisuje **docelowy kształt API i podział odpowiedzialności**,
> nie jest implementacją. Sygnatury w blokach kodu są poglądowe.

---

## 1. Zasada przewodnia

> **Model danych (otypowana tabela agregatów) jest oddzielony od prezentacji
> (wiersz `TOTAL`, kolejność wierszy, numeracja, ograniczanie kolumn).**

Źródłem chaosu typów jest dziś doklejanie wiersza `TOTAL` (a w ścieżce ciągłej
także `<NA>`) **do otypowanej tabeli**. Wstrzyknięcie stringa `"TOTAL"` albo
`pd.NA` w kolumnę numeryczną psuje jej dtype i wymusza łańcuch
`astype`/`convert_dtypes` rozsiany po całym kodzie (zob.
[buck.py:198-253](../src/buckets/buck.py#L198-L253)).

Rozwiązanie: rdzeniem jest klasa `BucketTable`, która trzyma **wyłącznie wiersze
binów** (łącznie z binem braków `<NA>`, bo to realne dane) z jednym, sztywnym
kontraktem typów. `TOTAL` nie jest nigdy przechowywany — powstaje dopiero
w metodzie prezentacyjnej `to_frame(total=True)`.

---

## 2. Diagnoza obecnej struktury

Plik to zbiór 11 funkcji o splątanych odpowiedzialnościach:

| Funkcja | Robi naprawdę | Problem |
|---|---|---|
| `bckt_stats` | agregacja dyskretna + total + sort + permutacja + dobór kolumn | jedna funkcja miesza 5 odpowiedzialności; typy łatane w środku |
| `bckt_cut_stats` | binowanie kwantylowe → deleguje do `bckt_stats` + dolicza `od/do/mean/median` | powiela logikę total/sort/permutacji; mapowanie bin→kwantyl pozycyjne (kruche) |
| `bckt_stats_over_time` | rozkład/target/pred w czasie | zwraca `list[DataFrame]` (na `main`), kontrakt niejasny |
| `bckt_tree_stats` | binowanie drzewem → `bckt_cut_stats` | OK, cienka nakładka |
| `bckt_guessed_type_stats` | dispatch po typie analitycznym | duplikuje logikę z `gen_buckets_for_df` |
| `gen_buckets_for_df` | iteracja po kolumnach df | duplikuje dispatch |
| `gen_report_objects` | pełny raport (gini, wykresy, over-time) | zwraca `list` pozycyjną — kruchy kontrakt (TODO w [buck.py:803](../src/buckets/buck.py#L803)) |
| `plot`, `plot_gini_over_time` | wykresy | wolne funkcje operujące na „gołym” DataFrame |
| `assign` | scoring (mapowanie wartości na biny) | operuje na DataFrame bez kontekstu, jaki to bucket |

Wspólne wady: powtórzony blok total+permutacja+`nr`, brak jednego miejsca decyzji
o typach, pozycyjne kontrakty zwracane (`list`), wolne funkcje wymagające
„magicznej” struktury kolumn.

---

## 3. Proponowany podział na klasy

```
           klasymetody-fabryki (strategie binowania):
           BucketTable.from_discrete()    (bckt_stats)
           BucketTable.from_bins()        (bckt_cut_stats, bins:list)  ← implementacja binowania ciągłego
           BucketTable.from_quantiles()   (bckt_cut_stats, bins:int)  ─┐ wyznaczają granice,
           BucketTable.from_tree()        (bckt_tree_stats)           ─┘ delegują do from_bins
           BucketTable.from_auto()        (bckt_guessed_type_stats)    (dispatch po typie)
                                 │ produkują
                                 ▼
                       ┌─────────────────────┐
                       │    BucketTable      │  RDZEŃ: agregaty per-bin
                       │  (kontrakt typów)   │  bez TOTAL; TOTAL on-demand
                       │  .to_frame()        │
                       │  .plot()  .score()  │
                       └─────────┬───────────┘
                                 │ używane przez
            ┌────────────────────┼────────────────────┐
   ┌────────▼─────────┐                       ┌────────▼──────────┐
   │ VariableAnalysis │                       │ DistributionOverTime
   │ (1 zmienna:      │                       │ (bckt_stats_over_time)
   │  bucket+gini+    │                       └───────────────────┘
   │  wykresy)        │
   └────────┬─────────┘
            ▼
   ┌──────────────────┐
   │  DatasetReport   │  iteracja po kolumnach df (gen_buckets/gen_report)
   └──────────────────┘
```

---

## 4. Klasy — odpowiedzialności i API

### 4.1. `BucketTable` — rdzeń (value object)

Centralna, otypowana tabela agregatów dla **jednej** zmiennej. Trzyma wyłącznie
wiersze binów (w tym bin `<NA>`), **nigdy** wiersza `TOTAL`.

```python
class BucketTable:
    def __init__(
        self,
        core: pd.DataFrame,      # agregaty per-bin, kontrakt typów z sekcji 6
        *,
        kind: Kind,              # DISCRETE | CONTINUOUS  (Enum)
        is_numeric: bool,        # czy zmienna wejściowa była numeryczna
        has_pred: bool,          # czy liczono avg_pred
    ): ...
```

**Atrybuty / właściwości** (read-only widoki na `core`):
`bins`, `n_obs`, `sum_target`, `avg_target`, `avg_pred`, `pct_obs`,
`od`, `srodek`, `do`, `mean`, `median`, `discrete`.

**Metody prezentacji** (jedyne miejsce, gdzie powstaje `TOTAL` i ustala się
kolejność/typy wyjścia):

```python
def to_frame(
    self,
    total: bool = True,
    min_info: bool = False,
    sort_by: str | None = None,
    ascending: bool = True,
) -> pd.DataFrame
```
- doklejenie `TOTAL` (jeśli `total`) — z jawnym typem `discrete`
  (`"TOTAL"` dla nienumerycznej, `pd.NA` dla numerycznej),
- sortowanie po `sort_by`,
- permutacja `<NA>` → … → `TOTAL` **zawsze** na końcu,
- renumeracja `nr`,
- dobór kolumn (`min_info`),
- finalny `convert_dtypes()`.

Zwraca DataFrame zgodny z dzisiejszym wyjściem `bckt_stats`/`bckt_cut_stats`
(wstecznie kompatybilny z testami przez `assert_frame_equal`).

**Operacje analityczne:**
```python
def plot(self, title=None) -> matplotlib.figure.Figure      # dawne plot()
def score(self, df, var, val="avg_target") -> pd.Series      # dawne assign()
def gini_discrete(self) -> float                             # gini z agregatów, biny sortowane po avg_target
```

`gini_discrete()` liczy gini zdyskretyzowanej zmiennej **wyłącznie z agregatów
bucketu** (`n_obs`, `sum_target`), z binami uporządkowanymi po `avg_target` —
nie potrzebuje surowych danych. Pełne GINI (na oryginalnej zmiennej ciągłej)
wymaga danych wierszowych i zostaje w `VariableAnalysis`, nie tutaj.

**Decyzja typowa (kluczowa):** `core` ma sztywny kontrakt z sekcji 6 i nigdy nie
zawiera wierszy psujących dtype. Dzięki temu z `BucketTable` znika cały rozrzucony
dziś łańcuch konwersji — kanonizacja typów jest **raz**, na końcu `to_frame()`.
`TOTAL` jako warstwa prezentacji nie kontaminuje modelu.

**Decyzja: bin `<NA>` zostaje w `core`, `TOTAL` nie.**
Bin braków to realne obserwacje (zmienna miała `NaN`) — jego agregaty to
poprawne liczby, `discrete = pd.NA` mieści się w `Float64`/`Int64`.
`TOTAL` to wiersz-podsumowanie, którego `discrete`/`od`/`do` nie mają sensu
i wymuszałyby `pd.NA`/string w kolumnach — dlatego liczony dopiero przy
materializacji.

---

### 4.2. Strategie binowania — klasymetody-fabryki `BucketTable`

Strategie binowania udostępniamy jako **klasymetody-fabryki** na `BucketTable`,
a nie jako hierarchię klas `Binner` (decyzja — sekcja 7, pkt 1). Każda fabryka
bierze surowe serie i produkuje `BucketTable`:

```python
class BucketTable:
    @classmethod
    def from_discrete(cls, var, target, pred=None, weights=None) -> "BucketTable": ...
    @classmethod
    def from_quantiles(cls, variable, target, *, n_bins=50, pred=None, weights=None) -> "BucketTable": ...
    @classmethod
    def from_bins(cls, variable, target, *, bins: list[float], pred=None, weights=None) -> "BucketTable": ...
    @classmethod
    def from_tree(cls, df, var, target, *, max_depth=3, min_samples_split=2) -> "BucketTable": ...
    @classmethod
    def from_auto(cls, variable, target, *, discrete_threshold=20, ...) -> "BucketTable": ...
```

Wspólną logikę (dziś skopiowaną w każdej funkcji) trzyma **jedna prywatna funkcja
agregująca** `_aggregate(...)`, wołana przez fabryki:
- walidacja braków w `target` (`ValueError`),
- domyślne `weights = 1`,
- domyślne `pred = target` + flaga `has_pred`,
- **zachowanie naturalnego typu zmiennej** (bez promocji `Int64 → Float64` —
  patrz sekcja 6, reguła 1),
- rdzeniowa agregacja `groupby(..., dropna=False, sort=False, observed=False)`.

#### `from_discrete` (dawne `bckt_stats`)
Grupuje po wartościach zmiennej. `discrete` = kopia indeksu grup (zachowuje typ
wejścia), `od/srodek/do/mean/median` = `pd.NA` (`Float64`).

#### `from_bins` (dawne `bckt_cut_stats` z `bins: list[float]`) — implementacja binowania ciągłego
`from_bins` jest **jedyną** implementacją binowania zmiennej ciągłej; pozostałe
fabryki ciągłe tylko wyznaczają granice i do niej delegują.
- `bins: list[float]` — jawnie podane granice przedziałów,
- walidacja: `variable` i `target` numeryczne,
- binowanie `pd.cut(..., ordered=True)`,
- agregacja przez `_aggregate` na etykietach binów,
- `mean/median` z oryginalnej zmiennej per bin,
- `od/do` przypisane przez **mapę `bin → (od, do)`** (nie pozycyjnie) — usuwa
  kruchość zdiagnozowaną w [spek typów](2026-05-31-typy-danych.md),
- `discrete = pd.NA` we wszystkich wierszach (przedział opisują `od/srodek/do`).

#### `from_quantiles` (dawne `bckt_cut_stats` z `bins: int`)
Wyznacza granice z kwantyli (`variable.quantile(...)` dla `n_bins` progów,
`drop_duplicates`), po czym **deleguje do `from_bins`**. Cienka nakładka.

#### `from_tree` (dawne `bckt_tree_stats`)
Wyznacza granice drzewem (`tree.make_tree` + `extract_leaf_bounds`), po czym
**deleguje do `from_bins`**. Cienka nakładka.

> Rozdzielenie „liczba kwantyli" (`from_quantiles`) i „jawne granice" (`from_bins`)
> na osobne fabryki: dwie różne semantyki, trzymanie ich w jednym
> `bins: int | list[float]` było pozostałością po `bckt_cut_stats` i po przejściu
> na fabryki przestało mieć sens. `from_quantiles` i `from_tree` to teraz
> jednolicie „wyznacz granice → `from_bins`".

> **Dlaczego fabryki, a nie hierarchia `Binner`:** mniej typów do ogarnięcia,
> naturalne dla biblioteki, a wspólną logikę i tak trzyma jedna prywatna funkcja
> agregująca. Hierarchię `Binner` (ze stanem/parametrami strategii) wprowadzić
> dopiero, gdy pojawi się potrzeba binowania na innym backendzie (np. Spark —
> patrz TODO w [buck.py:38-42](../src/buckets/buck.py#L38)).

---

### 4.3. `from_auto` / dispatch po typie (dawne `bckt_guessed_type_stats`)

Fabryka `BucketTable.from_auto(...)` wybiera strategię na podstawie
`ct.guess_column_type`:
- `discrete`/`categorical` → `from_discrete` (z limitem `categorical_max_levels`),
- `continuous` → `from_quantiles`.

Zwraca `BucketTable` (albo obiekt sygnalizujący „za dużo poziomów” — patrz
sekcja 7, pkt 3). Tę samą logikę dispatchu używa `DatasetReport`, więc **nie
ma już duplikacji** między `bckt_guessed_type_stats` a `gen_buckets_for_df`.

---

### 4.4. `DistributionOverTime` (dawne `bckt_stats_over_time`)

Rozwiązuje konflikt API `typy` (pojedynczy pivot) vs `main` (lista 4 ramek):
zamiast zwracać `list`/pojedynczy frame — **klasa z nazwanymi akcesorami**.

```python
class DistributionOverTime:
    def __init__(self, czas, var, target, pred=None, weights=None): ...
    def counts(self) -> pd.DataFrame          # liczności w czasie
    def distribution(self) -> pd.DataFrame    # rozkład znormalizowany (suma=1 w okresie)
    def avg_target(self) -> pd.DataFrame
    def avg_pred(self) -> pd.DataFrame | None # None gdy bez pred
```

Pivoty liczone leniwie/cache'owane. Eliminuje kontrakt pozycyjny i jest
nadzbiorem obu wcześniejszych wariantów.

> **Uwaga:** docelowo `DistributionOverTime` ma wchodzić do domyślnej analizy
> zmiennej (`VariableAnalysis.build()`), gdy zdefiniowana jest główna kolumna
> czasowa — nie tylko stać obok. Opis: [2026-05-31-backlog.md](2026-05-31-backlog.md), pkt 2.

---

### 4.5. `VariableAnalysis` (rozbicie `gen_report_objects` per zmienna)

Realizuje TODO z [buck.py:803](../src/buckets/buck.py#L803) (klasa zamiast listy
pozycyjnej). Reprezentuje komplet analizy jednej zmiennej:

```python
class VariableAnalysis:
    name: str
    buckets: BucketTable
    discrete: BucketTable          # dyskretyzacja (dla ciągłych — drzewem)
    gini: pd.DataFrame
    gini_over_time: pd.DataFrame | None
    fig_buckets: Figure
    fig_gini_over_time: Figure | None

    @classmethod
    def build(cls, df, var, types) -> "VariableAnalysis": ...
    def to_report_payload(self) -> list   # adapter do report_html (zgodność)
```

Każdy element ma jawny typ i jest walidowany przy budowie (dziś błąd typu
wychodzi dopiero w `report_html`). `to_report_payload()` zachowuje obecny
kontrakt listy dla [report_html.py](../src/buckets/report_html.py) na czas migracji.

> **Uwaga:** sztywne pola powyżej to świadomy etap pierwszej implementacji.
> Docelowe rozwiązanie jest inne — `VariableAnalysis` ma się stać otwartą
> kolekcją elementów (`ReportElement`), do której można dodać dowolną analizę
> (tabelę, wykres, tekst). Opis wymagania i propozycji: [2026-05-31-backlog.md](2026-05-31-backlog.md).

---

### 4.6. `DatasetReport` (dawne `gen_buckets_for_df` + `gen_report_objects`)

Orkiestracja po kolumnach ramki, wg ról z `ct.ColumnTypes`:

```python
class DatasetReport:
    def __init__(self, df, types: ct.ColumnTypes, categorical_max_levels=20): ...
    def analyses(self) -> dict[str, VariableAnalysis]: ...
    def to_html(self) -> str             # deleguje do report_html.generate_report
```

Pętla, pomijanie ról `skipped`/`target`/`main_time_col`, obsługa „za dużo
poziomów” — w jednym miejscu, bez duplikacji dispatchu.

---

## 5. Mapowanie stare → nowe

| Dziś (funkcja) | Docelowo |
|---|---|
| `bckt_stats(...)` | `BucketTable.from_discrete(...).to_frame(...)` |
| `bckt_cut_stats(..., bins=int)` | `BucketTable.from_quantiles(...).to_frame(...)` |
| `bckt_cut_stats(..., bins=list)` | `BucketTable.from_bins(...).to_frame(...)` |
| `bckt_tree_stats(...)` | `BucketTable.from_tree(...)` |
| `bckt_guessed_type_stats(...)` | `BucketTable.from_auto(...)` |
| `bckt_stats_over_time(...)` | `DistributionOverTime(...)` + akcesory |
| `plot(bucket, ...)` | `BucketTable.plot(...)` |
| `plot_gini_over_time(...)` | `VariableAnalysis.fig_gini_over_time` (buduje wewnętrznie) |
| `assign(df, var, buckets, val)` | `BucketTable.score(df, var, val)` |
| `gen_buckets_for_df(...)` | `DatasetReport.analyses()` |
| `gen_report_objects(...)` | `DatasetReport` + `VariableAnalysis` |

Cienkie funkcje-fasady o starych nazwach można zachować jako `@deprecated`
opakowania na czas migracji testów i [report_html.py](../src/buckets/report_html.py).

---

## 6. Kontrakt typów (utrwalony w `BucketTable._canonicalize`)

Powtórzony z [spec/2026-05-31-typy-danych.md](2026-05-31-typy-danych.md) — egzekwowany w jednym miejscu.

| Kolumna | Typ | Uwagi |
|---|---|---|
| indeks `core` | `RangeIndex` | `core` nie używa binu jako indeksu; `to_frame` ustawia indeks=`bin` (decyzja, sekcja 7 pkt 8) |
| `bin` | `string` | stringowa reprezentacja wartości/przedziału; `<NA>` = `NA_BIN_NAME` |
| `discrete` | typ wejścia (`Int64`/`Float64`/`string`) | `TOTAL`: `"TOTAL"` (nienumeryczne) / `pd.NA` (numeryczne); dla ciągłych = `pd.NA` |
| `nr` | `Int64` | nadawane w `to_frame` po permutacji |
| `od`, `srodek`, `do`, `mean`, `median` | `Float64` | bywają złożone z samych braków → muszą być rozszerzone |
| `sum_target`, `n_obs` | `Int64` | (gdy wagi całkowite; przy wagach ułamkowych `Float64`) |
| `avg_target`, `avg_pred`, `pct_obs` | `Float64` | |

Reguły ogólne (z [spek typów](2026-05-31-typy-danych.md)):
1. **zachowanie naturalnego typu zmiennej** — `Int64` zostaje `Int64`, `Float64`
   zostaje `Float64`. Świadome odejście od podejścia z gałęzi `typy`, która
   promowała numeryczną z brakami `Int64 → Float64` przed `groupby`. Powód:
   `Int64` jest nullable, więc trzyma `pd.NA` bez promocji, a `discrete` powinien
   zachować całkowitoliczbowość zmiennej. Promocja w `typy` była naginaniem kodu
   pod arbitralne oczekiwanie testu (`bin = "8.0"` zamiast `"8"`), a nie naprawą
   realnego problemu — różnica `"8"`/`"8.0"` dotyczy tylko etykiety `bin` i nie
   wpływa na scoring (`assign` mapuje po numerycznym `discrete`, nie po stringu).
   **Konsekwencja:** dotychczasowe testy oczekujące `"8.0"`/`Float64` trzeba będzie
   poprawić — to osobna kwestia do ogarnięcia przy migracji testów,
2. indeks wyjścia `object`/`string` ustalany w `to_frame`, nie w środku,
3. jeden punkt kanonizacji typów (`convert_dtypes` + mapa `astype`) na końcu,
4. permutacja `<NA>`/`TOTAL` i `nr` zawsze na końcu, niezależnie od `sort_by`,
5. braki w indeksie tylko jako `NA_BIN_NAME`, nigdy `pd.NA` (unika `ValueError`
   przy `.isin`/hashowaniu).

### 6.1. `_canonicalize`: kolejność `convert_dtypes` → `astype` (jawna mapa wygrywa)

Kanonizacja typów (reguła 3) to **kod produkcyjny** w `BucketTable._canonicalize`,
wołany na końcu `to_frame()` — nie jest to tylko zabieg testowy.

Problem z samym `convert_dtypes()`: **dobiera typy na podstawie danych**, nie
deklaracji, i potrafi złamać kontrakt z tabeli powyżej. W szczególności kolumnę
zmiennoprzecinkową o samych wartościach całkowitych zrzuca do `Int64`:

```python
pd.Series([1.0, 2.0]).convert_dtypes()   # -> Int64, NIE Float64!
```

Gdyby `mean`/`median`/`od`/`srodek`/`do` zawierały akurat liczby całkowite,
`convert_dtypes()` zepsułby ich zadeklarowany `Float64`. To ta sama klasa
niestabilności, którą opisuje [spec/2026-05-31-typy-danych.md](2026-05-31-typy-danych.md).

Dlatego **jawna mapa `astype` jest jedynym źródłem prawdy i jest aplikowana jako
ostatnia**, żeby wygrywała z inferencją. Sekwencja w `_canonicalize`:

1. **`convert_dtypes()`** — pierwszy przebieg: podnosi kolumny **nieobjęte** jawną
   mapą do typów rozszerzonych (`object→string`, `int64→Int64`) i zapewnia wsparcie
   dla `pd.NA`.
2. **`astype(mapa)`** — wymusza zadeklarowany typ na **wszystkich** kolumnach
   kontraktu z tabeli w sekcji 6. To deklaracja, nie zgadywanie — i ma ostatnie
   słowo. Mapa aplikowana tylko do kolumn faktycznie obecnych w wyniku
   (`min_info`, opcjonalne `avg_pred`):
   ```python
   astype_map = {k: v for k, v in TYPE_CONTRACT.items() if k in wyn.columns}
   wyn = wyn.convert_dtypes().astype(astype_map)
   ```

**Konsekwencja dla testów:** skoro to kod narzuca kontrakt, referencje testowe też
powinny **deklarować typy jawnie** (`astype`), a nie polegać na `convert_dtypes()`.
Inaczej test i kod „zgadują" niezależnie i mogą się rozjechać.

---

## 7. Decyzje i kompromisy

1. **Klasymetody-fabryki zamiast pełnej hierarchii `Binner`.** Hierarchia
   strategii jest „podręcznikowa”, ale dziś zysk jest mały (3 strategie,
   wspólny rdzeń). Klasymetody dają to samo przy mniejszej liczbie typów.
   Hierarchię wprowadzić, gdy pojawi się drugi backend (Spark).

2. **`TOTAL` poza modelem danych.** Najważniejsza decyzja — bezpośrednio leczy
   chaos typów. Koszt: każdy konsument, który dziś czyta `wyn.loc["TOTAL"]`,
   musi wołać `to_frame(total=True)`. Akceptowalne — to warstwa prezentacji.

3. **„Too many categorical levels” — typ wyniku.** Dziś zwracany jest „udawany”
   DataFrame z kolumną `warning` (sprawdzany przez `buckets.columns[0] == "warning"`
   w [buck.py:754](../src/buckets/buck.py#L754)). Docelowo: osobny obiekt
   `SkippedVariable(reason=...)` lub `BucketTable` w stanie pustym z flagą
   `.skipped`. Czystsze niż sprawdzanie nazwy kolumny. **Do decyzji.**

4. **Zgodność wsteczna przez `to_frame`.** Wyjście `to_frame()` ma pasować do
   referencji testowych (`assert_frame_equal`), żeby migrować bez przepisywania
   wszystkich testów naraz. **Zastrzeżenie:** dotyczy to referencji *poprawnych*
   wg kontraktu typów — część dzisiejszych referencji koduje błędne oczekiwania
   (sprzeczne typy, `"8.0"` zamiast `"8"`) i te są prostowane przy migracji
   (sekcja 9), a nie traktowane jako wzorzec.

5. **`DistributionOverTime` jako klasa z akcesorami** rozwiązuje rozjazd
   `typy` vs `main` bez wybierania „jednego słusznego” kształtu zwracanego.

6. **Wykresy jako metody.** `plot`/`plot_gini_over_time` przestają być wolnymi
   funkcjami wymagającymi konkretnego układu kolumn — stają się metodami obiektów,
   które gwarantują ten układ.

7. **`BucketTable` niemutowalny.** `core` jest read-only; `to_frame` i operacje
   nigdy nie zmieniają stanu obiektu, tylko zwracają nowy DataFrame/obiekt.
   Przewidywalne i łatwiejsze do testowania.

8. **`to_frame` zwraca indeks = `bin`.** Zachowuje zgodność z dzisiejszymi
   referencjami testowymi i minimalizuje migrację. Czystsza alternatywa
   (`RangeIndex` + kolumna `bin`) trafia do [2026-05-31-backlog.md](2026-05-31-backlog.md) jako
   późniejsze porządkowanie.

---

## 8. Kolejność wdrożenia (etapy)

1. **Fundament testowy** (sekcja 9) — helper referencji nakładający jawnie typy
   z kontraktu (sekcja 6) zamiast `convert_dtypes`, oraz dedykowany test kontraktu
   typów. Daje stabilny punkt odniesienia dla wszystkich kolejnych etapów.
2. **`BucketTable` + `from_discrete` + `to_frame`** — przepisać `bckt_stats`
   tak, by stał się fasadą `from_discrete(...).to_frame(...)`. Równolegle: triage
   dotychczasowych testów `bckt_stats` (sekcja 9, pkt 4) — czerwone rozdzielić na
   „błąd testu" i „świadoma zmiana zachowania". **Nie** zakładać, że stare testy
   przechodzą bez zmian — część koduje błędne oczekiwania typów.
3. **`from_bins` + `from_quantiles` + `from_tree`** — `from_bins` jako jedyna
   implementacja binowania ciągłego (z mapą `bin → (od, do)`), `from_quantiles`
   i `from_tree` delegują do niej. Triage testów `bckt_cut_stats`/`bckt_tree_stats`.
4. **`from_auto`** — dispatch po typie (cienka nakładka).
5. **Metody `BucketTable`** — `score` (dawne `assign`), `plot`, `gini_discrete`.
6. **`DistributionOverTime`**.
7. **`VariableAnalysis` + `DatasetReport`** — rozbić `gen_report_objects`,
   dodać adapter `to_report_payload()` dla [report_html.py](../src/buckets/report_html.py).
8. Usunięcie fasad `@deprecated` po przepięciu testów i raportu.

---

## 9. Strategia testów (kontrakt typów)

Część dzisiejszych testów ma **przypadkowe** oczekiwania typów, bo referencje są
budowane przez **inferencję** (`DataFrame.from_records(...).convert_dtypes()` albo
`from_dict`), a nie przez **deklarację**. Test sprawdza wtedy „co pandas zgadł",
a nie „co kontrakt każe" — stąd sprzeczności (te same dane `test_df_2` raz `Int64`,
raz `float64`) i oczekiwanie `"8.0"` zamiast `"8"`.

1. **Jedno źródło prawdy: kontrakt typów (sekcja 6).** Testy sprawdzają zgodność
   z kontraktem, nie z tym, co wyjdzie z `convert_dtypes`. Ta sama zasada co dla
   kodu (`astype` wygrywa, sekcja 6.1) — inaczej kod i test „zgadują" niezależnie.

2. **Referencje budowane przez deklarację, nie inferencję.** Zastąpić helper
   `df_from_array` (dziś `...convert_dtypes()`) wersją nakładającą **jawną mapę
   `astype`** z kontraktu. Dtype referencji staje się zadeklarowany i
   deterministyczny, niezależny od konkretnych wartości w danych.

3. **Rozdzielić dwa rodzaje asercji:**
   - **testy wartości** — czy liczby się zgadzają,
   - **dedykowany test kontraktu typów** — dla reprezentatywnych wejść
     (kategoryczne / dyskretne numeryczne bez braków / dyskretne numeryczne z
     brakami / ciągłe) sprawdza `result.dtypes` kolumna po kolumnie. Pilnuje m.in.,
     że `discrete` to `Int64`, **nie** `Float64`, a `bin` to `"8"`, nie `"8.0"`.

   Dzięki temu zmiana kontraktu dotyka **jednego** testu, a nie wielu referencji
   (dziś każda referencja koduje oczekiwanie typu osobno — stąd sprzeczności).

4. **Triage istniejących testów** — każdy fail zakwalifikować do:
   - **błąd testu** (np. `KeyError df["weights"]`, przestawione `od/srodek/do`
     z copy-paste, sprzeczne oczekiwania) → poprawić test,
   - **świadoma zmiana zachowania** (np. `discrete` = `Int64` zamiast `Float64`,
     `bin` = `"8"`) → zaktualizować oczekiwanie **zgodnie z kontraktem** i
     odnotować dlaczego.

   Analiza z gałęzi `typy` ([spec/2026-05-31-typy-danych.md](2026-05-31-typy-danych.md)) już częściowo
   ten triage wykonała.

5. **`assert_frame_equal(..., check_dtype=True)`** — **decyzja**. Strict łapie
   przypadkowe regresje typów, a poprawny dtype referencji gwarantuje helper z
   pkt 2. Nie używać `check_dtype=False` — to ukryłoby właśnie ten rodzaj błędów,
   który chcemy kontrolować.

---

## 10. Otwarte pytania

1. Reprezentacja „za dużo poziomów” — `SkippedVariable(reason=...)` vs flaga
   `.skipped` na `BucketTable`? **Decyzja odłożona** do etapu `from_auto`/
   `DatasetReport` (nie blokuje prac nad rdzeniem). Kontekst: sekcja 7, pkt 3.

> Rozstrzygnięte (przeniesione do sekcji 7): styl API strategii (fabryki, pkt 1),
> niemutowalność `BucketTable` (pkt 7), indeks `to_frame` = `bin` (pkt 8).
