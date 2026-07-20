# Trwała kolekcja analiz — `DatasetReport` jako obiekt roboczy, nie przelot

Status: propozycja (realizuje backlog pkt 1, domyka pkt 2). Cel: obiekt raportu
**posiada** kolekcję statystyk cech — można się do nich odnosić, używać ich
i modyfikować je w dowolnym momencie — a wygenerowanie HTML jest tylko jedną
z operacji na tej kolekcji. Wzorzec sprawdzony w MDBinom (R), który realizował
dokładnie ten przepływ.

---

## 1. Motywacja: dziś raport jest przelotowy

Stan obecny ([report.py](../src/buckets/report.py)):

- `DatasetReport.to_html()` → `to_payload()` → `analyses()` → `buckets()` —
  żadna z tych metod nie zapisuje wyniku na `self`. Słownik
  `{zmienna: VariableAnalysis}` powstaje jako wartość tymczasowa, jest
  spłaszczany do pozycyjnych list i po zwróceniu HTML znika.
- Każde kolejne wywołanie liczy wszystko od zera (buckety, drzewa, gini,
  wykresy) — brak cache'u; na ścieżce Spark oznacza to ponowne agregacje.
- Render jest destrukcyjny: `report_html` po osadzeniu wykresu robi
  `plt.close(element)` ([report_html.py:54](../src/buckets/report_html.py#L54)) —
  nawet ręcznie zatrzymany wynik `analyses()` ma po renderze pozamykane figury.

Nie da się więc: obejrzeć statystyk po wygenerowaniu raportu, poprawić jednej
zmiennej i wyrenderować ponownie, ani użyć policzonych bucketów do czegokolwiek
innego (scoring, dokumentacja) bez ponownego liczenia.

---

## 2. Wzorzec z MDBinom — co już działało i jak

MDBinom ([MDBinom/R](../MDBinom/R)) realizował ideę „kolekcja najpierw,
raport później" wprost:

**Kolekcja to nazwana lista czystych danych.** `univariate_loop`
([univariate.R:393](../MDBinom/R/univariate.R#L393)) zwraca listę
`wyniki[[zmienna]] = list(dyskretyzacja, rozklady, dyskryminacja)`:

| element MDBinom | zawartość | odpowiednik w buckets |
|---|---|---|
| `dyskretyzacja` | data.frame bucketów (br, woe, logit, granice) | `BucketTable` / `discrete` |
| `rozklady` | liczności, rozkład %, średni target po czasie, `estim` | `DistributionOverTime` |
| `dyskryminacja` | AR/GINI total i po okresach (i po próbach) | `gini`, `gini_over_time` |

Kluczowe: **żadnych wykresów w kolekcji** — wyłącznie tabele. Lista żyła
w workspace użytkownika (via `.RData` przeżywała sesję), można było ją
indeksować, podmieniać elementy (ręczna redyskretyzacja przez
`breaks`/`mapping`/`interactive` i wpisanie z powrotem), liczyć na niej.

**Raport to osobna, powtarzalna funkcja.** `genRaport(wyniki, dir, kolejnosc)`
([raport.R:19](../MDBinom/R/raport.R#L19)) rysuje wszystkie wykresy
**w momencie renderowania, z danych kolekcji** (lattice/barplot na
`dyskryminacja`, `rozklady$pct_all_tbl`, `avg_t_tbl`…). Dzięki temu raport
można generować dowolnie wiele razy, a parametr `kolejnosc` steruje kolejnością
sekcji; menu (sortowanie po GINI i alfabetyczne) też liczy się z kolekcji
([raport.R:161](../MDBinom/R/raport.R#L161)).

**Kolekcja była centralnym artefaktem całego warsztatu**, nie tylko raportu:

- `przypisz_z_listy` / `przypisz_woe_z_listy`
  ([other.R:113](../MDBinom/R/other.R#L113)) — scoring: przypisanie
  fitted/WOE do danych wprost ze zbudowanej listy bucketów,
- `univariate_stats_new_data` ([univariate.R:102](../MDBinom/R/univariate.R#L102)) —
  monitoring: nałożenie zapisanej dyskretyzacji na **nowe dane** i policzenie
  tych samych statystyk (porównanie `br` vs `br_orig`),
- `makeCoarseClassingTables` ([other.R:66](../MDBinom/R/other.R#L66)) —
  tabele do dokumentacji (GB odds, GB index, IV) liczone z kolekcji,
- `genSQL` ([gen_SQL.R:20](../MDBinom/R/gen_SQL.R#L20)) — kod SQL wdrożeniowy
  generowany z kolekcji + modelu.

**Czego nie kopiować:**

- goła lista bez schematu — błędy sygnalizowane stringiem
  (`"Too many categorical levels"`) i sprawdzane przez `typeof(...)=="character"`
  w wielu miejscach (raport, przypisywanie); u nas ma to być jawne pole
  (`skipped`) na otypowanym obiekcie,
- raport pisze dziesiątki plików PNG + ramki HTML do katalogu
  (patrz `raport_aa/`) — ścieżka Python z base64 w jednym pliku jest lepsza,
- stan globalny (`numeric_var_treatment.params`) jako źródło progów.

---

## 3. Zasada projektowa: statystyki to dane, wykresy to renderowanie

Lekcja z MDBinom rozstrzygająca dzisiejszy problem `plt.close`:

> `VariableAnalysis` przechowuje **wyłącznie dane** (ramki, `BucketTable`,
> `DistributionOverTime`). Figury matplotlib **nie są stanem** — powstają
> w momencie renderowania z przechowywanych danych i są zamykane po osadzeniu.

Skutki:

- render jest powtarzalny i nie mutuje kolekcji (`to_html()` można wołać
  wielokrotnie),
- kolekcja jest picklowalna (figury matplotlib są kłopotliwe w serializacji),
- pamięć: brak setek otwartych figur przy szerokich zbiorach,
- funkcje rysujące (`buck.plot`, `trellis.plot_*`, `buck.plot_gini_over_time`)
  już dziś przyjmują dane i zwracają figurę — zmienia się tylko **moment**
  ich wołania (render zamiast `build`).

Ta sama zasada wymaga doprowadzenia `DistributionOverTime` do modelu
`BucketTable` (rdzeń-agregat zamiast surowych obserwacji, wykresy jako
metody `plot_*`) — osobna specyfikacja:
[2026-07-07-dist-over-time.md](2026-07-07-dist-over-time.md).

---

## 4. Docelowe API

```python
report = DatasetReport(df, types)        # nic nie liczy

report.build()                           # policz analizy (jawnie); albo
report["age"]                            # leniwie przy pierwszym dostępie

# kolekcja: dict-podobny dostęp do VariableAnalysis
report["age"].buckets                    # tabela bucketów
report["age"].gini                       # ramka z GINI
report["age"].dist_over_time             # DistributionOverTime
list(report)                             # nazwy zmiennych
"age" in report

# modyfikacja
report.rebuild("dochod", buckets=moje_buckety)   # przelicz jedną zmienną
report["dochod_log"] = VariableAnalysis.build(...)  # dodaj/podmień ręcznie
del report["zonk_s_1__01m"]                         # usuń z raportu
report["age"].add_text("Zmienna po korekcie z 2026-06.", title="Uwagi")

# render — operacja na kolekcji, powtarzalna, bez efektów ubocznych
html = report.to_html(order="gini")      # "gini" (domyślnie) | "alpha" | lista nazw
report.save_html("raport.html")

# trwałość między sesjami (odpowiednik .RData)
report.save("analizy.pkl")
report2 = DatasetReport.load("analizy.pkl")
```

Zasady:

- `build()` liczy raz i zapisuje kolekcję na obiekcie; kolejne `to_html()` /
  dostępy niczego nie przeliczają,
- `DatasetReport` zachowuje referencję do źródła (`PandasSource`/`SparkSource`),
  żeby `rebuild(var, ...)` mógł przeliczyć pojedynczą zmienną z nowymi
  parametrami (odpowiednik ręcznej redyskretyzacji w MDBinom),
- `to_html()` nie modyfikuje kolekcji i nie zamyka figur, których nie stworzył,
- zmienna pominięta (za dużo poziomów) to normalny `VariableAnalysis`
  z `skipped=True` i elementem tekstowym z powodem — żadnych stringów-sentyneli.

---

## 5. Model elementów (spójnie z backlog pkt 1)

`VariableAnalysis` staje się uporządkowaną kolekcją `ReportElement`
(backlog pkt 1) — z jedną korektą wynikającą z sekcji 3: standardowe wykresy
wchodzą do kolekcji jako **fabryki**, nie gotowe figury.

```python
class ReportElement(ABC):
    key: str | None                      # nazwany dostęp: va["gini"]
    title: str | None
    def render_html(self) -> str: ...

class TableElement(ReportElement): ...   # pd.DataFrame
class TextElement(ReportElement): ...    # tekst / Markdown

class FigureElement(ReportElement):
    # factory: Callable[[], plt.Figure] — figura powstaje w render_html,
    # jest osadzana jako base64 i zamykana; element trzyma dane pośrednio
    # (przez domknięcie na ramkach VariableAnalysis)
    ...
```

- `build()` dodaje standardowy zestaw z kluczami: `"gini"`, `"gini_over_time"`,
  `"pit_ttc"`, `"buckets"`, `"distribution"`, `"avg_target"` — struktura
  zostaje (nazwany dostęp `va["buckets"]`, skróty właściwości),
- `add_table` / `add_figure` / `add_text` dają otwartość; `add_figure` przyjmuje
  też gotową `Figure` (element ad hoc od użytkownika) — wtedy render **nie
  zamyka** cudzej figury,
- walidacja typu w konstruktorze elementu — błąd przy `add`, nie w
  `report_html` (przejmuje rolę `_validate_payload_element`),
- `report_html.generate_report` iteruje `va.elements` i woła
  `el.render_html()`; pozycyjny kontrakt listy oraz `to_report_payload()`
  znikają; sortowanie po GINI przechodzi z `payload[0].iloc[0, 1]` na jawne
  `va.gini_value` (float, także dla `skipped`).

---

## 6. Etapy wdrożenia

**Etap 1 — trwałość bez zmiany kontraktu render:**
`DatasetReport` cache'uje `analyses()` na `self`; `VariableAnalysis` przestaje
trzymać pola `fig_*` (figury liczone w `to_report_payload()`, świeże przy
każdym renderze — `report_html` może je nadal zamykać); dochodzą `__getitem__`
/ `__setitem__` / `__delitem__` / `__iter__`, `rebuild`, `save`/`load`,
`save_html`. Testy raportów (`test_raport_default*.py`) przechodzą bez zmian
referencji.

**Etap 2 — otwarty model elementów (backlog pkt 1):**
`ReportElement` + `elements` jako jedyne źródło prawdy; `report_html` iteruje
elementy; usunięcie `to_report_payload()` i pozycyjnego kontraktu.

**Etap 3 — operacje MDBinomowe na kolekcji (nowe pozycje backlogu):**
`report.assign(df, val="woe")` (odpowiednik `przypisz_woe_z_listy`),
statystyki na nowych danych z zapisanych bucketów (odpowiednik
`univariate_stats_new_data`), tabele coarse-classing do dokumentacji,
generowanie SQL. Poza zakresem tej specyfikacji — tu tylko kierunek: to
kolekcja, nie HTML, jest artefaktem, na którym te operacje mają siedzieć.

---

## 7. Kwestie rozstrzygnięte

1. **Leniwe czy jawne liczenie?** Oba: `build()` jawnie, a dostęp do
   niezbudowanej kolekcji buduje ją przy pierwszym użyciu. Spark: preferować
   jawne `build()` (koszt widoczny dla użytkownika).
2. **Czy trzymać figury „na wszelki wypadek"?** Nie — sekcja 3. Figura jest
   funkcją danych; kto chce ją obejrzeć poza raportem, woła
   `va.plot_buckets()` (cienkie metody-fabryki na `VariableAnalysis`).
3. **Serializacja:** pickle całego `DatasetReport` bez źródła danych
   (`source` wyłączony z `__getstate__` — po `load` dostępny jest render
   i odczyt, `rebuild` wymaga ponownego podpięcia danych).
