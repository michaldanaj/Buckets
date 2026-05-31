# Backlog — pomysły do późniejszego wdrożenia

Rzeczy świadomie odłożone na później: w pierwszej implementacji idziemy prościej,
ale chcemy mieć zapisany kierunek docelowy, żeby nie zamknąć sobie drogi.

---

## 1. Otwarty model raportu: `VariableAnalysis` jako kolekcja `ReportElement`

### Wymaganie

Pierwotną ideą generowanego raportu była **dowolność zawartości** — do raportu
dla danej zmiennej można dodać dowolną analizę:
- tabelę (`pd.DataFrame`),
- wykres (`matplotlib.Figure`),
- docelowo także tekst / komentarz (Markdown).

Projekt z [buck-refaktor-klasy.md](buck-refaktor-klasy.md) (sekcja 4.5) ujmuje
analizę zmiennej w klasę `VariableAnalysis` ze **sztywnymi polami**
(`gini`, `fig_buckets`, `gini_over_time`, …). Daje to spójny „zestaw statystyk
na zmienną", ale **ogranicza dowolność** — nie da się dorzucić ad hoc dodatkowego
elementu bez zmiany klasy.

Cel: pogodzić **strukturę** (znany, otypowany zestaw standardowych statystyk)
z **otwartością** (możliwość dołożenia dowolnego elementu, w tym tekstu).

### Propozycja rozwiązania

`VariableAnalysis` przestaje trzymać sztywne pola, a staje się **uporządkowaną
kolekcją elementów raportu**. Standardowy zestaw statystyk jest dodawany domyślnie
przy budowie (nie tracimy „zestawu na zmienną"), a `.add(...)` pozwala dorzucić
cokolwiek (odzyskujemy elastyczność).

Wspólny typ elementu, który zna swój rodzaj i potrafi się wyrenderować:

```python
class ReportElement(ABC):            # wspólny element raportu
    key: str | None                  # nazwany dostęp, np. "gini", "buckets"
    title: str | None
    def render_html(self) -> str: ...

class TableElement(ReportElement): ...   # opakowuje pd.DataFrame
class FigureElement(ReportElement): ...  # opakowuje matplotlib.Figure
class TextElement(ReportElement): ...    # tekst / Markdown — docelowy element
```

Każda podklasa waliduje swój wkład w konstruktorze — błąd typu wychodzi przy
`add`, a nie cicho dopiero w `report_html` (realizuje TODO z
[buck.py:803](../src/buckets/buck.py#L803)).

```python
class VariableAnalysis:
    name: str
    elements: list[ReportElement]        # jedyne źródło prawdy; iterowane przez raport

    @classmethod
    def build(cls, df, var, types) -> "VariableAnalysis":
        va = cls(name=var)
        va.add_table(gini, key="gini", title="GINI")
        va.add_figure(fig_buckets, key="buckets")
        ...                              # standardowy zestaw
        return va

    # otwartość:
    def add(self, element: ReportElement): ...
    def add_table(self, df, *, key=None, title=None): ...
    def add_figure(self, fig, *, key=None, title=None): ...
    def add_text(self, text, *, key=None, title=None): ...

    # struktura — nazwany dostęp do dobrze znanych elementów:
    def __getitem__(self, key) -> ReportElement: ...   # va["gini"]
    @property
    def buckets(self) -> BucketTable: ...              # cienki skrót do standardowego
```

Te same elementy są dostępne dwojako: jako lista do iteracji przy renderowaniu
(`for el in va.elements`) oraz jako nazwane skróty dla kodu, który wie, czego
szuka (`va["gini"]`).

Po stronie [report_html.py](../src/buckets/report_html.py): `generate_report`
przestaje zależeć od pozycyjnej listy (`[gini, gini_over_time, …]`) i iteruje
`va.elements`, wołając `el.render_html()`.

### Dlaczego można to odłożyć (etap pośredni jest bezpieczny)

Model otwarty jest **nadzbiorem** wersji ze sztywnymi polami — `build()` zamiast
przypisywać pola będzie wołać `add_*(...)`, a logika liczenia statystyk się nie
zmienia. Warunki taniej migracji:

1. **Konsumpcja payloadu tylko w jednym miejscu.** Strukturę znają wyłącznie
   `VariableAnalysis.build()` i `report_html` (przez adapter `to_report_payload()`).
   Nie rozsiewać dostępu `va.gini`/`va.fig_buckets` po kodzie — inaczej przejście
   na `va["gini"]` dotknie wszystkich takich miejsc.
2. **Walidację typu dokładanego elementu wprowadzić od razu** — nawet bez
   `ReportElement`, `build()` może sprawdzać, że element to `DataFrame`/`Figure`.
   Tani krok, który nie marnuje się po refaktorze.

---

## 2. `DistributionOverTime` jako standardowy element analizy zmiennej

### Wymaganie

`DistributionOverTime` (rozkład/target/pred zmiennej w czasie — dawne
`bckt_stats_over_time`, sekcja 4.4 w [buck-refaktor-klasy.md](buck-refaktor-klasy.md))
jest dziś projektowane jako klasa stojąca **obok** analizy zmiennej. Docelowo
rozkład w czasie powinien wchodzić do **domyślnego zestawu statystyk na zmienną** —
tak jak gini, buckety i wykresy — gdy zdefiniowana jest główna kolumna czasowa
(`Role.MAIN_TIME_COL` w `ColumnTypes`).

### Propozycja rozwiązania

Wpięcie w `VariableAnalysis.build()`: jeśli `types.time_col is not None`, `build()`
liczy `DistributionOverTime` i dodaje jej wyniki jako standardowe elementy raportu
(tabela rozkładu + wykres), analogicznie do dzisiejszego „gini over time"
w [buck.py:786-798](../src/buckets/buck.py#L786-L798).

Spina się to wprost z pkt. 1: gdy `VariableAnalysis` jest otwartą kolekcją
`ReportElement`, rozkład w czasie to po prostu kolejne elementy dokładane przez
`build()` (`add_table(dist.distribution())`, `add_figure(...)`). Do czasu wdrożenia
otwartego modelu można go trzymać jako kolejne sztywne pole.

Zależność: warto wdrożyć **po** lub **razem z** pkt. 1, żeby nie mnożyć sztywnych
pól, które i tak trafią do kolekcji elementów.

---

## 3. ~~Wydzielić binowanie po jawnych granicach do osobnej fabryki~~ (przeniesione do specyfikacji)

Rozdzielenie binowania kwantylowego od binowania po jawnych granicach zostało
wpisane wprost do projektu — `from_quantiles` (tylko `n_bins: int`) i osobne
`from_bins` (`bins: list[float]`). Opis: [buck-refaktor-klasy.md](buck-refaktor-klasy.md),
sekcja 4.2. Pozycja zostawiona jako ślad decyzji.

---

## 4. Indeks `to_frame()` na `RangeIndex` zamiast `bin`

Na pierwszym etapie `to_frame()` zwraca DataFrame z indeksem = `bin` (zgodność
z dzisiejszymi referencjami testowymi — decyzja w
[buck-refaktor-klasy.md](buck-refaktor-klasy.md), sekcja 7 pkt 8). Czystsze
docelowo: neutralny `RangeIndex` + `bin` jako zwykła kolumna. Wymaga przepisania
referencji testowych, więc odłożone na po ustabilizowaniu rdzenia.

---

## 5. Sprzątanie ostrzeżeń (warnings) przy testach

Istniejące wcześniej ostrzeżenia, ujawnione przy pełnym `pytest` (spoza zakresu
refaktoru BucketTable, dlatego odłożone):

1. **`DataFrameGroupBy.apply operated on the grouping columns` (DeprecationWarning)**
   w [statitics.py:34](../src/buckets/statitics.py#L34) — `gini(..., by=...)` używa
   `df.groupby("by").apply(...)`. Poprawka: przekazać `include_groups=False` albo
   wybrać kolumny po `groupby`, żeby nie operować na kolumnie grupującej.

2. **`More than 20 figures have been opened` (RuntimeWarning)**
   z [buck.py](../src/buckets/buck.py) (`plot`) — figury matplotlib nie są zamykane
   po wygenerowaniu. Przy raporcie z wieloma zmiennymi rośnie zużycie pamięci.
   Poprawka: `plt.close(fig)` po osadzeniu wykresu (np. w `report_html` po zapisaniu
   do base64 — częściowo już jest) albo zwracać figury i domykać je u konsumenta.

---

## 6. Oś X wykresu dla dyskretnej numerycznej: `bin` (string) vs `discrete` (liczby)

`plot` rysuje zmienną dyskretną po kolumnie `bin` (string), bo `median` jest dla
niej całe `NA` (warunek wyboru osi w [buck.py](../src/buckets/buck.py), `plot`).
Kolejność na osi jest poprawna — **numeryczna** (`1,2,…,20`), bo `groupby`
sortuje klucze liczbowo zanim powstanie string (nie leksykalnie). Ale oś jest
**kategoryczna**: punkty są rozmieszczone równomiernie, niezależnie od odległości
wartości.

Skutek: dla wartości z dziurami (np. `1, 2, 5, 100`) odstępy na osi są jednakowe,
a nie proporcjonalne do wartości. Dla zmiennej dyskretnej numerycznej często
chcielibyśmy oś liczbową z proporcjonalnymi odstępami.

Propozycja: w `plot` dla dyskretnej numerycznej używać osi `discrete` (wartości
liczbowe) zamiast `bin` (string) — wtedy odstępy odzwierciedlają faktyczne
wartości. Dla kategorycznej zostaje `bin`. Wymaga rozróżnienia w `plot`, czy
`discrete` jest numeryczne (np. po `kind`/`is_numeric` z `BucketTable`).
