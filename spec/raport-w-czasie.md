# Specyfikacja: buckety w czasie — sekcje raportu wzorowane na MDBinom (R)

Cel: odwzorowanie w raporcie Python sekcji „w czasie" z raportu R
(`MDBinom/R/raport.R`, `genRaportBody`; przykład wyników: katalog
`raport_aa/`), z wykresami w układzie **Trellis** (jak lattice w R),
rysowanymi gotową biblioteką — **seaborn** (`FacetGrid`).

Realizuje pkt 2 backlogu (`DistributionOverTime` jako standardowy element
analizy zmiennej). Decyzje uzgodnione z użytkownikiem (2026-07-03):
**pełen zakres** sekcji z R (rozkłady, average target, PIT/TTC),
dyskretyzacja **drzewem** dla zmiennych ciągłych (jak w R), układ Trellis
z gotowej biblioteki.

> Zakres: dokument opisuje docelowy kształt i wymagane zmiany; sygnatury
> poglądowe.

---

## 1. Wzorzec: co robi raport R

Rdzeń danych: `univariate_anal_stats2(x_discr, y, czas, estim)`
([univariate.R:145](../MDBinom/R/univariate.R#L145)) liczy na dyskretyzacji
**drzewem** cztery obiekty:

| R | Zawartość | Odpowiednik Python |
|---|---|---|
| `obs_all_tbl` | liczności bucket × okres (+ wiersz/kolumna TOTAL) | `DistributionOverTime.counts()` (bez TOTAL) |
| `pct_all_tbl` | udziały w obrębie okresu (kolumny sumują się do 1) | `DistributionOverTime.distribution()` |
| `avg_t_tbl` | średni target bucket × okres (+ TOTAL-e) | `DistributionOverTime.avg_target()` |
| `estim` | średnia **przypisanego** `avg_target` bucketu per okres | brak — do dodania (sekcja 3.1) |

Sekcje raportu R budowane z tych danych (nazwy plików PNG jak w `raport_aa/`):

1. **Distribution of buckets** (`* distribution.png`) — lattice
   `barchart(value ~ okres | bucket)`: **panel na bucket**, słupki = udział
   bucketu w kolejnych okresach; pod wykresem tabele `obs_all_tbl`
   („Number of observations") i `pct_all_tbl` („% share at given date").
2. **Average target** — dwa lattice `xyplot(type='b')` z `avg_t_tbl`:
   - `* target by bucket.png`: `value ~ okres | bucket` — **panel na
     bucket**, target w czasie (paski paneli zielone,
     `strip.custom(bg='green')`),
   - `* target over time.png`: `value ~ bucket | okres` — **panel na
     okres**, target po bucketach;
   pod wykresami tabela `avg_t_tbl` („Average target").
3. **PIT/TTC** (`* cycle.png`) — punkty: target **obserwowany** per okres
   (wiersz TOTAL `avg_t_tbl`, czarne) i **estymowany** (`estim`, zielone);
   podpis „Does changes in variable distribution follow changes of target
   over time?" + tabelka obu szeregów.
4. *(Discrimination — GINI w czasie: odpowiednik `gini_over_time` już
   istnieje w Pythonie; ta specyfikacja go nie zmienia. Wariant R liczony
   po wielu próbach (`proby`) — poza zakresem, patrz sekcja 8.)*

Kluczowe detale wzorca:

- **Kolejność bucketów** w panelach i tabelach = kolejność wierszy tabeli
  dyskretyzacji (`ordered(levels=rownames(dyskretyzacja))`), nie
  alfabetyczna.
- `estim` odpowiada na pytanie, czy target „chodzi" za rozkładem zmiennej:
  to prognoza targetu w okresie wynikająca wyłącznie ze zmiany struktury
  bucketów (stały `avg_target` bucketu z całej próby × ruchome udziały).

---

## 2. Decyzje projektowe

### 2.1 Dane: dyskretyzacja drzewem, jeden `DistributionOverTime` na zmienną

Dla zmiennej ciągłej rozkłady w czasie liczone są na binach z **drzewa**
(tabela `discrete` w `VariableAnalysis`) — jak w R; dla
dyskretnej/kategorycznej — na jej poziomach (= `buckets`). ~5 paneli
z drzewa jest czytelne; 50 binów kwantylowych nie byłoby.

Wejściem jest **etykieta bucketu** (`bin`), nie score: w `build` dochodzi
`x_label = buck.assign(df, var, discrete, val="bin")` (odpowiednik
`przypisz2` z `fitted=label`). Dotychczasowe `x` (przypisany `avg_target`)
wchodzi jako `pred` — z niego powstaje `estim`.

### 2.2 Wykresy: seaborn `FacetGrid` (układ Trellis)

Wybór biblioteki: **seaborn** —

- `FacetGrid` to wprost implementacja układu trellis (siatka paneli ze
  wspólnymi osiami, tytuł panelu w pasku nad panelem, zawijanie paneli
  `col_wrap`),
- rysuje na matplotlib i zwraca zwykłą `Figure` (`g.figure`) — istniejący
  `report_html` osadza ją bez żadnych zmian (base64 + `plt.close`),
- lekka zależność (numpy/pandas/matplotlib już są w pakiecie).

Alternatywa rozważona i odrzucona: `plotnine` (facet_wrap jak ggplot —
też trellis, ale dodatkowe zależności i odrębny model obiektów zamiast
`Figure`; niepotrzebne, skoro FacetGrid wystarcza).

Stylizacja paneli „à la lattice", wspólna funkcja pomocnicza: pasek
tytułowy panelu jako ramka z tłem (szare; dla „target by bucket" zielone
jak w R), punkty połączone linią (`marker="o"`), wspólne osie X/Y,
etykiety okresów obrócone. Bez ambicji piksel-w-piksel — chodzi
o rozpoznawalny układ, nie klon.

### 2.3 Ścieżka Spark bez zmian

Wszystkie nowe statystyki są ważonymi sumami/średnimi po (okres × bucket),
więc na pseudo-obserwacjach (spec/raport-spark.md) wychodzą **dokładnie** —
`estim` to ważona średnia `pred` per okres, a `pred` pseudo-obserwacji
niesie `avg_pred` grupy. Zero zmian w `spark.py`; test równoważności
w sekcji 6.

---

## 3. Zmiany w kodzie

### 3.1 `DistributionOverTime` — brakujące akcesory (over_time.py)

Klasa już przyjmuje `czas/var/target/pred/weights` i liczy trzy pivoty.
Dochodzą:

```python
def avg_target_total(self) -> pd.Series:
    """Średni target per okres (wiersz TOTAL z avg_t_tbl w R) —
    sum(w*target)/sum(w) po okresie."""

def estim(self) -> pd.Series | None:
    """Średnia ważona `pred` per okres (odpowiednik `estim` z R);
    None gdy nie podano pred."""

def bucket_order(self) -> list:
    """Kolejność poziomów `var` do paneli/tabel; ustawiana z zewnątrz
    (kolejność wierszy tabeli dyskretyzacji), domyślnie kolejność pivotów."""
```

Kolejność bucketów: konstruktor przyjmuje opcjonalne `var_order: list`
i reindeksuje kolumny pivotów w tej kolejności (odpowiednik
`ordered(levels=rownames(...))`).

Prezentacja tabel do raportu (z TOTAL-ami jak w R):

```python
def counts_frame(self) -> pd.DataFrame        # + wiersz/kolumna TOTAL
def distribution_frame(self) -> pd.DataFrame  # + TOTAL-e
def avg_target_frame(self) -> pd.DataFrame    # + TOTAL-e
```

(model bez TOTAL zostaje w `counts()/distribution()/avg_target()` —
zasada jak w `BucketTable`: TOTAL tylko w warstwie prezentacji).

### 3.2 Nowy moduł `src/buckets/trellis.py` — wykresy

Funkcje przyjmują `DistributionOverTime`, zwracają `matplotlib.Figure`:

```python
def plot_distribution(dot, title) -> Figure
    # FacetGrid: panel na bucket, barplot udziału per okres
    # (odpowiednik lattice barchart value ~ okres | bucket)

def plot_avg_target_by_bucket(dot, title) -> Figure
    # panel na bucket, avg_target w czasie, type='b'; paski zielone

def plot_avg_target_by_period(dot, title) -> Figure
    # panel na okres, avg_target po bucketach, type='b'
    # (odpowiednik "* target over time.png")

def plot_pit_ttc(dot, title) -> Figure
    # jeden panel: avg_target_total() (czarne) + estim() (zielone)
```

Wspólny helper `_facet(...)` konfiguruje FacetGrid (col_wrap, wspólne osie,
paski tytułowe, rotacja etykiet X). Liczba paneli = liczba bucketów
(lub okresów) — `col_wrap` dobierane tak, by układ był zbliżony do
lattice (≈ kwadratowa siatka).

### 3.3 Wpięcie w `VariableAnalysis` (report.py)

Nowe pola (sztywne — do czasu wdrożenia otwartego modelu z pkt. 1 backlogu):

```python
dist_over_time: DistributionOverTime | None   # None gdy brak time_col
fig_distribution: Figure | None
fig_target_by_bucket: Figure | None
fig_target_by_period: Figure | None
fig_pit_ttc: Figure | None
```

W `build`, gdy `types.time_col is not None` i zmienna nie jest pominięta:

```python
x_label = buck.assign(df, var=variable, buckets=discrete, val="bin")
dot = DistributionOverTime(
    time_series, x_label, df[types.target], pred=x,   # x = przypisany avg_target
    weights=weights,
    var_order=discrete.loc[discrete["bin"] != "TOTAL", "bin"].tolist(),
)
```

`to_report_payload()` dokłada elementy w kolejności sekcji raportu R:

1. gini + gini_over_time + fig_gini_over_time *(jak dziś — Discrimination)*
2. fig_pit_ttc + tabela obserwowany/estymowany target *(PIT/TTC)*
3. discrete + fig_buckets *(jak dziś — Buckets)*
4. fig_distribution + counts_frame + distribution_frame *(Distribution)*
5. fig_target_by_bucket + fig_target_by_period + avg_target_frame
   *(Average target)*

`report_html` nie wymaga zmian (payload to nadal DataFrame'y i Figury);
opcjonalnie później: tytuły sekcji zamiast generycznych „Statystyki"/
„Wykres" — patrz sekcja 8.

### 3.4 Zależności

`seaborn>=0.13` dochodzi do `dependencies` w `pyproject.toml`.

---

## 4. Mapowanie R → Python (podsumowanie)

| Artefakt R (raport_aa) | Python |
|---|---|
| `* distribution.png` | `trellis.plot_distribution` |
| „Number of observations" | `dot.counts_frame()` |
| „% share at given date" | `dot.distribution_frame()` |
| `* target by bucket.png` | `trellis.plot_avg_target_by_bucket` |
| `* target over time.png` | `trellis.plot_avg_target_by_period` |
| „Average target" (tabela) | `dot.avg_target_frame()` |
| `* cycle.png` (PIT/TTC) | `trellis.plot_pit_ttc` |
| tabelka Observed/Estimated | ramka z `avg_target_total()` + `estim()` |
| `* discrimination.png` | istniejące `plot_gini_over_time` (bez zmian) |
| `* tree.png` | istniejące `fig_buckets` (bez zmian) |

---

## 5. Przypadki brzegowe

- **Brak `time_col`** — nowe pola `None`, payload jak dotychczas (dziś już
  tak działa `gini_over_time`).
- **Zmienna pominięta** (za dużo poziomów) — sekcje w czasie pomijane.
- **Bin `<NA>`** — osobny panel, jak każdy inny bucket (w R braki były
  substytuowane; u nas `<NA>` jest jawnym bucketem — świadoma różnica).
- **Okresy bez obserwacji bucketu** — udział 0 (słupek zerowy),
  `avg_target` brak punktu (NaN nie jest rysowany przez matplotlib).
- **Dużo okresów** — etykiety X przerzedzać/obracać (helper `_facet`);
  w R oś czasu bywała nieczytelna (zlane etykiety na PNG w `raport_aa`) —
  tu celujemy lepiej.
- **Jeden bucket** (degeneracja) — FacetGrid z jednym panelem; nie rysować
  `plot_avg_target_by_period`? Rysować — jeden punkt na panel to nadal
  poprawna informacja.

---

## 6. Testy

1. **Akcesory `DistributionOverTime`**: `avg_target_total`/`estim` vs
   wartości policzone ręcznie na małej ramce; `var_order` ustawia kolejność
   kolumn pivotów; `*_frame()` — TOTAL-e zgodne z definicją R
   (kolumna TOTAL = agregat po całym czasie, wiersz TOTAL = po wszystkich
   bucketach).
2. **Równoważność wag/pseudo-obserwacji**: nowe akcesory na danych ważonych
   == na zreplikowanych; na pseudo-obserwacjach z kanonicznego agregatu ==
   na surowych (rozszerzenie `tests/test_weights.py`
   i `tests/test_pseudo_obs.py`).
3. **Wykresy (smoke)**: funkcje `trellis.*` zwracają `Figure`; liczba
   paneli = liczba bucketów/okresów; wywołanie bez `pred` → `plot_pit_ttc`
   rysuje tylko szereg obserwowany.
4. **Raport end-to-end**: `DatasetReport.to_html()` z `time_col` zawiera
   nowe sekcje; bez `time_col` — raport identyczny jak przed zmianą
   (regresja istniejących referencji).
5. **Spark**: rozszerzenie testu równoważności analiz w
   `tests/test_spark.py` o nowe pola (`dist_over_time` pivoty równe
   ścieżce pandas).

---

## 7. Etapy wdrożenia

1. Rozszerzenia `DistributionOverTime` (3.1) + testy (6.1, 6.2).
2. Moduł `trellis.py` (3.2) + seaborn w zależnościach + testy smoke (6.3).
3. Wpięcie w `VariableAnalysis`/payload (3.3) + testy raportu (6.4, 6.5);
   weryfikacja wizualna na `test_raport_default.py`
   i `raport_spark_default.py` — porównanie z PNG z `raport_aa/`.

---

## 8. Poza zakresem (kandydaci do backlogu)

- **Discrimination po próbach** (`proby` w `univariate_anal_stats3` — GINI
  osobno dla train/test/OOT): wymaga pojęcia podziału na próby, którego
  pakiet nie ma.
- **Tytuły sekcji w report_html** (dziś generyczne „Statystyki"/„Wykres") —
  naturalnie rozwiąże to otwarty model `ReportElement` (backlog pkt 1,
  pole `title`).
- **Wygładzenie locfit/span** z R (`estim` w R bywało wygładzane) — nie
  przenosimy.
