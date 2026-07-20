# `DistributionOverTime` na modelu `BucketTable` — rdzeń-agregat i metody `plot`

Status: propozycja. Cel: `DistributionOverTime` działa na tej samej zasadzie
co `BucketTable` — obiekt **posiada wyliczone statystyki** (kompaktowy
agregat), nie surowe obserwacje, a wykresy są jego **metodami**. Warunek
lekkiej, picklowalnej kolekcji analiz
z [2026-07-07-raport-kolekcja.md](2026-07-07-raport-kolekcja.md).

---

## 1. Stan obecny i diagnoza

[over_time.py](../src/buckets/over_time.py) częściowo realizuje ideę
„obiekt ze statystykami", ale z trzema odstępstwami:

1. **Konstruktor zatrzymuje na stałe surowe obserwacje** — `self._df`
   z kolumnami czas/var/target/weights/pred, O(n_obs) pamięci na każdą
   zmienną raportu. `BucketTable` przechowuje wyłącznie agregat per bin;
   tu analogii brak.
2. **Cache jest niepełny.** Cztery pivoty (`counts`, `distribution`,
   `avg_target`, `avg_pred`) są liczone leniwie i cache'owane — to było
   zapisane wprost ([2026-05-31-buck-refaktor-klasy.md](2026-05-31-buck-refaktor-klasy.md),
   sekcja 4.4: „pivoty liczone leniwie/cache'owane"). Ale `avg_target_total()`
   i `estim()` liczą się z surowych danych **przy każdym wywołaniu**
   (`_weighted_mean_by_time`), a metody prezentacyjne `*_frame()` również
   przeliczają się za każdym razem.
3. **Wykresy są wolnymi funkcjami** w [trellis.py](../src/buckets/trellis.py)
   (`plot_distribution(dot, ...)`, …) — tak zdecydowano
   w [2026-07-03-raport-w-czasie.md](2026-07-03-raport-w-czasie.md), sekcja 3.2,
   **wbrew** wcześniejszej decyzji 6
   z [2026-05-31-buck-refaktor-klasy.md](2026-05-31-buck-refaktor-klasy.md)
   („wykresy jako metody obiektów, które gwarantują układ kolumn"),
   zrealizowanej tylko w `BucketTable.plot()`.

Skutek: obiekt jest ciężki (nie nadaje się do trzymania w trwałej kolekcji
ani do pickle), część statystyk nie jest „posiadana" tylko przeliczana,
a API wykresów jest niespójne z `BucketTable`.

---

## 2. Weryfikacja w MDBinom

MDBinom rozstrzygał to jednoznacznie po stronie „gotowych statystyk":

- `univariate_anal_stats2` ([univariate.R:145](../MDBinom/R/univariate.R#L145))
  liczyło **zachłannie** komplet tabel i tylko one wchodziły do elementu
  kolekcji `wyniki[[zmienna]]$rozklady`:
  `obs_all_tbl` (liczności bucket × okres), `pct_all_tbl` (udziały),
  `avg_t_tbl` (średni target), `estim` (średnia predykcja per okres).
  Surowe wektory (`x_discr`, `y`, `czas`) **nie były przechowywane**.
- `genRaport` ([raport.R:114-152](../MDBinom/R/raport.R#L114)) rysował
  wszystkie wykresy „w czasie" z tych zapisanych tabel
  (`reshape::melt(wynik$rozklady$pct_all_tbl)` → `lattice::barchart`, itd.) —
  wykres był funkcją zapisanych statystyk, nigdy surowych danych.

Czyli: element kolekcji = wyliczone statystyki; render = funkcja statystyk.
Dokładnie ta zasada ma obowiązywać w `DistributionOverTime`.

---

## 3. Projekt docelowy

Model jak w `BucketTable`: **rdzeń = agregat** liczony raz w konstruktorze,
prezentacja osobno, obiekt niemutowalny.

```python
class DistributionOverTime:
    # rdzeń: JEDEN agregat czas × var, liczony w konstruktorze;
    # surowe obserwacje nie są zatrzymywane
    _core: pd.DataFrame     # indeks: (czas, var); kolumny: sum_w,
                            # sum_w_target, sum_w_pred (gdy podano pred)

    # akcesory — wszystko wyprowadzalne z rdzenia (tanio, bez cache'u
    # na surowych danych); sygnatury bez zmian:
    def counts(self) -> pd.DataFrame
    def distribution(self) -> pd.DataFrame
    def avg_target(self) -> pd.DataFrame
    def avg_pred(self) -> pd.DataFrame | None
    def avg_target_total(self) -> pd.Series     # z rdzenia: sum po var
    def estim(self) -> pd.Series | None         # jw.
    def bucket_order(self) -> list

    # prezentacja (TOTAL-e tylko tutaj — jak dziś):
    def counts_frame(self) -> pd.DataFrame
    def distribution_frame(self) -> pd.DataFrame
    def avg_target_frame(self) -> pd.DataFrame

    # wykresy jako metody (spójnie z BucketTable.plot):
    def plot_distribution(self, title=None) -> Figure
    def plot_avg_target_by_bucket(self, title=None) -> Figure
    def plot_avg_target_by_period(self, title=None) -> Figure
    def plot_pit_ttc(self, title=None) -> Figure
```

Uzasadnienie rdzenia: trzy sumy ważone w przecięciu czas × var wystarczają
na **wszystkie** dzisiejsze wyniki —
`counts = pivot(sum_w)`, `distribution = counts / suma wiersza`,
`avg_target = pivot(sum_w_target) / pivot(sum_w)`,
`avg_target_total = Σ_var sum_w_target / Σ_var sum_w`, analogicznie `estim`.
Pomijanie par z brakiem wartości (dzisiejsza semantyka
`_weighted_mean_by_time` i `avg_pred`) realizowane przy budowie rdzenia:
osobne mianowniki dla `target` i `pred` (kolumny `sum_w_target_obs`,
`sum_w_pred_obs`), jeśli braki w `pred` faktycznie występują.

Zasady:

- **konstruktor bez zmian** — nadal przyjmuje surowe serie
  (`czas, var, target, pred, weights, var_order`); zmienia się wyłącznie to,
  co obiekt zatrzymuje po policzeniu agregatu,
- **niemutowalność** — `_core` jest read-only; akcesory zwracają nowe ramki,
- **wykresy**: metody `plot_*` delegują do funkcji w `trellis.py`
  (`trellis` zostaje jako moduł implementacyjny — jak `buck.plot` pod
  `BucketTable.plot`); wołający sam odpowiada za zamknięcie figury,
- pamięć: O(okresy × buckety) zamiast O(n_obs).

---

## 4. Wpływ na resztę kodu

- [report.py](../src/buckets/report.py): `VariableAnalysis.build` — bez zmian
  w wywołaniu; `to_report_payload` woła `self.dist_over_time.plot_*()`
  zamiast `trellis.plot_*(dot, ...)`. `VariableAnalysis` z lekkim
  `dist_over_time` staje się picklowalny
  ([2026-07-07-raport-kolekcja.md](2026-07-07-raport-kolekcja.md), sekcja 7 pkt 3).
- [trellis.py](../src/buckets/trellis.py): funkcje zostają (implementacja),
  przestają być publicznym API — analogicznie do `buck.plot`.
- Testy: `tests/test_over_time.py` i `tests/test_report_over_time.py`
  przechodzą bez zmiany referencji (sygnatury i wyniki akcesorów identyczne);
  `tests/test_trellis.py` — wywołania przez metody.
- Ścieżka Spark: bez zmian — pseudo-obserwacje wchodzą do konstruktora jak
  dotąd; zysk pamięciowy dotyczy obu ścieżek.

---

## 5. Etapy

1. Rdzeń-agregat w konstruktorze + akcesory z rdzenia; usunięcie `self._df`
   i `self._cache`. Testy over_time bez zmiany referencji.
2. Metody `plot_*` + przełączenie `report.py` i testów trellis na metody.
3. (Razem z etapem 1 kolekcji z
   [2026-07-07-raport-kolekcja.md](2026-07-07-raport-kolekcja.md)) —
   `VariableAnalysis` przestaje trzymać figury; render korzysta z metod
   `plot_*` obiektów.
