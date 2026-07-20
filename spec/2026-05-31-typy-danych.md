# Specyfikacja: obsługa typów danych w statystykach bucketów

Dokument opisuje problem z typami danych w module [src/buckets/buck.py](../src/buckets/buck.py)
(funkcje `bckt_stats`, `bckt_cut_stats`, `bckt_stats_over_time`) oraz w testach
`tests/test_buck.py`. Bazuje na rozwiązaniach wypracowanych na gałęzi `typy`
(commit `5ffaee8`, analiza w `specs/typy-bledy-analiza.md`) i konfrontuje je
z aktualnym stanem gałęzi `main`.

> Uwaga o gałęziach: `typy` odgałęziła się od `main` w commicie `50dd86b`.
> Na `main` powstały później niezależne zmiany (Gini w czasie, obsługa braków
> danych, refaktor `bckt_stats_over_time`). Gałąź `typy` to alternatywna eksploracja
> samego problemu typów — nie jest prostym następcą `main` i nie da się jej
> bezpośrednio zmergować. Ten dokument przenosi z niej **wnioski i reguły**,
> a nie commity.

---

## 1. Problem

Statystyki bucketów są budowane przez `groupby` + agregację, a następnie
doklejany jest wiersz `TOTAL` i (dla zmiennych ciągłych) wiersz braków `<NA>`.
Pandas miesza tu trzy światy typów:

- **typy numpy** (`int64`, `float64`, `object`),
- **rozszerzone typy pandas** (`Int64`, `Float64`, `string`) — jedyne, które
  potrafią trzymać `pd.NA` w kolumnie całkowitoliczbowej,
- **indeks** ramki, który rządzi się własnymi prawami.

Źródłem chaosu jest to, że **każda operacja kompozycji tabeli zmienia typy
w sposób trudny do przewidzenia**, a testy porównują wynik przez
`assert_frame_equal`, które jest czułe na dtype (łącznie z dtype indeksu).

### 1.1. Konkretne tryby awarii (zdiagnozowane na `typy`)

1. **Indeks: `string` vs `object`.**
   Kod robił `wyn.index = wyn.index.astype("string")`, podczas gdy referencje
   testowe powstają przez `DataFrame.from_records(...).convert_dtypes()`.
   `convert_dtypes()` **nie dotyka indeksu**, więc referencja ma indeks
   `object`, a kod zwracał `string[python]` → 6 niezgodnych asercji.

2. **Wiersz `TOTAL` psuje typ kolumny `discrete`.**
   Po `pd.concat` z wierszem totala kolumna `discrete` dostawała `NaN`
   niezależnie od typu wejścia. Testy oczekują rozróżnienia:
   - wejście kategoryczne → `discrete = "TOTAL"` (string),
   - wejście numeryczne → `discrete = pd.NA`,
   - wejście ciągłe (`bckt_cut_stats`) → `discrete = pd.NA` we wszystkich
     wierszach (informację o przedziale niosą kolumny `od`/`srodek`/`do`).

3. **`Int64` + `NaN` → stringifikacja indeksu jako `"8"` zamiast `"8.0"`.**
   `groupby` po kolumnie `Int64` z brakami wypycha do indeksu wartość całkowitą
   (`8` → `"8"`), podczas gdy testy oczekiwały binu `"8.0"`. To jednak **nie był
   realny problem**, tylko rozbieżność z oczekiwaniem testu — `"8"` jest poprawną
   etykietą dla zmiennej całkowitoliczbowej. Rozwiązaniem docelowym jest poprawa
   testu (oczekiwać `"8"`/`Int64`), a nie promocja typu (patrz sekcja 2, pkt 3
   i sekcja 3).

4. **`UnboundLocalError: cols_to_Float`.**
   Mapa rzutowań na `Float64` była definiowana tylko w jednej gałęzi `if`,
   a używana zawsze → wyjątek przy `min_info=True`.

5. **Permutacja `<NA>` → … → `TOTAL` bywała wyłączana.**
   Permutacja gwarantująca „braki na początku, TOTAL na końcu, niezależnie
   od `sort_by`” była zakomentowywana, przez co `TOTAL` lądował w środku tabeli.

6. **Helper testowy `df_from_array` nie obsługuje `pd.NA` przy numpy `float64`.**
   Kolumna złożona z samych `pd.NA` wymaga rozszerzonego typu `Float64`;
   z `discr_type="float64"` (numpy) helper się wywracał.

7. **Sprzeczne oczekiwania między testami.**
   Dla tego samego `test_df_2` jeden test oczekiwał `discrete` jako `Int64`,
   inny jako numpy `float64`. Część referencji budowana była przez
   `DataFrame.from_dict` (surowe numpy dtypes) zamiast przez
   `from_records → convert_dtypes` → niespójność nie do pogodzenia bez
   ujednolicenia testów.

8. **`bckt_cut_stats` mieszał `pd.NA` z indeksem typu `object`.**
   `wyn.index.isin(["TOTAL", pd.NA])` rzucał `ValueError` na hashowaniu `pd.NA`
   — w indeksie braki są reprezentowane stałą stringową `NA_BIN_NAME = "<NA>"`
   ([buck.py:48](../src/buckets/buck.py#L48)), a nie obiektem `pd.NA`.

### 1.2. Dlaczego to się dzieje

Pierwotny zamysł `convert_dtypes()` ([buck.py:198](../src/buckets/buck.py#L198))
brał się stąd, że kolumny całkowite z brakami muszą stać się `Int64`/`Float64`.
Problem w tym, że **konwersja jest robiona punktowo w różnych miejscach pipeline'u**,
a doklejenie `TOTAL`/`<NA>` po niej znów rozjeżdża typy. Brakuje jednego,
deterministycznego miejsca, w którym zapada decyzja o typie każdej kolumny.

---

## 2. Rozwiązanie wypracowane na gałęzi `typy`

Zasada przewodnia: **przestać „naprawiać” typy `astype`-em na końcu, a zamiast
tego ustalić jawne, powtarzalne reguły dla indeksu i każdej kolumny.**

### 2.1. Reguły dla `bckt_stats`

1. **Indeks pozostaje `object` (stringi).**
   Usunięto `wyn.index = wyn.index.astype("string")`. Po agregacji indeks
   budowany jest listowo:
   `[NA_BIN_NAME if pd.isna(v) else str(v) for v in wyn.index]`.
   Zgodne z referencją (`from_records → convert_dtypes`, które nie tyka indeksu).

2. **Kolumna `discrete` liczona raz, z kopii indeksu po `groupby`.**
   `wyn["discrete"] = wyn.index.copy()` zachowuje naturalny typ grupowania.
   Dla wiersza `TOTAL` ustawiana jawnie: `pd.NA` dla zmiennej numerycznej,
   `"TOTAL"` dla nienumerycznej.

3. **Promocja `Int64` → `Float64` dla zmiennych numerycznych z brakami.**
   ```python
   is_numeric_var = pd.api.types.is_numeric_dtype(var)
   if is_numeric_var and df["var"].isna().any():
       df["var"] = df["var"].astype("Float64")
   ```
   Ujednolica reprezentację binów (`"8.0"`, nie `"8"`).

   > **Uwaga (rewizja): to podejście zostało odrzucone.** `Int64` jest nullable,
   > więc obsługuje `pd.NA` bez promocji. Różnica `"8"`/`"8.0"` dotyczy wyłącznie
   > etykiety `bin` i nie wpływa na scoring (`assign` mapuje po numerycznym
   > `discrete`, nie po stringu) — była naginaniem kodu pod oczekiwanie testu,
   > a nie naprawą realnego problemu. Docelowo **zachowujemy naturalny typ**
   > (sekcja 3, „Zasady ogólne", pkt 1).

4. **`groupby(by="var", dropna=False, sort=False, observed=False)`** — braki
   trafiają do osobnej grupy zamiast wypadać, kolejność kontrolowana ręcznie.

5. **`cols_to_Float` definiowana raz, przed `if min_info`**, aplikowana tylko
   do kolumn faktycznie obecnych w wyniku:
   ```python
   astype_map = {k: v for k, v in cols_to_Float.items() if k in wyn.columns}
   if astype_map:
       wyn = wyn.astype(astype_map)
   ```
   (`cols_to_Float` = `od`, `srodek`, `do`, `mean`, `median` → `Float64`,
   bo bywają złożone z samych braków.)

6. **Permutacja `<NA>` → … → `TOTAL` stosowana zawsze**, niezależnie od `sort_by`:
   ```python
   order_key = []
   for idx in wyn.index:
       if idx == NA_BIN_NAME:   order_key.append(-1)
       elif idx == "TOTAL":     order_key.append(n)
       else:                    order_key.append(len(order_key))
   wyn = wyn.iloc[sorted(range(n), key=lambda i: order_key[i])]
   wyn["nr"] = list(range(1, len(wyn) + 1))   # renumeracja po permutacji
   ```

7. **`convert_dtypes()` na samym końcu** — dociąga rozszerzone typy (`int64 → Int64`),
   żeby wynik był zgodny z referencją budowaną tym samym pipeline'em.

### 2.2. Reguły dla `bckt_cut_stats`

- Binowanie przez `pd.cut(..., ordered=True)` zapisywane do kolumny `discrete`
  (typ `Categorical`), a nie stringifikowane od razu.
- Mapowanie `bin → (od, do)` budowane **przed** sortowaniem, dzięki czemu
  granice kwantyli trafiają do właściwych wierszy mimo późniejszych permutacji:
  ```python
  bin_to_quantile = {}
  bin_count = 0
  for idx in wyn.index:
      if idx not in ["TOTAL", NA_BIN_NAME]:
          bin_to_quantile[idx] = (float(kwantyle.iloc[bin_count]),
                                  float(kwantyle.iloc[bin_count + 1]))
          bin_count += 1
  ```
- `nr` renumerowane po permutacji (`range(1, len(wyn) + 1)`).
- `discrete` ustawiane na `np.nan` dla wszystkich wierszy (przedział opisują
  `od`/`srodek`/`do`).

### 2.3. Zmiana łamiąca: `bckt_stats_over_time`

Funkcja zwracała listę 4 ramek (`[pivot, pivot_normalized, pivot_target, pivot_pred]`).
Na `typy` uproszczono ją do **zwracania pojedynczego znormalizowanego pivotu**
(udział wartości `var` w obrębie każdej daty sumuje się do 1):
```python
df = pd.DataFrame({"czas": czas, "var": var, "weights": weights})
pivot = df.pivot_table(index="czas", columns="var",
                       values="weights", aggfunc="sum", fill_value=0)
return pivot.div(pivot.sum(axis=1), axis=0)
```
Domyślne wagi = 1/wiersz, gdy `weights is None`.

> Uwaga: na gałęzi `main` `bckt_stats_over_time` poszła **inną drogą** (Gini w czasie,
> dwa wykresy). Ta zmiana z `typy` jest niekompatybilna z `main` i wymaga decyzji,
> które API zostaje. Patrz sekcja 4.

### 2.4. Korekty w testach

Część failów to były **błędy w testach**, nie w kodzie:
- ujednolicenie `discr_type` (`Int64` / `Float64` zamiast numpy `float64`)
  tam, gdzie kolumna zawiera `pd.NA`,
- dodanie `.convert_dtypes()` do referencji budowanych przez `from_dict`,
- `weights=None` zamiast odwołania do nieistniejącej kolumny `df["weights"]`,
- poprawa zafałszowanych wartości oczekiwanych (przestawione `od/srodek/do`,
  błędna normalizacja per-okres).

Po komplecie zmian na `typy`: **14/14 testów przechodzi**.

---

## 3. Docelowy model (rekomendacja architektoniczna)

Sednem wszystkich problemów jest **doklejanie wiersza `TOTAL` do otypowanej
tabeli**. Zgodnie z TODO zostawionym w [buck.py](../src/buckets/buck.py)
(okolice deklaracji funkcji) proponowany kierunek:

> Wprowadzić klasę, w której podstawową strukturą danych jest tabela bucketów
> **bez** wiersza `TOTAL`. Total (i ewentualny wiersz braków) jest dorzucany
> dopiero na żądanie — przez parametr `with_total` przy zwracaniu/wyświetlaniu.

Korzyści:
- agregaty mają jednorodne, przewidywalne typy (brak `NaN`/`"TOTAL"`
  wstrzykiwanych do otypowanych kolumn),
- `TOTAL` i `<NA>` to warstwa prezentacji, a nie część modelu danych,
- znikają specjalne ścieżki typowania i większość `astype`/`convert_dtypes`.

### Kontrakt typów (do utrwalenia)

| Kolumna | Typ docelowy | Uwagi |
|---|---|---|
| indeks | `object` (stringi) | `<NA>` jako stała `NA_BIN_NAME`; **bez** rzutowania na `string` |
| `bin` | `string` | stringowa reprezentacja wartości / przedziału |
| `discrete` | typ wejścia (`Int64`/`Float64`/`string`/`Categorical`) | `TOTAL`: `"TOTAL"` dla nienumerycznych, `pd.NA` dla numerycznych |
| `nr` | `Int64` | renumeracja po każdej permutacji |
| `od`, `srodek`, `do`, `mean`, `median` | `Float64` | bywają złożone z samych braków → muszą być rozszerzone |
| `sum_target`, `n_obs` | `Int64` | |
| `avg_target`, `avg_pred`, `pct_obs` | `Float64` | |

### Zasady ogólne

1. **Zachować naturalny typ zmiennej** — `Int64` zostaje `Int64`, `Float64`
   zostaje `Float64`. **NIE** promować `Int64 → Float64` (rewizja względem
   pierwotnego podejścia z `typy` — patrz sekcja 2, pkt 3). `Int64` jest nullable,
   więc trzyma `pd.NA` bez promocji, a `discrete` ma zachować całkowitoliczbowość.
2. Indeks tabeli wynikowej zawsze `object`.
3. Jeden punkt „kanonizacji typów” (`convert_dtypes` + mapa `astype`)
   na samym końcu funkcji, nigdy w środku pipeline'u.
4. **Bin `<NA>` zawsze jako pierwszy wiersz, `TOTAL` zawsze jako ostatni —
   bez względu na sortowanie.**
   Tabela ma dwa „specjalne" wiersze, które nie powinny mieszać się ze zwykłymi
   binami: wiersz braków (`<NA>`) i wiersz podsumowania (`TOTAL`). Niezależnie od
   tego, czy i po której kolumnie użytkownik posortował tabelę (`sort_by`), na
   końcu przestawiamy (permutujemy) wiersze tak, by `<NA>` był na samej górze, a
   `TOTAL` na samym dole; zwykłe biny zachowują kolejność z sortowania pomiędzy
   nimi. Inaczej np. `sort_by="avg_target"` wrzuciłby `TOTAL` w środek tabeli.
   Kolejność operacji w pipelinie:
   1. agregacja i wyliczenie kolumn,
   2. opcjonalne `sort_values(by=sort_by)`,
   3. **na samym końcu** — permutacja `<NA>` na początek / `TOTAL` na koniec,
   4. nadanie kolumny `nr` = `1..n` **po** permutacji (numeracja musi odpowiadać
      finalnej kolejności wierszy, a nie tej sprzed sortowania/permutacji).
5. Braki w indeksie reprezentować wyłącznie `NA_BIN_NAME`, nigdy `pd.NA`
   (unika `ValueError` przy `.isin`/hashowaniu).

---

## 4. Otwarte decyzje

1. **`bckt_stats_over_time` — które API?** Wersja z `typy` (pojedynczy pivot)
   vs wersja z `main` (Gini w czasie, dwa wykresy). Trzeba wybrać jeden kontrakt;
   ewentualnie rozdzielić na dwie funkcje o różnych nazwach.
2. **Sprzeczne oczekiwania testów** (`Int64` vs `float64` dla tego samego
   `test_df_2`) — ujednolicić referencje na pipeline
   `from_records → convert_dtypes` i rozszerzone typy pandas.
3. **Klasa-wrapper z `with_total`** — czy wdrażać teraz, czy najpierw ustabilizować
   reguły typów na obecnej, „płaskiej” strukturze.
4. **Przeniesienie na `main`** — zmiany z `typy` nie da się zmergować wprost
   (rozjazd od `50dd86b`). Reguły z sekcji 2–3 trzeba zaaplikować ręcznie do
   aktualnego [buck.py](../src/buckets/buck.py).
