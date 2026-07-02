# coding: utf-8
"""
Most między danymi zagregowanymi (np. w Sparku) a pandasowym rdzeniem pakietu.

Projekt: spec/raport-spark.md. Kontrakt kanonicznego agregatu (sekcja 1):
ramka pandas z jedną grupą na wiersz, o kolumnach:

- kolumna zmiennej (dowolna nazwa; braki jako osobna grupa),
- opcjonalne kolumny dodatkowe przenoszone bez zmian (np. okres czasowy),
- ``n_obs``      — suma wag w grupie,
- ``sum_target`` — suma ``waga * target`` w grupie,
- ``sum_pred``   — suma ``waga * pred`` w grupie (opcjonalna).

`to_pseudo_obs` zamienia taki agregat na ważone pseudo-obserwacje, które
przechodzą przez istniejący, ważony pipeline (BucketTable, gini, drzewo,
DistributionOverTime) dając wyniki IDENTYCZNE jak dane wierszowe — dowód
równoważności: tabela w spec/raport-spark.md, sekcja 1; testy:
tests/test_pseudo_obs.py.

Ten moduł nie importuje pyspark na poziomie modułu — funkcje czysto
pandasowe działają bez Sparka; funkcje sparkowe importują pyspark leniwie.
"""

from __future__ import annotations

import copy

import pandas as pd

import buckets.column_types as ct

#: kolumny statystyk kanonicznego agregatu (wszystkie pozostałe kolumny
#: agregatu są przenoszone do pseudo-obserwacji bez zmian)
STAT_COLUMNS = ("n_obs", "sum_target", "sum_pred")

#: nazwy kolumn wynikowych pseudo-obserwacji
PSEUDO_TARGET = "target"
PSEUDO_WEIGHTS = "weights"
PSEUDO_PRED = "pred"


def to_pseudo_obs(agg: pd.DataFrame) -> pd.DataFrame:
    """
    Zamienia kanoniczny agregat na ważone pseudo-obserwacje.

    Każdy wiersz agregatu rozpada się na dwa wiersze (rozbicie targetu 0/1):

    - ``target=1`` z wagą ``sum_target``,
    - ``target=0`` z wagą ``n_obs - sum_target``,

    oba z ``pred = sum_pred / n_obs`` (gdy ``sum_pred`` obecne). Wiersze
    o wadze 0 są pomijane. Rozbicie 0/1 (a nie ``target=avg_target``)
    zachowuje całkowitość sum — kontrakt Int64 dla ``sum_target``/``n_obs``
    (spec/raport-spark.md, 5.1) — oraz dokładność drzewa i gini.

    Args:
        agg: kanoniczny agregat (patrz docstring modułu). Kolumny spoza
            ``STAT_COLUMNS`` (zmienna, okres czasowy, ...) są przenoszone
            do wyniku bez zmian.

    Returns:
        Ramka pseudo-obserwacji: kolumny przeniesione + ``target``,
        ``weights`` (+ ``pred``).
    """
    missing = [c for c in ("n_obs", "sum_target") if c not in agg.columns]
    if missing:
        raise ValueError(f"Agregat nie ma wymaganych kolumn: {missing}.")

    carry = [c for c in agg.columns if c not in STAT_COLUMNS]
    collisions = set(carry) & {PSEUDO_TARGET, PSEUDO_WEIGHTS, PSEUDO_PRED}
    if collisions:
        raise ValueError(
            f"Kolumny agregatu kolidują z nazwami pseudo-obserwacji: {collisions}."
        )

    ones = agg[carry].copy()
    ones[PSEUDO_TARGET] = 1
    ones[PSEUDO_WEIGHTS] = agg["sum_target"]

    zeros = agg[carry].copy()
    zeros[PSEUDO_TARGET] = 0
    zeros[PSEUDO_WEIGHTS] = agg["n_obs"] - agg["sum_target"]

    if "sum_pred" in agg.columns:
        avg_pred = agg["sum_pred"] / agg["n_obs"]
        ones[PSEUDO_PRED] = avg_pred
        zeros[PSEUDO_PRED] = avg_pred

    out = pd.concat([ones, zeros], ignore_index=True)
    return out[out[PSEUDO_WEIGHTS] > 0].reset_index(drop=True)


# --------------------------------------------------------------- część Spark
def _functions():
    """Leniwy import pyspark z czytelnym błędem przy braku zależności."""
    try:
        import pyspark.sql.functions as F
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Ta funkcja wymaga pyspark — zainstaluj pakiet z ekstrasem: "
            "buckets[spark]."
        ) from exc
    return F


def _micro_bin(sdf, var: str, max_levels: int, relative_error: float):
    """
    Bezpiecznik kardynalności (spec/raport-spark.md, 5.2): zastępuje wartości
    `var` środkami mikro-binów kwantylowych. To NIE jest binowanie raportowe
    (patrz sekcja 1 specyfikacji) — tylko redukcja stanu pośredniego;
    od tego momentu wyniki są kontrolowanie przybliżone.
    """
    from pyspark.ml.feature import Bucketizer

    F = _functions()

    qs = [i / max_levels for i in range(max_levels + 1)]
    edges = sdf.approxQuantile(var, qs, relative_error)
    edges = sorted(set(edges))
    if len(edges) < 2:
        return sdf  # praktycznie stała zmienna — nie ma czego zwijać

    # skrajne biny otwarte na ±inf (approxQuantile może nie objąć ekstremów);
    # jako value: środek binu, dla binów skrajnych — skrajna skończona granica
    splits = [float("-inf")] + edges + [float("inf")]
    mids = (
        [edges[0]]
        + [(lo + hi) / 2 for lo, hi in zip(edges[:-1], edges[1:])]
        + [edges[-1]]
    )

    non_null = sdf.filter(F.col(var).isNotNull())
    bucketized = Bucketizer(
        splits=splits, inputCol=var, outputCol="__bucket"
    ).transform(non_null)
    replaced = bucketized.withColumn(
        var,
        F.element_at(
            F.array(*[F.lit(m) for m in mids]),
            F.col("__bucket").cast("int") + 1,
        ),
    ).drop("__bucket")

    nulls = sdf.filter(F.col(var).isNull()).withColumn(
        var, F.col(var).cast("double")
    )
    return replaced.unionByName(nulls)


def aggregate_variable(
    sdf,
    var: str,
    target: str,
    pred: str | None = None,
    weights: str | None = None,
    time_col: str | None = None,
    max_levels: int = 100_000,
    relative_error: float = 1e-4,
) -> pd.DataFrame:
    """
    Kanoniczny agregat zmiennej liczony w Sparku (kontrakt: docstring modułu).

    `groupBy(var[, time_col])` z sumami wag — jedyna operacja wykonywana po
    stronie Sparka. Zmienna ciągła jest grupowana po SUROWEJ wartości; biny
    raportowe wyznaczy pandas na pseudo-obserwacjach (spec/raport-spark.md,
    sekcja 1). Dopiero gdy `approx_count_distinct(var) > max_levels`,
    wartości są wcześniej zwijane do środków mikro-binów kwantylowych
    (`_micro_bin`) — wyniki stają się wtedy przybliżone.

    Args:
        sdf: pyspark.sql.DataFrame z danymi wierszowymi.
        var: nazwa zmiennej analizowanej.
        target: nazwa kolumny celu (bez braków — walidowane po stronie Sparka,
            bo do agregatu braki targetu już by nie dotarły).
        pred: opcjonalna nazwa kolumny predykcji.
        weights: opcjonalna nazwa kolumny wag.
        time_col: opcjonalna kolumna okresu — dokłada wymiar czasowy agregatu.
        max_levels: próg mikro-binowania zmiennych numerycznych.
        relative_error: dokładność approxQuantile przy mikro-binowaniu.
    """
    F = _functions()

    if sdf.filter(F.col(target).isNull()).limit(1).count() > 0:
        raise ValueError("W zmiennej 'target' nie może być braków danych!")

    numeric = dict(sdf.dtypes)[var] not in ("string", "boolean")
    if numeric:
        n_distinct = sdf.agg(F.approx_count_distinct(var)).first()[0]
        if n_distinct > max_levels:
            sdf = _micro_bin(sdf, var, max_levels, relative_error)

    w = F.col(weights) if weights is not None else F.lit(1.0)
    aggs = [
        F.sum(w).alias("n_obs"),
        F.sum(w * F.col(target)).alias("sum_target"),
    ]
    if pred is not None:
        aggs.append(F.sum(w * F.col(pred)).alias("sum_pred"))

    keys = [var] + ([time_col] if time_col is not None else [])
    return sdf.groupBy(*keys).agg(*aggs).toPandas()


class SparkSource:
    """
    Źródło danych per zmienna dla `DatasetReport` — ścieżka Spark.

    Ten sam kontrakt co `report.PandasSource` (`frame_for`/`n_levels`):
    dla każdej zmiennej liczy kanoniczny agregat (jeden job Spark, wynik
    cache'owany) i podaje raportowi ważone pseudo-obserwacje. Użycie:

        source = SparkSource(sdf, types)
        report = DatasetReport(source, source.types)

    Uwaga: używać `source.types`, nie oryginalnego `types` — SparkSource
    pracuje na kopii, w której kolumna wag pseudo-obserwacji ma rolę WEIGHTS
    (wagi są w tej ścieżce obowiązkowe, bo niosą krotności z agregatu).

    Zalecenie wydajnościowe: `sdf.persist()` przed generowaniem raportu
    (jeden groupBy na zmienną — spec/raport-spark.md, 5.4).
    """

    def __init__(self, sdf, types: ct.ColumnTypes, max_levels: int = 100_000):
        self.sdf = sdf
        self.types = copy.deepcopy(types)
        self.max_levels = max_levels
        self._cache: dict[str, pd.DataFrame] = {}

        # wagi pseudo-obserwacji muszą być widoczne dla raportu: jeśli dane
        # nie miały kolumny wag, rejestrujemy syntetyczną (PSEUDO_WEIGHTS)
        if self.types.weights_col is None:
            if PSEUDO_WEIGHTS in sdf.columns:
                raise ValueError(
                    f"Ramka ma kolumnę {PSEUDO_WEIGHTS!r}, która nie jest "
                    "kolumną wag — ustaw types.weights_col albo zmień jej nazwę."
                )
            self.types.types.loc[PSEUDO_WEIGHTS] = {
                "column_name": PSEUDO_WEIGHTS,
                "dtype": "double",
                "analytical_type": ct.AnalyticalType.CONTINUOUS,
                "role": ct.Role.WEIGHTS,
            }

    def _agg(self, var: str) -> pd.DataFrame:
        if var not in self._cache:
            self._cache[var] = aggregate_variable(
                self.sdf, var, self.types.target,
                weights=ct_weights_in_sdf(self.types, self.sdf),
                time_col=self.types.time_col,
                max_levels=self.max_levels,
            )
        return self._cache[var]

    def frame_for(self, var: str) -> pd.DataFrame:
        pobs = to_pseudo_obs(self._agg(var))
        # nazwy kolumn pseudo-obserwacji -> nazwy, których oczekuje raport
        rename = {}
        if PSEUDO_TARGET != self.types.target:
            rename[PSEUDO_TARGET] = self.types.target
        if PSEUDO_WEIGHTS != self.types.weights_col:
            rename[PSEUDO_WEIGHTS] = self.types.weights_col
        return pobs.rename(columns=rename)

    def n_levels(self, var: str) -> int:
        # liczba poziomów zmiennej (bez braków) — jak Series.nunique()
        return self._agg(var)[var].nunique()


def ct_weights_in_sdf(types: ct.ColumnTypes, sdf) -> str | None:
    """Nazwa kolumny wag, o ile istnieje w ramce Spark (pomija syntetyczną)."""
    w = types.weights_col
    return w if w is not None and w in sdf.columns else None


def column_types_from_spark(sdf, discrete_threshold: int = 20) -> ct.ColumnTypes:
    """
    Buduje `ColumnTypes` ze schematu ramki Spark (spec/raport-spark.md, 4.3).

    Typ analityczny: kolumny nienumeryczne → categorical; numeryczne →
    discrete/continuous wg `approx_count_distinct` (jeden job Spark na całą
    ramkę) i progu `discrete_threshold`. Role — heurystyka `guess_role`,
    wspólna ze ścieżką pandas. Uwaga: zliczanie jest przybliżone, więc
    zmienne na granicy progu mogą być sklasyfikowane inaczej niż w pandas —
    do skorygowania przez `set(...)` po utworzeniu obiektu.
    """
    F = _functions()

    dtypes = dict(sdf.dtypes)
    counts = sdf.agg(
        *[F.approx_count_distinct(c).alias(c) for c in sdf.columns]
    ).first().asDict()

    non_numeric = ("string", "boolean", "binary", "date", "timestamp")
    results = []
    for col in sdf.columns:
        dtype = dtypes[col]
        if dtype.startswith(non_numeric):
            analytical_type = ct.AnalyticalType.CATEGORICAL
        elif counts[col] < discrete_threshold:
            analytical_type = ct.AnalyticalType.DISCRETE
        else:
            analytical_type = ct.AnalyticalType.CONTINUOUS
        results.append({
            "column_name": col,
            "dtype": dtype,
            "analytical_type": analytical_type,
            "role": ct.guess_role(col),
        })

    frame = pd.DataFrame(results, index=list(sdf.columns))
    return ct.ColumnTypes.from_frame(frame, discrete_threshold)
