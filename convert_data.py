import numpy as np
import pandas as pd

dane = pd.read_csv(
    "data/default of credit card clients.csv",
    dtype_backend="numpy_nullable",
)

dane = dane.rename(columns={
    "default payment next month": "target",
    "LIMIT_BAL": "credit_limit",
    "SEX": "sex",
    "EDUCATION": "education",
    "MARRIAGE": "marriage",
    "AGE": "age",
})

# Kolumny statyczne (te same dla każdego klienta)
static_cols = ["ID", "credit_limit", "sex", "education", "marriage", "age", "target"]

# Mapowanie: data -> (kolumna statusu spłaty, kolumna salda, kolumna kwoty spłaty)
# PAY_0=wrzesień, PAY_2=sierpień, ..., PAY_6=kwiecień (numeracja oryginalna z datasetu)
months = [
    (pd.Timestamp("2005-09-01"), "PAY_0", "BILL_AMT1", "PAY_AMT1"),
    (pd.Timestamp("2005-08-01"), "PAY_2", "BILL_AMT2", "PAY_AMT2"),
    (pd.Timestamp("2005-07-01"), "PAY_3", "BILL_AMT3", "PAY_AMT3"),
    (pd.Timestamp("2005-06-01"), "PAY_4", "BILL_AMT4", "PAY_AMT4"),
    (pd.Timestamp("2005-05-01"), "PAY_5", "BILL_AMT5", "PAY_AMT5"),
    (pd.Timestamp("2005-04-01"), "PAY_6", "BILL_AMT6", "PAY_AMT6"),
]

frames = []
for date, pay_col, bill_col, paid_col in months:
    temp = dane[static_cols + [pay_col, bill_col, paid_col]].copy()
    temp["date"] = date
    temp = temp.rename(columns={
        pay_col:  "pay_status",
        bill_col: "bill_amount",
        paid_col: "pay_amount",
    })
    frames.append(temp)

dane_long = (
    pd.concat(frames, ignore_index=True)
    .sort_values(["ID", "date"])
    .reset_index(drop=True)
)

col_order = ["ID", "date"] + [c for c in dane_long.columns if c not in ("ID", "date")]
dane_long = dane_long[col_order]

dane.to_parquet("data/default_credit_card.parquet", index=False)
dane_long.to_parquet("data/default_credit_card_long.parquet", index=False)

print("Zapisano data/default_credit_card.parquet")
print("Zapisano data/default_credit_card_long.parquet")
print(f"Wymiary: {dane_long.shape}")


# ---------------------------------------------------------------------------
# Wersja "amend": rozszerzenie tabeli long o trzy kolumny testowe
# (zmienna dyskretna numeryczna, stringowa kategoryczna, uporządkowana Category)
# ---------------------------------------------------------------------------
dane_long_amend = pd.read_parquet("data/default_credit_card_long.parquet")

rng = np.random.default_rng(42)
n = len(dane_long_amend)

# 1) zmienna liczbowa dyskretna o wartościach 1, 3, 8, 10, 11, 18
dane_long_amend["disc_num"] = pd.array(
    rng.choice([1, 3, 8, 10, 11, 18], size=n), dtype="Int64"
)

# 2) zmienna stringowa o wartościach 'a', 'b', 'c'
dane_long_amend["litera"] = pd.array(
    rng.choice(["a", "b", "c"], size=n), dtype="string"
)

# 3) uporządkowana Category: 'mały' < 'średni' < 'duży'
dane_long_amend["rozmiar"] = pd.Categorical(
    rng.choice(["mały", "średni", "duży"], size=n),
    categories=["mały", "średni", "duży"],
    ordered=True,
)

dane_long_amend.to_parquet("data/default_credit_card_long_amend.parquet", index=False)

print("Zapisano data/default_credit_card_long_amend.parquet")
print(f"Wymiary: {dane_long_amend.shape}")
