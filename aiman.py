import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

pd.set_option("display.max_columns", None)

df = pd.read_csv("Data.csv", delimiter=";")

df["возрастная группа"] = pd.cut(
    df["возраст"],
    bins=[0, 18, 30, 45, 60, 120],
    labels=["<18", "18-30", "30-45", "45-60", "60+"],
    right=False
)

df["RH комфорт"] = pd.cut(
    df["RH"],
    bins=[0, 30, 60, 100],
    labels=["сухо", "комфорт", "влажно"],
    right=False
)

os.makedirs("plots", exist_ok=True)

df["способ охлаждения"] = df["способ охлаждения"].astype(str).str.strip()

for country in df["страна"].dropna().unique():
    print(f"\n=== {country.upper()} ===")

    df_country = df[df["страна"] == country].copy()

    summary = (
        df_country.dropna(subset=["пол", "возрастная группа", "температура воздуха в помещении"])
        .groupby(["пол", "возрастная группа"], observed=True)
        .agg({
            "температура воздуха в помещении": "mean",
            "температура воздуха на улице": "mean",
            "RH": "mean"
        })
        .reset_index()
        .rename(columns={
            "температура воздуха в помещении": "средняя температура в помещении",
            "температура воздуха на улице": "средняя температура на улице",
            "RH": "средняя влажность"
        })
    )

    print("\nСводная таблица:")
    print(summary)

    corr_cols = [
        "температура воздуха в помещении",
        "температура воздуха на улице",
        "RH",
        "скорость воздуха",
        "оценка комфорта",
        "среднемесячная температура на улице"
    ]
    corr_df = df_country[corr_cols].dropna()

    if not corr_df.empty:
        corr = corr_df.corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title(f"Корреляция ({country})")
        plt.tight_layout()
        plt.savefig(f"plots/correlation_{country}.png")
        plt.close()
        print(f"Сохранена корреляционная матрица: plots/correlation_{country}.png")
    else:
        print("Нет данных для корреляции.")

    model_cols = [
        "температура воздуха в помещении",
        "температура воздуха на улице",
        "RH",
        "скорость воздуха",
        "оценка комфорта",
        "способ охлаждения"
    ]
    df_model = df_country[model_cols].dropna()

    if not df_model.empty:

        X = pd.concat([
            df_model[["температура воздуха на улице", "RH", "скорость воздуха", "оценка комфорта"]],
            pd.get_dummies(df_model["способ охлаждения"], prefix="охлаждение", dtype=float)
        ], axis=1)

        X = X.astype(float)
        y = df_model["температура воздуха в помещении"].astype(float)

        X.insert(0, "Intercept", 1.0)

        beta = np.linalg.lstsq(X.values, y.values, rcond=None)[0]
        y_pred = X @ beta
        r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))

        print("\nКоэффициенты регрессии:")
        for name, coef in zip(X.columns, beta):
            print(f"{name}: {coef:.3f}")
        print(f"R2: {r2:.4f}")
        print(f"RMSE: {rmse:.4f}")
    else:
        print("Недостаточно данных для регрессии.")
