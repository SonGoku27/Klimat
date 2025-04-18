import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

pd.set_option("display.max_columns", None)

df = pd.read_csv("Data.csv", delimiter=";")

df["Возрастная группа"] = pd.cut(
    df["Возраст"],
    bins=[0, 18, 30, 45, 60, 120],
    labels=["<18", "18-30", "30-45", "45-60", "60+"],
    right=False
)

df["RH комфорт"] = pd.cut(
    df["RH"],
    bins=[0, 30, 60, 100],
    labels=["Сухо", "Комфорт", "Влажно"],
    right=False
)

os.makedirs("plots", exist_ok=True)

df["Способ охлаждения"] = df["Способ охлаждения"].astype(str).str.strip()

for country in df["Страна"].dropna().unique():
    print(f"\n=== {country.upper()} ===")

    df_country = df[df["Страна"] == country].copy()

    summary = (
        df_country.dropna(subset=["Пол", "Возрастная группа", "Температура воздуха в помещении"])
        .groupby(["Пол", "Возрастная группа"], observed=True)
        .agg({
            "Температура воздуха в помещении": "mean",
            "Температура воздуха на улице": "mean",
            "RH": "mean"
        })
        .reset_index()
        .rename(columns={
            "Температура воздуха в помещении": "Средняя температура в помещении",
            "Температура воздуха на улице": "Средняя температура на улице",
            "RH": "Средняя влажность"
        })
    )

    print("\nСводная таблица:")
    print(summary)

    corr_cols = [
        "Температура воздуха в помещении",
        "Температура воздуха на улице",
        "RH",
        "Скорость воздуха",
        "Оценка комфорта",
        "Среднемесячная температура на улице"
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
        "Температура воздуха в помещении",
        "Температура воздуха на улице",
        "RH",
        "Скорость воздуха",
        "Оценка комфорта",
        "Способ охлаждения"
    ]
    df_model = df_country[model_cols].dropna()

    if not df_model.empty:

        X = pd.concat([
            df_model[["Температура воздуха на улице", "RH", "Скорость воздуха", "Оценка комфорта"]],
            pd.get_dummies(df_model["Способ охлаждения"], prefix="Охлаждение", dtype=float)
        ], axis=1)

        X = X.astype(float)
        y = df_model["Температура воздуха в помещении"].astype(float)

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
