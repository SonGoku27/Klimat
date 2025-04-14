import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def emissions(zxc):
    # выбросы - утепление(max - 2.08), температура_воздуха_в_помещении(max - 77.00), скорость_воздуха(max - 63.8300),
    # рост(min - 5.23), среднемесячная_температура_на_улице(max - 328.00)
    for i in ['утепление', 'температура_воздуха_в_помещении', 'скорость_воздуха', 'рост',
              'среднемесячная_температура_на_улице']:
        zxc = zxc[(zxc[i] > zxc[i].quantile(0.01)) & (zxc[i] < zxc[i].quantile(0.99))]
    return zxc


df = pd.read_csv('Data.csv', sep=';', encoding='Utf-8')
pd.set_option('display.max_columns', None)
df.dropna()
df.columns = df.columns.str.replace(' ', '_').str.lower()
df['год'] = df['год'].astype('int')
df_no_duplicates = df.drop_duplicates(
    subset=['время_года', 'способ_охлаждения', 'режим_при_смешанном_типе_охлаждения', 'способ_обогрева', 'пол',
            'ощущение_температуры_(bool)', 'предпочтительное_изменение_температуры', 'ощущение_движения_воздуха_(bool)',
            'предпочтительное_изменение_движения_воздуха', 'занавески', 'вентилятор',
            'окно', 'двери', 'отопление'],
    keep=False)

df['возраст'].plot(kind='hist', density=1, bins=20, stacked=False, alpha=.5, color='grey')


df1 = emissions(df)
print(df1.describe())
