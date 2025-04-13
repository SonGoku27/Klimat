import pandas as pd

df = pd.read_csv('../PythonProject4/Data.csv', sep=';', encoding='Utf-8')
df.dropna()
df.columns = df.columns.str.replace(' ', '_').str.lower()
df['год'] = df['год'].astype('int')
df_no_duplicates = df.drop_duplicates(
    subset=['время_года', 'способ_охлаждения', 'режим_при_смешанном_типе_охлаждения', 'способ_обогрева', 'пол',
            'ощущение_температуры_(bool)', 'предпочтительное_изменение_температуры', 'ощущение_движения_воздуха_(bool)',
            'предпочтительное_изменение_движения_воздуха', 'занавески', 'вентилятор',
            'окно', 'двери', 'отопление'],
    keep=False)

