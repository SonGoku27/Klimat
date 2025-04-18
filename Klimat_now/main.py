import pandas as pd

df = pd.read_csv('Data.csv', sep=";")


def emissions(zxc):
    for i in ['утепление', 'температура_воздуха_в_помещении', 'скорость_воздуха', 'рост',
              'среднемесячная_температура_на_улице']:
        zxc = zxc[(zxc[i] > zxc[i].quantile(0.01)) & (zxc[i] < zxc[i].quantile(0.99))]
    return zxc

def zxc(z):
    if z >= 44:
        return 'молодой возраст'
    elif 45 <= z <= 59 :
        return 'средний возраст'
    elif z > 60:
        return 'пожилой возраст'

def qwe(x):
    if x <= 1:
        return 'мало'
    elif x == 2:
        return 'средне'
    elif x > 2:
        return 'много'

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
print()
df['рост'] = df['рост'].fillna(df['рост'].describe().median())
df['вес'] = df['вес'].fillna(df['вес'].describe().median())
df['оценка_комфорта'] = df['оценка_комфорта'].fillna(df['оценка_комфорта'].describe().loc['50%'])
df.loc[df['климат'] == 'Субтроп океанич', 'климат'] = 'Субтропический океанический'
df['количество_рекламаций_кат'] = [zxc(i) for i in df['количество_рекламаций']]
df['возраст_кат'] = [qwe(i) for i in df['возраст']]
df1 = emissions(df)
print(df1.describe())

print(df1)
