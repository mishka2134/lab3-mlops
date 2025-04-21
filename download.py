import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

def download_data():
    # Загрузка данных из гита
    url = "https://raw.githubusercontent.com/mishka2134/lab1/main/energy_data.csv"
    df = pd.read_csv(url)
    df.to_csv("energy_raw.csv", index=False)
    return df

def clear_data(path2df):
    # Отчистка данных
    df = pd.read_csv(path2df).dropna()
    cats = ['Building Type', 'Day of Week']
    nums = ['Square Footage', 'Number of Occupants', 'Appliances Used', 'Average Temperature']

    # Проверка данных
    df = df[(df['Square Footage'] > 0) & (df['Square Footage'] <= 1e6)]
    df = df[(df['Number of Occupants'] > 0) & (df['Number of Occupants'] <= 1000)]
    df = df[(df['Average Temperature'] >= -30) & (df['Average Temperature'] <= 50)]
    df = df[(df['Energy Consumption'] > 0) & (df['Energy Consumption'] <= 1e6)]
    df = df[(df['Appliances Used'] >= 0) & (df['Appliances Used'] <= 100)]

    # Кодирование категорий
    df[cats] = OrdinalEncoder().fit_transform(df[cats])
    df.to_csv('energy_cleaned.csv', index=False)
    return True

# Выполнение pipeline
data = download_data()
clear_data("energy_raw.csv")
