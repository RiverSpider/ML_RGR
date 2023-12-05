import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

uploaded_file = st.file_uploader("Выберите файл датасета", type=["csv", "xlsx", "txt"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file, sep='\t') 

    st.write("Загруженный датасет:", df)

    st.title("Датасет airlines task")

    st.header("Тепловая карта с корреляцией между основными признаками")

    plt.figure(figsize=(12, 8))
    selected_cols = ['Flight', 'DayOfWeek', 'Time', 'Length', 'Delay']
    selected_df = df[selected_cols]
    sns.heatmap(selected_df.corr(), annot=True, cmap='coolwarm')
    plt.title('Тепловая карта с корреляцией')
    st.pyplot(plt)
    
    st.header("Гистограммы для основных признаков")
    
    columns = ['Flight', 'DayOfWeek', 'Time', 'Length']

    for col in columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[col], bins=100, kde=True)
        plt.title(f'Гистограмма для {col}')
        st.pyplot(plt)
    
    st.header("Ящик с усами для основных признаков")
    for col in columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(df[col])
        plt.title(f'{col}')
        plt.xlabel('Значение')
        st.pyplot(plt)
    
    columns = ['Airline', 'DayOfWeek', 'DayOfWeek']
    st.header("Круговая диаграмма основных категориальных признаков")
    for col in columns:
        plt.figure(figsize=(8, 8))
        df[col].value_counts().plot.pie(autopct='%1.1f%%')
        plt.title(f'{col}')
        plt.ylabel('')
        st.pyplot(plt)
        