import streamlit as st
import pandas as pd
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import NearMiss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import StackingClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    completeness_score
)
from sklearn.cluster import DBSCAN
import tensorflow as tf
import pickle


def Teacher():
    st.title("Модель обучения")    
    models = ["k-NN", "Logistic Regression", "SVM"]

    return st.selectbox("Выберите модель", models)
    
def WTeacher():
    st.title("Модель обучения")    
    models = ["CART", "KMeans", "DBSCAN"]

    return st.selectbox("Выберите модель", models)
    
def Ancemble():
    st.title("Модель обучения")    
    models = ["Bagging", "Gradient Boosting", "Stacking"]

    return st.selectbox("Выберите модель", models)
    
def NN():
    return "NN"
    
def Prediction(model):
    y_pred = model.predict(X_test)

    threshold = 0.5
    y_pred = (y_pred > threshold).astype(int)

    st.write('Accuracy: {:.3f}'.format(accuracy_score(y_test, y_pred)))
    st.write('Precision: {:.3f}'.format(precision_score(y_test, y_pred)))
    st.write('Recall: {:.3f}'.format(recall_score(y_test, y_pred)))
    st.write('F1-score: {:.3f}'.format(f1_score(y_test, y_pred)))
    st.write('ROC-AUC: {:.3f}'.format(roc_auc_score(y_test, y_pred)))
    
def PredictionClaster(model):
    y_pred = model.predict(X)
    st.write("Silhouette Score:", silhouette_score(X, y_pred))
    st.write("Calinski-Harabasz Score:", calinski_harabasz_score(X, y_pred))
    st.write("Davies-Bouldin Score:", davies_bouldin_score(X, y_pred))
    st.write("Adjusted Rand Index:", adjusted_rand_score(y, y_pred))
    st.write("Completeness Score:", completeness_score(y, y_pred))
    

uploaded_file = st.file_uploader("Выберите файл датасета", type=["csv", "xlsx", "txt"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file, sep='\t') 
        
        
    st.title("Тип модели обучения")    
    models = ["Обучение с учителем", "Обучение без учителя", "Ансабль", "Нейронная сеть"]

    selected_model_type = st.selectbox("Выберите тип", models)
    
    if selected_model_type is not None:
        if selected_model_type == "Обучение с учителем":
            selected_model = Teacher()
        elif selected_model_type == "Обучение без учителя":
            selected_model = WTeacher()
        elif selected_model_type == "Ансабль":
            selected_model = Ancemble()
        elif selected_model_type == "Нейронная сеть":
            selected_model = NN()
            
            
    button_clicked = st.button("Начать обучение")

    if button_clicked:
        st.title("Предобработка данных")   
    
        progress_bar = st.progress(0)
        i = 0
        df = df.set_index('id')
        progress_bar.progress(i + 10)
        i = i + 10
        cols = df.columns.tolist()
        cols = [col.lower().replace(" ", "_") for col in cols]
        df.columns = cols
        progress_bar.progress(i + 10)
        i = i + 10
        if df.isnull().any().any():
            for column in df.columns:
                if df[column].dtype == 'int64':
                    df[column].fillna(df[column].median(), inplace=True)
                elif df[column].dtype == 'float64':
                    df[column].fillna(df[column].mean(), inplace=True)
                else:
                    df[column].fillna(df[column].mode().iloc[0], inplace=True)
        progress_bar.progress(i + 10)
        i = i + 10
        df = df.drop_duplicates().reset_index(drop=True)
        progress_bar.progress(i + 10)
        i = i + 10
        df = pd.get_dummies(df)
        outlier = df[['length']]
        Q1 = outlier.quantile(0.25)
        Q3 = outlier.quantile(0.75)
        IQR = Q3-Q1
        df_filtered = outlier[~((outlier < (Q1 - 1.5 * IQR)) |(outlier > (Q3 + 1.5 * IQR))).any(axis=1)]
        index_list = list(df_filtered.index.values)
        df_filtered = df[df.index.isin(index_list)]
        progress_bar.progress(i + 20)
        i = i + 20
        
        scaler = StandardScaler()

        numeric_features = ['length'] 
        df[numeric_features] = scaler.fit_transform(df[numeric_features])
        progress_bar.progress(i + 20)
        i = i + 20
        
        y = df_filtered["delay"]
        X = df_filtered.drop(["delay"], axis=1)
        X, _, y, _ = train_test_split(X, y, test_size=0.9, random_state=42)
        
        nm = NearMiss()
        X, y = nm.fit_resample(X, y.ravel())
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        progress_bar.progress(i + 20)
        i = i + 20
        st.success("Предобработка! Завершена")
        
        if selected_model == "k-NN":
            X_train = np.array(X_train, order='C')
            y_train = np.array(y_train, order='C')
            with open('models/knn_model.pkl', 'rb') as file:
                loaded_knn_model = pickle.load(file)
            Prediction(loaded_knn_model)
        elif selected_model == "Logistic Regression":
            with open('models/lr_model.pkl', 'rb') as file:
                lr_model = pickle.load(file)
            Prediction(lr_model)
        elif selected_model == "SVM":
            with open('models/svm_model.pkl', 'rb') as file:
                svm_model = pickle.load(file)
            Prediction(svm_model)
        elif selected_model == "CART":
            with open('models/CART_model.pkl', 'rb') as file:
                CART_model = pickle.load(file)
            Prediction(CART_model)
        elif selected_model == "KMeans":
            with open('models/KMeans_model.pkl', 'rb') as file:
                KMeans_model = pickle.load(file)
            Prediction(KMeans_model)
        elif selected_model == "DBSCAN":
            with open('models/DBSCAN_model.pkl', 'rb') as file:
                DBSCAN_model = pickle.load(file)
            y_pred = DBSCAN_model.fit_predict(X)
            st.write("Silhouette Score:", silhouette_score(X, y_pred))
            st.write("Calinski-Harabasz Score:", calinski_harabasz_score(X, y_pred))
            st.write("Davies-Bouldin Score:", davies_bouldin_score(X, y_pred))
            st.write("Adjusted Rand Index:", adjusted_rand_score(y, y_pred))
            st.write("Completeness Score:", completeness_score(y, y_pred))
        elif selected_model == "Bagging":
            with open('models/bagging_model.pkl', 'rb') as file:
                bagging_model = pickle.load(file)
            Prediction(bagging_model)
        elif selected_model == "Gradient Boosting":
            with open('models/gradient_model.pkl', 'rb') as file:
                gradient_model = pickle.load(file)
            Prediction(gradient_model)
        elif selected_model == "Stacking":
            with open('models/stacking_model.pkl', 'rb') as file:
                stacking_model = pickle.load(file)
            Prediction(stacking_model)
        elif selected_model == "NN":
            from tensorflow.keras.models import load_model
            model = load_model('models/NN_model.h5')
            X_test = X_test.astype('float32')
            y_test = y_test.astype('float32')
            Prediction(model)
            