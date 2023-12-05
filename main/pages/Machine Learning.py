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
            knn = KNeighborsClassifier()
            param_grid_knn = {'n_neighbors': [100, 1001, 300]}
            grid_search_knn = GridSearchCV(knn, param_grid=param_grid_knn, cv=5)
            grid_search_knn.fit(X_train, y_train)
            best_knn = grid_search_knn.best_estimator_
            Prediction(best_knn)
        elif selected_model == "Logistic Regression":
            lr = LogisticRegression(max_iter=1000)
            penalty = ['l1', 'l2']
            param_grid_lr = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
            lr_grid = GridSearchCV(lr, param_grid=param_grid_lr, cv=5)
            lr_grid.fit(X_train, y_train)
            lr = lr_grid.best_estimator_
            Prediction(lr)
        elif selected_model == "SVM":
            svm = SVC()
            svm.fit(X_train, y_train)
            Prediction(svm)
        elif selected_model == "CART":
            #with open('../models/CART.pkl', 'rb') as file:
                #CART = pickle.load(file)
            classifier = DecisionTreeClassifier()

            param_dist = {
                "criterion": ["gini", "entropy"],
                "max_depth": randint(1, 4),
                "min_samples_split": randint(2, 20),
                "min_samples_leaf": randint(1, 10)
            }

            clf_random_search = RandomizedSearchCV(classifier, param_distributions=param_dist, n_iter=10, cv=5, random_state=42)
            clf_random_search.fit(X_train, y_train)

            CART = clf_random_search.best_estimator_
            Prediction(CART)
        elif selected_model == "KMeans":
            kmeans = KMeans(n_init=10, random_state=42) 
            parameters = {'n_clusters': [2, 3, 4, 5]}
            grid_search = GridSearchCV(kmeans, parameters)
            grid_search.fit(X)
            best_kmeans = grid_search.best_estimator_
            PredictionClaster(best_kmeans)
        elif selected_model == "DBSCAN":
            best_dbscan = DBSCAN(eps=0.1, min_samples=2)
            y_pred = best_dbscan.fit_predict(X)
            st.write("Silhouette Score:", silhouette_score(X, y_pred))
            st.write("Calinski-Harabasz Score:", calinski_harabasz_score(X, y_pred))
            st.write("Davies-Bouldin Score:", davies_bouldin_score(X, y_pred))
            st.write("Adjusted Rand Index:", adjusted_rand_score(y, y_pred))
            st.write("Completeness Score:", completeness_score(y, y_pred))
        elif selected_model == "Bagging":
            bagging = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=10)
            bagging.fit(X_train, y_train)
            Prediction(bagging)
        elif selected_model == "Gradient Boosting":
            gradient_boosting = GradientBoostingClassifier()
            gradient_boosting.fit(X_train, y_train)
            Prediction(gradient_boosting)
        elif selected_model == "Stacking":
            bagging_classifier = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=10)
            gradient_boosting_classifier = GradientBoostingClassifier()
            stacking_classifier = StackingClassifier(
                estimators=[('bagging', bagging_classifier), ('gb', gradient_boosting_classifier)],
                final_estimator=LogisticRegression()
            )
            stacking_classifier.fit(X_train, y_train)
            Prediction(stacking_classifier)
        elif selected_model == "NN":
            model_classification = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(64, activation="relu", input_shape=(608,)),
                    tf.keras.layers.Dense(128, activation="relu"),
                    tf.keras.layers.Dropout(0.05),
                    tf.keras.layers.Dense(64, activation="relu"),
                    tf.keras.layers.Dense(32, activation="relu"),
                    tf.keras.layers.Dense(16, activation="relu"),
                    # используем 1 нейрон и sigmoid
                    tf.keras.layers.Dense(1, activation="sigmoid"),
                ]
            )
            X_train = X_train.astype('float32')
            y_train = y_train.astype('float32')
            # в качестве функции активации используется бинарная  кроссэнтропия
            model_classification.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                             loss="binary_crossentropy",
                             metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC(), tf.keras.metrics.F1Score()])
            model_classification.fit(X_train, y_train, epochs=100, verbose=None)
            X_test = X_test.astype('float32')
            y_test = y_test.astype('float32')
            Prediction(model_classification)
            