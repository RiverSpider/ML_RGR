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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

st.title("Получить предсказание задержки рейса.")

st.header("Авиакомпания")        
airlines = ['XE', 'UA', 'OO', 'WN', 'AS', 'FL', 'AA', 'CO', 'DL', '9E', 'MQ', 'YV', 'US', 'EV', 'B6', 'OH', 'HA', 'F9']

airline = st.selectbox("Авиякомпания", airlines)

st.header("Номер полета")    
flight = st.number_input("Номер:", min_value=0, max_value=10000, value=3036)

st.header("Город отлета")    
airportfroms = ['IAH', 'DEN', 'SFO', 'PHX', 'LAS', 'ATL', 'MIA', 'PVD', 'BUR',
       'CLE', 'PDX', 'MKE', 'SAN', 'TUL', 'SJU', 'BOS', 'FLL', 'TPA',
       'ORD', 'DFW', 'MEM', 'SEA', 'MCO', 'MSY', 'COS', 'BUF', 'MDW',
       'MSP', 'SLC', 'LAX', 'OAK', 'LIH', 'CLT', 'EWR', 'ROC', 'SAV',
       'DTW', 'SYR', 'PWM', 'SNA', 'LGA', 'JFK', 'SMF', 'CEC', 'PHL',
       'RSW', 'BTV', 'ABQ', 'PIT', 'DCA', 'HNL', 'PBI', 'DAL', 'MKG',
       'TLH', 'CHA', 'CVG', 'CMI', 'FNT', 'MCI', 'LGB', 'TUS', 'MHT',
       'AEX', 'HSV', 'BDL', 'IAD', 'OMA', 'SJC', 'SAT', 'CID', 'BWI',
       'LBB', 'ICT', 'BNA', 'HRL', 'PSP', 'IND', 'LFT', 'ROA', 'MQT',
       'STL', 'LNK', 'DSM', 'DAY', 'CWA', 'PIA', 'EAU', 'BFL', 'ELP',
       'HDN', 'DRO', 'GPT', 'CRW', 'MBS', 'HPN', 'PHF', 'RIC', 'GEG',
       'RDU', 'GRR', 'MLU', 'ATW', 'RNO', 'CIC', 'LIT', 'LEX', 'HOU',
       'CAK', 'SHV', 'FWA', 'MLB', 'ALB', 'FAT', 'SGU', 'ITO', 'ORF',
       'ACV', 'JAN', 'GSP', 'TYS', 'HLN', 'CAE', 'ILM', 'FAI', 'WRG',
       'SBA', 'SGF', 'EUG', 'JAX', 'XNA', 'CSG', 'MOT', 'CMH', 'ONT',
       'PNS', 'GRB', 'RAP', 'SCE', 'MRY', 'SIT', 'AUS', 'KOA', 'OKC',
       'OGG', 'ISP', 'SDF', 'LCH', 'ERI', 'BOI', 'CHS', 'CRP', 'MDT',
       'MFR', 'AMA', 'ASE', 'MOB', 'GRK', 'FAR', 'AVL', 'MAF', 'IDA',
       'GSO', 'ANC', 'JAC', 'VPS', 'BKG', 'AVP', 'STT', 'ABE', 'ECP',
       'YUM', 'BHM', 'BIS', 'FAY', 'SPS', 'ABI', 'SUN', 'RKS', 'MGM',
       'DAB', 'MSN', 'SPI', 'BMI', 'GCC', 'FCA', 'TRI', 'EVV', 'AZO',
       'PSE', 'SRQ', 'JNU', 'BLI', 'SBN', 'GJT', 'DLH', 'KTN', 'BTR',
       'TWF', 'CMX', 'BZN', 'FSD', 'GNV', 'BRO', 'PAH', 'PIH', 'GGG',
       'MYR', 'GTF', 'TVC', 'CLD', 'RST', 'MMH', 'BIL', 'LAN', 'RDD',
       'CPR', 'EYW', 'MHK', 'MFE', 'LMT', 'PSG', 'EKO', 'MSO', 'ROW',
       'VLD', 'TXK', 'BET', 'SBP', 'SMX', 'CDC', 'ABY', 'IYK', 'MOD',
       'YAK', 'SWF', 'MLI', 'AGS', 'GFK', 'OTZ', 'EGE', 'OTH', 'PSC',
       'LWS', 'GUM', 'LRD', 'FLG', 'COD', 'BGR', 'TOL', 'GUC', 'MTJ',
       'LYH', 'LWB', 'CDV', 'BQN', 'EWN', 'MEI', 'COU', 'BRW', 'BTM',
       'FSM', 'RDM', 'HTS', 'CYS', 'ACT', 'BQK', 'OME', 'TYR', 'PLN',
       'LSE', 'BGM', 'OAJ', 'ADQ', 'ELM', 'SAF', 'STX', 'DBQ', 'GTR',
       'ACY', 'CLL', 'SCC', 'CHO', 'UTM', 'ADK', 'PIE', 'FLO', 'IPL',
       'SJT', 'DHN', 'ITH', 'TEX', 'ABR']

airportfrom = st.selectbox("Город", airportfroms)
    
st.header("Город прилета")    
airporttos = ['CHS', 'ONT', 'MRY', 'PDX', 'SAT', 'MSY', 'EWR', 'SJC', 'TPA',
       'ORD', 'DTW', 'SEA', 'MEM', 'DFW', 'ATL', 'JFK', 'BRO', 'MIA',
       'SLC', 'ICT', 'LGA', 'CLT', 'ANC', 'ORF', 'MDW', 'IAH', 'MSP',
       'MCI', 'SFO', 'MKE', 'SAN', 'HOU', 'BUR', 'HNL', 'DEN', 'DCA',
       'TUS', 'YUM', 'PHX', 'CHA', 'LAX', 'EVV', 'PSP', 'PNS', 'SMF',
       'IAD', 'CLE', 'RNO', 'RSW', 'PIT', 'TUL', 'LAS', 'TWF', 'LIT',
       'SNA', 'ITO', 'BWI', 'IND', 'STL', 'COS', 'GRR', 'SBP', 'GPT',
       'HSV', 'FLL', 'OAK', 'FSD', 'ABQ', 'PHL', 'MCO', 'RDM', 'CMI',
       'RAP', 'OGG', 'OMA', 'MOB', 'FAT', 'XNA', 'SAV', 'EUG', 'AZO',
       'HPN', 'AUS', 'ELM', 'BNA', 'TVC', 'ELP', 'DSM', 'CVG', 'SHV',
       'BTR', 'AMA', 'ABE', 'BOS', 'FSM', 'GEG', 'DAL', 'CWA', 'MAF',
       'CID', 'BTV', 'FAY', 'SDF', 'CMH', 'SBA', 'BOI', 'DLH', 'LIH',
       'ATW', 'LCH', 'MFR', 'JAX', 'LBB', 'BDL', 'BQN', 'PVD', 'LRD',
       'LGB', 'GSO', 'RIC', 'LFT', 'MSN', 'BUF', 'TYS', 'FWA', 'PSG',
       'FNT', 'OKC', 'SYR', 'RDU', 'SJU', 'CRW', 'CAK', 'KTN', 'MGM',
       'STT', 'PBI', 'PWM', 'ABI', 'ROC', 'GRB', 'TRI', 'LEX', 'GTR',
       'JAN', 'AVL', 'MKG', 'ACT', 'CAE', 'ALB', 'EGE', 'MBS', 'GRK',
       'FAR', 'AGS', 'DAY', 'SCC', 'FAI', 'DRO', 'ILM', 'IDA', 'BHM',
       'MFE', 'HTS', 'ECP', 'PIA', 'EKO', 'TEX', 'TYR', 'CPR', 'SRQ',
       'MDT', 'MTJ', 'PSC', 'MSO', 'HRL', 'BET', 'JAC', 'SWF', 'MOD',
       'MHT', 'VPS', 'CHO', 'LSE', 'BIL', 'GSP', 'GFK', 'CRP', 'LWS',
       'RKS', 'ACV', 'OAJ', 'MQT', 'ASE', 'SUN', 'COD', 'KOA', 'TLH',
       'FCA', 'JNU', 'FLG', 'MHK', 'BMI', 'CLD', 'EAU', 'CYS', 'SPS',
       'ROA', 'GTF', 'GJT', 'SAF', 'DHN', 'SGU', 'SBN', 'HDN', 'PIE',
       'GNV', 'CSG', 'VLD', 'MEI', 'LAN', 'MLI', 'RST', 'SGF', 'BFL',
       'CIC', 'MYR', 'BIS', 'BTM', 'BLI', 'SIT', 'AVP', 'WRG', 'ERI',
       'IYK', 'BGM', 'CEC', 'BZN', 'ISP', 'PAH', 'CDV', 'OME', 'SMX',
       'MLB', 'DBQ', 'MLU', 'CMX', 'PHF', 'ACY', 'AEX', 'DAB', 'ROW',
       'IPL', 'HLN', 'ABY', 'GUC', 'LNK', 'SJT', 'EWN', 'BQK', 'EYW',
       'RDD', 'LMT', 'STX', 'ITH', 'BKG', 'COU', 'OTZ', 'SPI', 'PSE',
       'UTM', 'ADQ', 'PIH', 'GCC', 'OTH', 'SCE', 'CLL', 'PLN', 'LWB',
       'BGR', 'BRW', 'CDC', 'GGG', 'MMH', 'YAK', 'MOT', 'ADK', 'LYH',
       'FLO', 'TXK', 'GUM', 'TOL', 'ABR']

airportto = st.selectbox("Город", airporttos)
    
st.header("День недели")    
dayofweek = st.number_input("Номер:", min_value=1, max_value=7, value=4)

st.header("Время полета (в минутах)")    
time = st.number_input("Число:", min_value=1, max_value=10000, value=1195)
    
st.header("Длинна полета (в километрах)")    
length = st.number_input("Число:", min_value=1, max_value=10000, value=131)

data = {'flight': [flight],
        'dayofweek': [dayofweek], 'time': [time], 'length': [length]}
df = pd.DataFrame(data)

for airport in airlines:
    if airport == airline:
        df[f"airline_{airport}"] = 1
    else: df[f"airline_{airport}"] = 0
    
for airport in airportfroms:
    if airport == airportfrom:
        df[f"airportfrom_{airport}"] = 1
    else: df[f"airportfrom_{airport}"] = 0
    
for airport in airporttos:
    if airport == airportto:
        df[f"airportto_{airport}"] = 1
    else: df[f"airportto_{airport}"] = 0
        
X = df.values.flatten()
X = X.reshape(1, -1)
st.header(X.shape)
button_clicked = st.button("Предсказать")

if button_clicked:
    st.header("1 - будет задержка")
    st.header("0 - задержки не будет")        
    with open('models/knn_model.pkl', 'rb') as file:
            loaded_knn_model = pickle.load(file)
    with open('models/lr_model.pkl', 'rb') as file:
            lr_model = pickle.load(file)
    with open('models/svm_model.pkl', 'rb') as file:
            svm_model = pickle.load(file)
    with open('models/CART_model.pkl', 'rb') as file:
            CART_model = pickle.load(file)
    with open('models/KMeans_model.pkl', 'rb') as file:
            KMeans_model = pickle.load(file)
    with open('models/DBSCAN_model.pkl', 'rb') as file:
            DBSCAN_model = pickle.load(file)
    with open('models/bagging_model.pkl', 'rb') as file:
            bagging_model = pickle.load(file)
    with open('models/gradient_model.pkl', 'rb') as file:
            gradient_model = pickle.load(file)
    with open('models/stacking_model.pkl', 'rb') as file:
            stacking_model = pickle.load(file)
    from tensorflow.keras.models import load_model
    model = load_model('models/NN_model.h5')
    
    st.header("KNN говорит:")
    st.write(f"{loaded_knn_model.predict(X)}")
    
    st.header("lr говорит:")
    st.write(f"{lr_model.predict(X)}")
    
    st.header("svm говорит:")
    st.write(f"{svm_model.predict(X)}")
    
    st.header("CART говорит:")
    st.write(f"{CART_model.predict(X)}")
    
    st.header("KMeans говорит:")
    st.write(f"{KMeans_model.predict(X)}")
    
    st.header("DBSCAN говорит:")
    st.write(f"{DBSCAN_model.fit_predict(X)}")
    
    st.header("bagging говорит:")
    st.write(f"{bagging_model.predict(X)}")
    
    st.header("gradient говорит:")
    st.write(f"{gradient_model.predict(X)}")
    
    st.header("stacking говорит:")
    st.write(f"{stacking_model.predict(X)}")
    
    st.header("NN говорит:")
    st.write(f"{model.predict(X)}")
    