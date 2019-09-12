#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importar librerías
import pandas as pd
import numpy as np

import os

import folium
from folium import plugins

from wwo_hist import retrieve_hist_data

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV



def leer_data():
    data = pd.read_csv('../Data/carpetas-de-investigacion-pgj-de-la-ciudad-de-mexico.csv', sep=';')
    return data


def transformar_data():
    data.drop(columns=['ao_hechos', 'mes_hechos', 'calle_hechos2', 'geopoint', 'ao_inicio', 'mes_inicio', 'fecha_inicio'], inplace=True)
    data['fecha_hechos'] = pd.to_datetime(data.fecha_hechos, errors='coerce')
    data.longitud.astype('float', inplace=True)
    data.latitud.astype('float', inplace=True)
    data.dropna(inplace=True)
    return data


def get_clima():
    os.chdir("../Data")
    frequency = 1
    start_date = '01-JAN-2014'
    end_date = '30-AUG-2019'
    api_key = '28f7f02aa28d4afe9dc215223190509'
    location_list = ['mexico_city']
    hist_weather_data = retrieve_hist_data(api_key, location_list, start_date, end_date, frequency, location_label = False, export_csv=True, store_df = True)
    clima = pd.read_csv('../Data/mexico_city.csv')
    clima.drop(columns=['maxtempC', 'mintempC', 'totalSnow_cm', 'sunHour', 'uvIndex.1', 'moonrise', 'moonset', 'sunrise', 'sunset', 'HeatIndexC', 'WindChillC', 'WindGustKmph'], inplace=True)
    clima.columns = ['fecha_hechos', 'uv', 'ilu_luna', 'punto_rocio', 'temp_sentir', 'nubosidad', 'humedad', 'precipitacion', 'presion', 'temperatura', 'visibilidad', 'dir_viento', 'vel_viento']
    return clima


def get_alcaldia_delitos():
    roma_n = data[(data.alcaldia_hechos == 'CUAUHTEMOC') & (data.colonia_hechos == 'ROMA NORTE')]
    roma_n = roma_n[(roma_n.categoria_delito == 'ROBO A TRANSEUNTE EN VÍA PÚBLICA CON Y SIN VIOLENCIA') | (roma_n.delito == 'ROBO A TRANSEUNTE DE CELULAR SIN VIOLENCIA') | (roma_n.delito == 'ROBO A TRANSEUNTE DE CELULAR CON VIOLENCIA') | (roma_n.delito == 'ROBO A TRANSEUNTE SALIENDO DEL BANCO CON VIOLENCIA')]
    roma_n.sort_values(by='fecha_hechos', ascending=True, inplace=True)
    roma_n = pd.merge_asof(roma_n, clima, on='fecha_hechos')
    roma_n['nombre_dia'] = roma_n.fecha_hechos.dt.weekday
    festivos = ['01-01', '01-05', '01-06', '02-05', '02-04', '03-21', '05-01', '05-05', '09-15', '09-16', '10-31', '11-01', '11-02', '11-20', '12-12', '12-24', '12-25', '12-31']
    años = ['2015', '2016', '2017', '2018', '2019']
    festivo = [i+'-'+x for i in años for x in festivos]
    festivo = pd.DataFrame(festivo, columns=['dia'])
    festivo['dia_festivo'] = 1
    festivo['dia'] = pd.to_datetime(festivo.dia).dt.date
    roma_n['dia'] = roma_n.fecha_hechos.dt.date
    roma_n = pd.merge(roma_n, festivo, how='left', on='dia')
    roma_n.dia_festivo.fillna(0, inplace=True)
    roma_n['dia_festivo'] = roma_n.dia_festivo.astype('int', inplace=True)
    roma_n.drop(columns='dia', inplace=True)
    return roma_n


def mapear():
    os.chdir("../Images")
    mapa=folium.Map(location=[19.443056, -99.144444], zoom_start=15)
    for index, row in roma_n.iterrows():
        folium.CircleMarker([row['latitud'], row['longitud']], radius=1, fill_color="#3db7e4").add_to(mapa)
    geo = roma_n[['latitud', 'longitud']].as_matrix()
    mapa.add_children(plugins.HeatMap(geo, radius=15))
    mapa.save('mapa.html')


def preparar_prediccion():
    prueba = roma_n.copy()
    prueba.drop(columns=['fiscalia', 'agencia', 'unidad_investigacion', 'alcaldia_hechos', 'punto_rocio', 'colonia_hechos', 'calle_hechos', 'temp_sentir', 'categoria_delito'], inplace=True)
    X = prueba.drop(columns=['fecha_hechos', 'latitud', 'longitud'])
    y = prueba[['fecha_hechos', 'latitud', 'longitud']]
    label = LabelEncoder()
    X.delito = label.fit_transform(X.delito)
    X.categoria_delito = label.fit_transform(X.categoria_delito)
    X.calle_hechos = label.fit_transform(X.calle_hechos)
    X['año'] = y.fecha_hechos.dt.year
    process = StandardScaler()
    X = process.fit_transform(X)
    y['dia'] = y.fecha_hechos.dt.day
    y['mese'] = y.fecha_hechos.dt.month
    y['hora'] = y.fecha_hechos.dt.hour
    y['minuto'] = y.fecha_hechos.dt.minute
    y.drop(columns='fecha_hechos', inplace=True)
    return X, y


def prediccion():
    etr = ExtraTreesRegressor(n_estimators=750, max_depth=400, random_state=1)
    etr.fit(X, y)
    pre = etr.predict(X)
    prediction = pd.DataFrame(pre, columns=['latitud', 'longitud', 'day', 'month', 'hour', 'minute'])
    prediction[['day', 'month', 'hour', 'minute']] = prediction[['day', 'month', 'hour', 'minute']].round().astype('int')
    prediction['year'] = '2020'
    prediction['fecha_hechos'] = pd.to_datetime(prediction[['year', 'day', 'month', 'hour', 'minute']], errors='coerce')
    prediction.drop(columns=['year', 'day', 'month', 'hour', 'minute'], inplace=True)
    prediction.sort_values(by='fecha_hechos', inplace=True)
    return prediction


def mapear_prediccion():
    os.chdir("../Images")
    prediction['weight'] = [i for i in range(len(prediction))]
    mapa_final = folium.Map(location=[19.443056, -99.144444], zoom_start=15)
    geo = [[[row['latitud'], row['longitud']] for index, row in prediction[prediction['weight'] == i].iterrows()] for i in range(len(prediction))]
    index = ['{:%Y-%m-%d %H-%M-%S}'.format(i) for i in prediction.fecha_hechos]
    hm = plugins.HeatMapWithTime(geo, index=index, radius=20, auto_play=True, max_opacity=0.8, name='Robo a transeunte')
    hm.add_to(mapa_final)
    mapa_final.save('mapa_final.html')