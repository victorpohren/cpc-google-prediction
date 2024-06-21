import sqlalchemy
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_absolute_error, mean_squared_error
from numpy import random
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pyodbc as odbc
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
import warnings
from datetime import datetime
import locale
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix
from collections import Counter
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import connectorx as cx
from prophet import Prophet
import holidays
from holidays import CountryHoliday
from datetime import date
from scipy import stats
from sklearn.model_selection import ParameterGrid
import json
from cpc_google_prediction.secrets.get_token import get_secret
from urllib.parse import quote_plus
from sqlalchemy import create_engine, text




def sql_connection (query):
    server = server
    database = database
    username = username
    password = password
    conn = odbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER=' +
                        server+';DATABASE='+database+';UID='+username+';PWD=' + password)
    dataset = pd.read_sql_query(query,conn)
    return dataset


def sqlConnector():
    secret = secret
    server = server
    database = database
    username = username
    password = password
    conn = ('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
    quoted = quote_plus(conn)
    new_con = f'mssql+pyodbc:///?odbc_connect={quoted}'

    return create_engine(new_con, fast_executemany=True)



warnings.filterwarnings("ignore")



query_investimentos = """
SELECT
  inv.dt_campaign,
  SUM(CAST(inv.alcance AS INT)) as alcance,
  SUM(CAST(inv.impressions AS INT)) as impressions,
  SUM(CAST(inv.clicks AS INT)) as clicks,
  SUM(CAST(inv.vlr_cost AS float)) AS investimento
FROM marketing.ads_info inv
WHERE DATEPART(year, dt_campaign) >= 2022 AND
plataforma = 'Google'
and campaign_name like '%assessoria%'
GROUP BY
  dt_campaign
"""

dataset = sql_connection(query_investimentos)
print('Registros Dataset Investimentos', dataset.shape)

# Cria CPM
dataset['cpm'] = dataset['investimento']/(dataset['impressions']/1000)
dataset['cpc'] = dataset['investimento']/ dataset['clicks']

# Converte data da campanha em datetime
dataset['dt_campaign'] = dataset['dt_campaign'].apply(lambda x: pd.to_datetime(x, format='%Y/%m/%d',errors='ignore'))

# Ordena o DF
dataset = dataset.groupby(['dt_campaign'])[['cpc']].median().reset_index()

# Renomeia as colunas
dataset.rename(columns={'dt_campaign': 'ds', 'cpc': 'y'}, inplace=True)


z_scores = stats.zscore(dataset['y'])
filtered_entries = (np.abs(z_scores) >= 3)
outliers = dataset[filtered_entries]



holiday = holidays.CountryHoliday(years=np.arange(2020, 2040, 1), country='BR', state='RS') 
holidays = pd.DataFrame(holiday.items(), columns=["ds", "holiday"])


params_grid = {'seasonality_mode':('multiplicative',#'additive'
),
               'changepoint_prior_scale':[0.1,0.2,0.3,
               #0.4,0.5, 1, 5, 10
               ],
              'holidays_prior_scale':[0.1,0.2,0.3,#0.4,0.5, 1, 5, 10
              ],
              'n_changepoints' : [25, 50, 100#,150,200
              ]}
grid = ParameterGrid(params_grid)



model_parameters = pd.DataFrame(columns = ['R2', 'MAE', 'MSE', 'MAPE','Parameters'])
for p in grid:
    test = pd.DataFrame()
    random.seed(0)
    train_model =Prophet(changepoint_prior_scale = p['changepoint_prior_scale'],
                         holidays_prior_scale = p['holidays_prior_scale'],
                         n_changepoints = p['n_changepoints'],
                         seasonality_mode = p['seasonality_mode'],
                         weekly_seasonality=True,
                         daily_seasonality = True,
                         yearly_seasonality = True,
                         holidays=holidays)

    train_model.add_country_holidays(country_name='BR')
    train_model.fit(dataset)
    HORIZON=40
    future = train_model.make_future_dataframe(periods=HORIZON)
    forecast = train_model.predict(future)
    R2 = r2_score(dataset['y'].iloc[0:len(dataset)], (forecast['yhat'].iloc[0:len(future)-HORIZON]))
    MSE = mean_squared_error(dataset['y'].iloc[0:len(dataset)], (forecast['yhat'].iloc[0:len(future)-HORIZON]))
    MAE = mean_absolute_error(dataset['y'].iloc[0:len(dataset)], (forecast['yhat'].iloc[0:len(future)-HORIZON]))
    MAPE = mean_absolute_percentage_error(dataset['y'].iloc[0:len(dataset)], (forecast['yhat'].iloc[0:len(future)-HORIZON]))


    new_row = pd.DataFrame({'R2': [R2], 'MAE': [MAE], 'MSE': [MSE], 'MAPE': [MAPE], 'Parameters': [p]})
    model_parameters = pd.concat([model_parameters, new_row], ignore_index=True)
    #model_parameters = model_parameters.append({'R2': R2, 'MAE': MAE, 'MSE': MSE,'MAPE':MAPE,'Parameters':p},ignore_index=True)


parameters = model_parameters.sort_values(by=['R2', 'MAE', 'MSE', 'MAPE'], ascending=False)
parameters = parameters.reset_index(drop=True)


final_model = Prophet(holidays=holidays,
                      changepoint_prior_scale= parameters['Parameters'][0]['changepoint_prior_scale'],
                      holidays_prior_scale =parameters['Parameters'][0]['holidays_prior_scale'],
                      n_changepoints = parameters['Parameters'][0]['n_changepoints'],
                      seasonality_mode = parameters['Parameters'][0]['seasonality_mode'],
                      weekly_seasonality=True,
                      daily_seasonality = True,
                      yearly_seasonality = True)
final_model.add_country_holidays(country_name='BR')
final_model.fit(dataset)


m = Prophet()
m.fit(dataset)


future = final_model.make_future_dataframe(periods=40)
forecast = final_model.predict(future)


# Python
future = m.make_future_dataframe(periods=40, freq='D')
future.tail(40)

forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(40)


forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
forecast.drop(columns=['ds'], axis=1)[forecast.drop(columns=['ds'], axis=1) < 0] = 0
forecast['yhat'][forecast['yhat']<0]=0

forecast['ds'] = forecast['ds'].replace(".",",")
forecast['yhat'] = forecast['yhat'].replace(".",",")
forecast['yhat_lower'] = forecast['yhat_lower'].replace(".",",")
forecast['yhat_upper'] = forecast['yhat_upper'].replace(".",",")

forecast.tail(40)

df_tratado = forecast.tail(40)

df_tratado = df_tratado.rename({'ds':'data', 'yhat': 'cpc_predito', 'yhat_lower': 'min_cpc_predito', 'yhat_upper': 'max_cpc_predito'}, axis=1)
inserted_at = 'inserted_at'
df_tratado[f'{inserted_at}'] = datetime.now()


for i in range(3):
        try:
            engine = sqlConnector()
            error_content = None
            df_tratado.to_sql(name='cpc_prediction_google', index=False, con=engine, schema='data_science', 
                           if_exists='append',method='multi',chunksize=((2100//len(df_tratado.columns))-1))
            print(f'inserted into data_science.cpc_prediction_google')
            break
        except Exception as err:
            error_content = err
            print(err)
            
            

          
            
