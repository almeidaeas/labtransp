#Manipulação de Dados
import pandas as pd
import numpy as np

#Utilitários
import pickle
import itertools
from datetime import date, timedelta
from joblib import Parallel, delayed
import warnings
from datetime import timedelta


#Gráficos
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams

#Stats Models
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.holtwinters import Holt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller


#Estatísticos
from scipy import stats
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error


# Suprimir todos os warnings
warnings.filterwarnings('ignore')


global local_origem 
global data_corte_treinamento

local_origem = "/home/cdsw"
data_corte_treinamento = '2024-01-01'


def agrupamento_periodico(df):
    #Agrupamento por mês
    #     
    df['Periodo'] = pd.to_datetime(dict(year=df.Ano, 
                                        month=df.Mes,
                                        day=1)) + pd.offsets.MonthEnd()    
    
    df_subpref = df[['Sub_Prefeitura', 'Volume_Passageiros_BU_VT_DIN', 'Periodo']]
    df_subpref.drop_duplicates(inplace=True, ignore_index=True)    
    df_subpref = df_subpref.groupby(['Periodo', 'Sub_Prefeitura'], as_index=False).agg({'Volume_Passageiros_BU_VT_DIN':"sum"}).reset_index(drop=True)
    
    
    df_zonas = df[['Zona', 'Volume_Passageiros_BU_VT_DIN', 'Periodo']]
    df_zonas.drop_duplicates(inplace=True, ignore_index=True)    
    df_zonas = df_zonas.groupby(['Periodo', 'Zona'], as_index=False).agg({'Volume_Passageiros_BU_VT_DIN':"sum"}).reset_index(drop=True)    
    
    return df_subpref, df_zonas

def ajuste_datas_historica(df, coluna, param):
    #Normalização das Datas
    
    #Filtra a série temporal desejada, considerando os parâmetros passados
    df = df[df[coluna]==param.upper()]


    #Referencia temporal para a série dummy
    inicio = df['Periodo'].min()
    final = df['Periodo'].max()


    #Dummy com datas entre os períodos analisados    
    dummy_datas = pd.date_range(inicio, final-timedelta(days=1),freq='m')
    Serie = pd.DataFrame(dummy_datas, columns=["Periodo"])
    print("O intevalo corresponde a " + str(len(Serie["Periodo"])), " datas")

    #Criação da Série histórica
    Serie_Historica = Serie.merge(df[['Periodo','Volume_Passageiros_BU_VT_DIN']], 
                                  on="Periodo", 
                                  how="left")
    Serie_Historica["Volume_Passageiros (BU + VT + DIN)"] = Serie_Historica["Volume_Passageiros_BU_VT_DIN"].fillna(0)
    
    
    Serie_Treinamento = Serie_Historica[Serie_Historica["Periodo"] < data_corte_treinamento]
    Serie_Teste = Serie_Historica[Serie_Historica["Periodo"] >= data_corte_treinamento]

    Serie_Historica = Serie_Historica.set_index('Periodo').asfreq('M')
    Serie_Treinamento = Serie_Treinamento.set_index('Periodo').asfreq('M')
    Serie_Teste = Serie_Teste.set_index('Periodo').asfreq('M')


    Serie_Historica.sort_index(inplace=True)
    Serie_Treinamento.sort_index(inplace=True)
    Serie_Teste.sort_index(inplace=True)
    print("Séries geradas para o modelo de séries temporais")
    
    return Serie_Historica, Serie_Treinamento, Serie_Teste

def conversao_array(df_historico, df_treinamento, df_teste):
    y = df_historico['Volume_Passageiros_BU_VT_DIN']
    y_treinamento = df_treinamento['Volume_Passageiros_BU_VT_DIN']
    y_teste = df_teste['Volume_Passageiros_BU_VT_DIN']
    y_treinamento_exp = df_treinamento['Volume_Passageiros_BU_VT_DIN'].pow(2)
    
    return y, y_treinamento, y_teste, y_treinamento_exp

def plot_serie(y):
    #y = y['Volume_Passageiros (BU + VT + DIN)'].resample('W-MON').sum()
    y.plot(figsize=(20, 5))
    plt.show()

def plot_decomposicao(y):
    rcParams['figure.figsize'] = 18, 12
    decomposicao = sm.tsa.seasonal_decompose(y, model='additive')
    fig = decomposicao.plot()
    plt.show()

def suavizacao_simples(y, α=0.8):
    '''
    Ajsute do modelo de Suavização Simples
    Leva em consideração um alpha de 0.8 por padrão
    '''
    model = SimpleExpSmoothing(y, 
                               initialization_method="estimated")
    
    model_fit = model.fit(smoothing_level=α,
                          optimized=True)
    print(model_fit.summary())
    return model_fit       

def suavizacao_exponencial(y, α=0.8):
    '''
    Ajsute do modelo de Suavização Exponencial
    Leva em consideração um alpha de 0.8 por padrão
    Tendencia e Sazonalidade foram definidas como multiplicativas
    A periodicidade da sazonalidade é mensal (seasonal_periods = 12)    
    '''
    
    tendencia = "multiplicative"
    sazonalidade = "multiplicative"
    
    model = ExponentialSmoothing(endog = y, 
                                 trend = tendencia,
                                 seasonal = sazonalidade,
                                 seasonal_periods = 12,
                                 damped_trend = False,
                                 initialization_method="estimated")
                                                                 
    model_fit = model.fit(smoothing_level= α, 
                          smoothing_trend=None,
                          smoothing_seasonal=None,
                          damping_trend=None,
                          optimized=True,
                          remove_bias=True)
    
    print(model_fit.summary())
    return model_fit      

def holt_winters(y, α=0.8):
    '''
    Ajsute do modelo de Suavização Exponencial
    Leva em consideração um alpha de 0.8 por padrão
    Tendencia e Sazonalidade foram definidas como multiplicativas
    A periodicidade da sazonalidade é mensal (seasonal_periods = 12)    
    '''
    
    tendencia = "multiplicative"
    sazonalidade = "multiplicative"
    
    
    model = Holt(endog = y,
                 damped_trend = False,                 
                 initialization_method="estimated")
                                                                 
    model_fit = model.fit(smoothing_level= α, 
                          damping_trend=None,
                          optimized=True,                          
                          remove_bias=True)
    
    print(model_fit.summary())
    return model_fit       

def arma(y, w=3):
    '''
    Ajuste do modelo de médias móveis    
    '''    
    param = (0, 0, w)
    model = ARIMA(y,
                  order=param)
    
    model_fit = model.fit()
    print(model_fit.summary())
    return model_fit    

def arima_sazonal(y, 
                  param=(0, 0, 0),
                  param_seasonal = (0, 0, 0, 12)):
    
    model = sm.tsa.statespace.SARIMAX(y,
                                    order=param,
                                    seasonal_order=param_seasonal,
                                    enforce_stationarity=True,
                                    enforce_invertibility=False)
    model_fit = model.fit(disp=0)
    print(model_fit.summary())
    return model_fit  

def analise_residuos(modelo):
    residuos = pd.DataFrame(modelo.resid)
    fig, ax = plt.subplots(1,2)
    residuos.plot(title="Resíduos", ax=ax[0])
    residuos.plot(kind='kde', title='Densidade', ax=ax[1])
    plt.show()
    return residuos


# Função para verificar a estacionaridade
def check_stationarity(y):
    result = adfuller(y)
    return result[1] < 0.05

# Função para realizar o grid search e encontrar o melhor modelo
def grid_search_best_model(df, 
                           column, 
                           param):
    # Normaliza as datas e divide em conjuntos de treinamento e teste
    Serie_Historica, Serie_Treinamento, Serie_Teste = ajuste_datas_historica(df, coluna=column, param=param)
    
    # Converte para arrays
    y, y_treinamento, y_teste, y_treinamento_exp = conversao_array(Serie_Historica, Serie_Treinamento, Serie_Teste)
    
    # Verifica a estacionaridade e diferencia se necessário
    if not check_stationarity(y_treinamento):
        y_treinamento = y_treinamento.diff().dropna()
        y_teste = y_teste.diff().dropna()
    
    
    best_models = {}
    
    # Função auxiliar para grid search
    def evaluate_model(model, y_treinamento, y_teste):
        forecast = model.forecast(len(y_teste))
        mape = mean_absolute_percentage_error(y_teste, forecast)
        return mape, model
    
    # Grid search para Simple Exponential Smoothing
    results = Parallel(n_jobs=-1)(delayed(evaluate_model)(SimpleExpSmoothing(y_treinamento).fit(smoothing_level=alpha), y_treinamento, y_teste) for alpha in np.arange(0.1, 2.1, 0.1))
    best_mape, best_model = min(results, key=lambda x: x[0])
    best_models['SimpleExpSmoothing'] = best_model
    
    # Grid search para Exponential Smoothing
    results = Parallel(n_jobs=-1)(delayed(evaluate_model)(ExponentialSmoothing(y_treinamento, trend='additive', seasonal='additive', seasonal_periods=12).fit(smoothing_level=alpha), y_treinamento, y_teste) for alpha in np.arange(0.1, 2.1, 0.1))
    best_mape, best_model = min(results, key=lambda x: x[0])
    best_models['ExponentialSmoothing'] = best_model
    
    # Grid search para Holt-Winters
    results = Parallel(n_jobs=-1)(delayed(evaluate_model)(Holt(y_treinamento).fit(smoothing_level=alpha), y_treinamento, y_teste) for alpha in np.arange(0.1, 2.1, 0.1))
    best_mape, best_model = min(results, key=lambda x: x[0])
    best_models['HoltWinters'] = best_model
    
    # Grid search para ARIMA e SARIMA
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in pdq]
    
    # Melhor ARMA
    results = Parallel(n_jobs=-1)(delayed(evaluate_model)(ARIMA(y_treinamento, order=(param[0], 0, param[2])).fit(), y_treinamento, y_teste) for param in pdq)
    best_mape, best_model = min(results, key=lambda x: x[0])
    best_models['ARMA'] = best_model    
    

    def evaluate_arima(param, y_treinamento, y_teste):
        try:
            model = ARIMA(y_treinamento, 
                          order=param,
                          enforce_stationarity=True,
                          enforce_invertibility=False).fit()
            
            forecast = model.forecast(len(y_teste))
            mape = mean_absolute_percentage_error(y_teste, forecast)
            return mape, model
        except (LinAlgError, ValueError):
            return float('inf'), None
    
    # Melhor ARIMA
    results = Parallel(n_jobs=-1)(delayed(evaluate_arima)(param,y_treinamento, y_teste) for param in pdq)
    best_mape, best_model = min(results, key=lambda x: x[0])
    best_models['ARIMA'] = best_model
    
    
    def evaluate_sarima(param, param_seasonal, y_treinamento, y_teste):
        try:
            model = sm.tsa.statespace.SARIMAX(y_treinamento, 
                                              order=param, 
                                              seasonal_order=param_seasonal, 
                                              enforce_stationarity=True, enforce_invertibility=False).fit(disp=0)
            
            forecast = model.forecast(len(y_teste))
            mape = mean_absolute_percentage_error(y_teste, forecast)
            return mape, model
        except (LinAlgError, ValueError):
            return float('inf'), None
        
        
    # Melhor SARIMA
    results = Parallel(n_jobs=-1)(delayed(evaluate_sarima)(param, param_seasonal, y_treinamento, y_teste) for param in pdq for param_seasonal in seasonal_pdq)
    best_mape, best_model = min(results, key=lambda x: x[0])
    best_models['SARIMA'] = best_model    
    
    
    return best_models

def load_models(file_path):
    '''
    Função utilizada para carregar os pickles 
    contendo os modelos de forecast
    '''    
    with open(file_path, 'rb') as f:
        models = pickle.load(f)
    return models

# Função para fazer o forecast e salvar os resultados em um CSV
def criar_forecast(df, 
                   column, 
                   param, 
                   model_file, 
                   output_file, 
                   periodos):
    
    print(f"Processando {column} = {param}")
    
    # Normaliza as datas e divide em conjuntos de treinamento e teste
    Serie_Historica, Serie_Treinamento, Serie_Teste = ajuste_datas_historica(df, coluna=column, param=param)
    
    # Verificar se há valores NaT nos índices de data
    if Serie_Treinamento.index.hasnans or Serie_Teste.index.hasnans:
        print("Erro: Os índices de data contêm valores NaT.")
        print(Serie_Treinamento.index)
        print(Serie_Teste.index)
        return
    
    # Converte para arrays
    y, y_treinamento, y_teste, y_treinamento_exp = conversao_array(Serie_Historica, Serie_Treinamento, Serie_Teste)
    
    # Preencher valores NaT com a data mínima válida
    y_treinamento = y_treinamento.fillna(method='ffill').fillna(method='bfill')
    
    # Decomposição da série temporal
    print(f"Gerando decomposição da série")
    decomposicao = seasonal_decompose(y_treinamento, model='additive')
    tendencia = decomposicao.trend
    sazonalidade = decomposicao.seasonal
    residuo = decomposicao.resid
    
    # Cálculo do ACF e PACF
    print(f"Gerando função de autocorrelação")
    acf_values = acf(y_treinamento, alpha=0.05)
    pacf_values = pacf(y_treinamento, alpha=0.05)
    
    # Carregar os modelos a partir do arquivo pickle
    models = load_models(model_file.format(param))
    
    # Fazer o forecast de 12 meses para cada modelo
    print(f"Gerando forecasts")
    forecasts = {model_name: model.forecast(steps=periodos) for model_name, model in models.items()}
    
    
    # Garantir que todos os arrays tenham o mesmo comprimento
    max_len = len(y)
    tendencia = np.pad(tendencia, (max_len - len(tendencia), 0), 'constant', constant_values=np.nan)
    sazonalidade = np.pad(sazonalidade, (max_len - len(sazonalidade), 0), 'constant', constant_values=np.nan)
    residuo = np.pad(residuo, (max_len - len(residuo), 0), 'constant', constant_values=np.nan)
    acf_values = np.pad(acf_values[0], (0, max_len - len(acf_values[0])), 'constant', constant_values=np.nan)
    pacf_values = np.pad(pacf_values[0], (0, max_len - len(pacf_values[0])), 'constant', constant_values=np.nan)
    forecasts = {model_name: np.pad(forecast, (max_len - len(forecast), 0), 'constant', constant_values=np.nan) for model_name, forecast in forecasts.items()}    
        
    
    # Criar DataFrame com os resultados
    results = pd.DataFrame({
        'Data': Serie_Historica.index,
        'Série Histórica': y,
        'Tendencia': tendencia,
        'Sazonalidade': sazonalidade,
        'Resíduo': residuo,
        'FAC': acf_values,
        'P_FAC': pacf_values,
        'MM': forecasts['SimpleExpSmoothing'],
        'Suavização': forecasts['SimpleExpSmoothing'],
        'Suavização Exponencial': forecasts['ExponentialSmoothing'],
        'Holt Winters': forecasts['HoltWinters'],
        'ARIMA': forecasts['ARIMA'],
        'SARIMA': forecasts['SARIMA']
    })
    
    # Salvar os resultados em um arquivo CSV
    results.to_csv(output_file, 
                   sep=";",
                   decimal=",",
                   encoding='latin-1', 
                   index=False)
    
    print(f"Forecast salvo em {output_file}")
    
    

    
    
    
def grid_search_best_model_no_parallel(df, column, param):
    # Normaliza as datas e divide em conjuntos de treinamento e teste
    Serie_Historica, Serie_Treinamento, Serie_Teste = ajuste_datas_historica(df, coluna=column, param=param)
    
    # Converte para arrays
    y, y_treinamento, y_teste, y_treinamento_exp = conversao_array(Serie_Historica, Serie_Treinamento, Serie_Teste)
    
    # Verifica a estacionaridade e diferencia se necessário
    if not check_stationarity(y_treinamento):
        y_treinamento = y_treinamento.diff().dropna()
        y_teste = y_teste.diff().dropna()
    
    best_models = {}
    
    # Função auxiliar para grid search
    def evaluate_model(model, y_treinamento, y_teste):
        forecast = model.forecast(len(y_teste))
        mape = mean_absolute_percentage_error(y_teste, forecast)
        return mape, model
    
    # Grid search para Simple Exponential Smoothing
    results = [evaluate_model(SimpleExpSmoothing(y_treinamento).fit(smoothing_level=alpha), y_treinamento, y_teste) for alpha in np.arange(0.1, 2.1, 0.1)]
    best_mape, best_model = min(results, key=lambda x: x[0])
    best_models['SimpleExpSmoothing'] = best_model
    
    # Grid search para Exponential Smoothing
    results = [evaluate_model(ExponentialSmoothing(y_treinamento, trend='additive', seasonal='additive', seasonal_periods=12).fit(smoothing_level=alpha), y_treinamento, y_teste) for alpha in np.arange(0.1, 2.1, 0.1)]
    best_mape, best_model = min(results, key=lambda x: x[0])
    best_models['ExponentialSmoothing'] = best_model
    
    # Grid search para Holt-Winters
    results = [evaluate_model(Holt(y_treinamento).fit(smoothing_level=alpha), y_treinamento, y_teste) for alpha in np.arange(0.1, 2.1, 0.1)]
    best_mape, best_model = min(results, key=lambda x: x[0])
    best_models['HoltWinters'] = best_model
    
    # Grid search para ARIMA e SARIMA
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in pdq]
    
    # Melhor ARMA
    results = [evaluate_model(ARIMA(y_treinamento, order=(param[0], 0, param[2])).fit(), y_treinamento, y_teste) for param in pdq]
    best_mape, best_model = min(results, key=lambda x: x[0])
    best_models['ARMA'] = best_model
    
    def evaluate_arima(param, y_treinamento, y_teste):
        try:
            model = ARIMA(y_treinamento, order=param, enforce_stationarity=True, enforce_invertibility=False).fit()
            forecast = model.forecast(len(y_teste))
            mape = mean_absolute_percentage_error(y_teste, forecast)
            return mape, model
        except (LinAlgError, ValueError):
            return float('inf'), None
    
    # Melhor ARIMA
    results = [evaluate_arima(param, y_treinamento, y_teste) for param in pdq]
    best_mape, best_model = min(results, key=lambda x: x[0])
    best_models['ARIMA'] = best_model
    
    def evaluate_sarima(param, param_seasonal, y_treinamento, y_teste):
        try:
            model = sm.tsa.statespace.SARIMAX(y_treinamento, order=param, seasonal_order=param_seasonal, enforce_stationarity=True, enforce_invertibility=False).fit(disp=0)
            forecast = model.forecast(len(y_teste))
            mape = mean_absolute_percentage_error(y_teste, forecast)
            return mape, model
        except (LinAlgError, ValueError):
            return float('inf'), None
    
    # Melhor SARIMA
    results = [evaluate_sarima(param, param_seasonal, y_treinamento, y_teste) for param in pdq for param_seasonal in seasonal_pdq]
    best_mape, best_model = min(results, key=lambda x: x[0])
    best_models['SARIMA'] = best_model

    return best_models