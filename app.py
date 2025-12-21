# Dashboard
import streamlit as st

# Data Wrangling
import pandas as pd
import numpy as np

# Visuals
from plotly import express as px

# SQL
import duckdb

# text data
import json

# Machine Learning
import joblib
from sklearn.metrics import root_mean_squared_error , mean_absolute_error

st.set_page_config(layout='wide', page_icon='🪨', page_title='Prediccion Rougher')

# Funciones
@st.cache_data
def load_metadata(rutas:list):
    with open(rutas[0],'r',encoding='utf-8') as f1:
        data_1 = json.load(f1)
    with open(rutas[1], 'r', encoding='utf-8') as f2:
        data_2 = json.load(f2)
    return data_1, data_2

@st.cache_data
def load_raw_data(ruta:str):
    data = pd.read_csv(ruta)
    data = data.dropna()
    return data

@st.cache_data
def load_models_and_scalers(path_models:list,path_scalers:list):
    model_1 = joblib.load(path_models[0])
    scaler_1 = joblib.load(path_scalers[0])
    model_2 = joblib.load(path_models[1])
    scaler_2 = joblib.load(path_scalers[1])
    return model_1, scaler_1, model_2, scaler_2

@st.cache_data
def load_test_matrices():
    X_test_rougher = pd.read_pickle('test_rougher_x.pkl')
    y_test_rougher = pd.read_pickle('test_rougher_y.pkl')
    X_test_final = pd.read_pickle('test_final_x.pkl')
    y_test_final = pd.read_pickle('test_final_y.pkl')
    return X_test_rougher, y_test_rougher, X_test_final, y_test_final

@st.cache_data
def sMAPE(true_values,predictions):
    numerator = np.sum(np.abs(true_values-predictions))
    denominator = np.sum((np.abs(true_values)+np.abs(predictions))/2)
    if denominator == 0:
        return 0.0
    else:
        sMAPE = (numerator/denominator) * 100
        return sMAPE

@st.cache_data
def load_plain_query(path:str):
    data = pd.read_csv(path)
    data['stage'] = pd.Series(['final','middle','rougher'])
    data = data.sort_values(by='au',ascending=True)
    return data

@st.cache_data
def average_metals_behavior(df:pd.DataFrame):
    query = duckdb.sql(f"""
    SELECT
        stage,
        metal,
        mean_value
    FROM
        df
    UNPIVOT (
    mean_value
        FOR metal 
        IN (ag,au,pb)
    );
    """).df()
    with st.expander(':material/format_list_numbered_rtl: Average Concentrate DataFrame'):
        st.dataframe(df.set_index('stage'))
    fig = px.line(query,x='stage',y='mean_value',color='metal',markers=True,title='Average Concentrate in function of Cleaning Stage')
    fig.update_layout(legend=dict(orientation="h",entrywidth=70,yanchor="bottom",y=1.02,xanchor="right",x=1))
    st.plotly_chart(fig)

def distribution_box_plot(data:pd.DataFrame, labels:list,title:str):
    query = duckdb.sql(f"""
    SELECT
        stage,
        concentrate
    FROM 
        data
    UNPIVOT(
        concentrate
            FOR stage
            IN {labels}
    );
    """).df()
    
    fig = px.box(query,x='stage',y='concentrate',color='stage',title=title)
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig)

def metal_distribution_per_stage(data:pd.DataFrame):
    metal_selection = st.pills(
        label='Concentrate Of Metal Per Cleaning Stage',
        options=[':material/diamond: Gold',':material/diamond: Silver',':material/diamond: Lead'],
        default=':material/diamond: Gold')
    if metal_selection == ':material/diamond: Gold':
        oro = ('rougher.input.feed_au', 'rougher.output.concentrate_au', 'final.output.concentrate_au')
        distribution_box_plot(data=data,labels=oro,title='Gold Concentrate Input, Middle Output, Final Output')
    if metal_selection == ':material/diamond: Silver':
        plata = ('rougher.input.feed_ag', 'rougher.output.concentrate_ag', 'final.output.concentrate_ag')
        distribution_box_plot(data=data,labels=plata,title='Silver Concentrate Input, Middle Output, Final Output')
    if metal_selection == ':material/diamond: Lead':
        plomo = ('rougher.input.feed_pb', 'rougher.output.concentrate_pb', 'final.output.concentrate_pb')
        distribution_box_plot(data=data,labels=plomo,title='Lead Concentrate Input, Middle Output, Final Output')

def testing_model_1(model,X,Y,metadata):
    with st.expander(':material/grid_4x4: Preview Columns Order For This Model'):
        st.write(f'X matrix: `{metadata['columnas_orden']}`')

    Y = np.ravel(Y) # Y viene en formato array 2D(fila, columna) con ravel obtenemos Vctor 1D 
    predictions = model.predict(X)
    score = sMAPE(true_values=Y,predictions=predictions)
    rmse = root_mean_squared_error(y_true=Y,y_pred=predictions)
    mae = mean_absolute_error(y_true=Y,y_pred=predictions)
    # Crear DataFrame para visualización profesional
    df_results = pd.DataFrame({'Real': Y,'Prediction': predictions,'MAE': np.abs(Y - predictions)})

    tab1, tab2 = st.tabs([":material/bar_chart: Distribution Analysis", ":material/bubble_chart: Fidelity: Predictions vs Real Values"])

    with tab1:
        # Histograma tipo "Violin" o "KDE" es mejor para contraste
        fig_dist = px.histogram(
            df_results,
            x=["Real", "Prediction"], # Especifico las columnas que se van a superponer
            barmode="overlay", # Barras superpuestas
            marginal="box",    # Añade un boxplot arriba para ver outliers
            opacity=0.5,
            nbins=70,
            title="Density Comparison Real Values/Prediction",
            color_discrete_sequence=px.colors.sequential.Sunsetdark)
        fig_dist.update_layout(legend=dict(orientation="h",entrywidth=70,yanchor="bottom",y=1.02,xanchor="right",x=1))
        st.plotly_chart(fig_dist, width='content')

    with tab2:
        # Gráfico de dispersión con línea de identidad
        fig_scatter = px.scatter(
            df_results, x='Real', y='Prediction',
            hover_data=['MAE'],
            opacity=0.5,
            title="Density Chart with an Identity Line",
            labels={'Real': 'Real Value (Recovery %)', 'Prediction': 'Prediction (%)'},
            color_discrete_sequence=px.colors.sequential.Sunsetdark)
        
        # Se agrega línea de 45 grados (el ideal) como referencia de prediccion con fig.add_shape
        min_val = min(Y.min(), predictions.min())
        max_val = max(Y.max(), predictions.max())
        fig_scatter.add_shape(
            type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val,
            line=dict(color="Red", dash="dash"))
        st.plotly_chart(fig_scatter, width='content')

    st.markdown(f'**MAPE Score: `{score:.4f}%`**. RMSE Score: `{rmse:.4f}`. MAE Score: `{mae:.4f}`.',text_alignment='center')
    with st.expander(':material/dataset: Preview Predictions'):
        st.dataframe(df_results)
    
    return score

def testing_model_2(model,X,Y,metadata):
    with st.expander(':material/grid_4x4: Preview Columns Order For This Model'):
        st.write(f'X matrix: `{metadata['columnas_orden']}`')

    Y = np.ravel(Y)
    predictions = model.predict(X)

    score = sMAPE(true_values=Y,predictions=predictions)
    rmse = root_mean_squared_error(y_true=Y,y_pred=predictions)
    mae= mean_absolute_error(y_true=Y,y_pred=predictions)

    df_results = pd.DataFrame({'Real': Y,'Predictions': predictions,'MAE': np.abs(Y - predictions)})
    
    tab1, tab2 = st.tabs([":material/bar_chart: Distribution Analysis", ":material/bubble_chart: Fidelity: Predictions vs Real Values"])

    with tab1:
        # Histograma tipo "Violin" o "KDE" es mejor para contraste
        fig_dist = px.histogram(
            df_results,
            x=["Real", "Predictions"], # Especifico las columnas que se van a superponer
            barmode="overlay", # Barras superpuestas
            marginal="box",    # Añade un boxplot arriba para ver outliers
            opacity=0.5,
            nbins=70,
            title="Density Comparison Real Values/Predictions",
            color_discrete_sequence=px.colors.sequential.Aggrnyl)
        fig_dist.update_layout(legend=dict(orientation="h",entrywidth=70,yanchor="bottom",y=1.02,xanchor="right",x=1))
        st.plotly_chart(fig_dist, width='content')

    with tab2:
        # Gráfico de dispersión con línea de identidad
        fig_scatter = px.scatter(
            df_results, x='Real', y='Predictions',
            hover_data=['MAE'],
            opacity=0.5,
            title="Density Chart with an Identity Line",
            labels={'Real': 'Real Value (Recovery %)', 'Predictions': 'Predictions (%)'},
            color_discrete_sequence=px.colors.sequential.Aggrnyl)
        
        # Se agrega línea de 45 grados (el ideal) como referencia de prediccion con fig.add_shape
        min_val = min(Y.min(), predictions.min())
        max_val = max(Y.max(), predictions.max())
        fig_scatter.add_shape(
            type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val,
            line=dict(color="Red", dash="dash"))
        st.plotly_chart(fig_scatter, width='content')

    st.markdown(f'**MAPE Score: `{score:.4f}%`**. RMSE Score: `{rmse:.4f}`. MAE Score: `{mae:.4f}`.',text_alignment='center')
    with st.expander(':material/dataset: Preview Predictions'):
        st.dataframe(df_results)

    return score

# Carga de datos:
query = load_plain_query(path='query_1.csv')
raw_df = load_raw_data(ruta='gold_recovery_train.csv')
meta_rougher, meta_final = load_metadata(rutas=['model_metadata_rougher.json','model_metadata_final.json'])
rougher_x, rougher_y, final_x, final_y = load_test_matrices()
GradientBoostingRegressor_1,scaler_rougher_stage,GradientBoostingRegressor_2, scaler_final_stage = load_models_and_scalers(
    path_models=['model_rougher_gb.joblib','model_final_gb.joblib'],
    path_scalers=['scaler_rougher.joblib','scaler_final.joblib'])

# ----- INTERFAZ
with st.expander(':material/landslide: Raw Data'):
    st.subheader(f'Data Dimensions: {raw_df.shape}')
    st.dataframe(raw_df,hide_index=1)

st.divider()
st.header(':material/analytics: Exploratory Data Analysis: Gold Recovery via Rougher Process',text_alignment='center')

col1, col2 = st.columns(2)
with col1:
    average_metals_behavior(df=query)
with col2:
    metal_distribution_per_stage(data=raw_df)

st.divider()
col3, col4 = st.columns(2)
with col3:
    st.subheader(':material/diagonal_line: GradientBoostingRegressor Rougher Stage')
    mape_1 = testing_model_1(model=GradientBoostingRegressor_1,X=rougher_x,Y=rougher_y,metadata=meta_rougher)
with col4:
    st.subheader(':material/diagonal_line: GradientBoostingRegressor Final Stage')
    mape_2 = testing_model_2(model=GradientBoostingRegressor_2,X=final_x,Y=final_y,metadata=meta_final)

st.subheader(
    f':material/workspace_premium: Weighted MAPE $(0.25 Rougher)$ $(0.75 Final)$ Score: :green[${(mape_1 * 0.25) + (mape_2 * 0.75)}$%]',
    text_alignment='center')