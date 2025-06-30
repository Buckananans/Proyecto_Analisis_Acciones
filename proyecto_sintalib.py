import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import socket
from scipy.fft import fft, fftfreq
import numpy as np

def detectar_patrones(data, ticker):
    resultados = []

    open_ = data[f'Open_{ticker}'].values
    high = data[f'High_{ticker}'].values
    low = data[f'Low_{ticker}'].values
    close = data[f'Close_{ticker}'].values
    dates = data['Date'].values

    for i in range(1, len(data)):
        o, h, l, c = open_[i], high[i], low[i], close[i]
        prev_o, prev_c = open_[i-1], close[i-1]
        body = abs(c - o)
        upper = h - max(c, o)
        lower = min(c, o) - l

        label = None

        # Doji
        if body < 0.02 * (h - l):
            label = "Doji"

        # Hammer
        elif lower > 2 * body and upper < body:
            label = "Martillo"

        # Shooting Star
        elif upper > 2 * body and lower < body:
            label = "Shooting Star"

        # Engulfing
        elif (o < c) and (prev_o > prev_c) and (o < prev_c) and (c > prev_o):
            label = "Engulfing Alcista"
        elif (o > c) and (prev_o < prev_c) and (o > prev_c) and (c < prev_o):
            label = "Engulfing Bajista"

        if label:
            resultados.append({
                "Date": dates[i],
                "Tipo": label,
                "Precio": c
            })

    return resultados

# Configurar la página
st.set_page_config(page_title="Stock Dashboard", layout="wide")
st.title("Análisis Histórico de Acciones (2010-2024)")

# Cargar datos
@st.cache_data
def load_data():
    df = pd.read_csv("15 Years Stock Data of NVDA AAPL MSFT GOOGL and AMZN.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df.copy()

df = load_data()

# Sidebar con controles
st.sidebar.header("Parámetros de Visualización")

hostname = socket.gethostname()
local_ip = socket.gethostbyname(hostname)
port = 8501

tickers = ['AAPL', 'AMZN', 'GOOGL', 'MSFT', 'NVDA']
selected_tickers = st.sidebar.multiselect(
    'Seleccione empresas:',
    tickers,
    default=['AAPL', 'NVDA']
)

date_range = st.sidebar.date_input(
    "Rango de fechas:",
    value=[df['Date'].min(), df['Date'].max()],
    min_value=df['Date'].min(),
    max_value=df['Date'].max()
)

price_type = st.sidebar.radio(
    "Tipo de precio:",
    ['Close', 'Open', 'High', 'Low'],
    horizontal=True
)

# Resumen Visual al inicio (últimos 30 días)
st.subheader("Resumen Visual (últimos 30 días)")
latest_date = df['Date'].max()
summary_df = df[df['Date'] >= latest_date - pd.Timedelta(days=30)]

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

with col1:
    fig1 = go.Figure()
    for ticker in selected_tickers:
        fig1.add_trace(go.Scatter(
            x=summary_df['Date'],
            y=summary_df[f'Close_{ticker}'],
            mode='lines',
            name=ticker
        ))
    fig1.update_layout(title="Evolución de cierre", height=300)
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    last_prices = [summary_df[f'Close_{ticker}'].iloc[-1] for ticker in selected_tickers]
    first_prices = [summary_df[f'Close_{ticker}'].iloc[0] for ticker in selected_tickers]
    pct_change = [(last - first)/first * 100 for last, first in zip(last_prices, first_prices)]
    fig2 = go.Figure(go.Bar(x=selected_tickers, y=pct_change, marker_color='orange'))
    fig2.update_layout(title="Cambio % en 30 días", height=300)
    st.plotly_chart(fig2, use_container_width=True)

with col3:
    vol_totales = [summary_df[f'Volume_{ticker}'].sum() for ticker in selected_tickers]
    fig3 = go.Figure(go.Pie(labels=selected_tickers, values=vol_totales))
    fig3.update_layout(title="Distribución del volumen", height=300)
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    close_cols = [f'Close_{ticker}' for ticker in selected_tickers]
    corr = summary_df[close_cols].corr()
    fig4 = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=selected_tickers,
        y=selected_tickers,
        colorscale='Blues'
    ))
    fig4.update_layout(title="Correlación entre cierres", height=300)
    st.plotly_chart(fig4, use_container_width=True)

# Filtrar datos por fechas
filtered_df = df[(df['Date'] >= pd.to_datetime(date_range[0])) & 
                 (df['Date'] <= pd.to_datetime(date_range[1]))]

if filtered_df.empty or len(selected_tickers) == 0:
    st.warning("No hay datos disponibles para los filtros seleccionados.")
    st.stop()

# Métricas clave
st.header("Métricas Clave")
if len(filtered_df) >= 2:
    cols = st.columns(len(selected_tickers))
    for idx, ticker in enumerate(selected_tickers):
        with cols[idx]:
            latest = filtered_df.iloc[-1]
            previous = filtered_df.iloc[-2]
            close_now = latest[f'Close_{ticker}']
            close_prev = previous[f'Close_{ticker}']
            delta = (close_now - close_prev) / close_prev * 100
            st.metric(
                label=f"Último cierre - {ticker}",
                value=f"${close_now:.2f}",
                delta=f"{delta:.2f}%",
                delta_color="inverse" if delta < 0 else "normal"
            )
else:
    st.info("No hay suficientes datos para mostrar métricas.")

# Evolución de precios
st.header("Evolución de Precios")
fig = go.Figure()
for ticker in selected_tickers:
    fig.add_trace(go.Scatter(
        x=filtered_df['Date'],
        y=filtered_df[f'{price_type}_{ticker}'],
        name=ticker,
        mode='lines'
    ))
fig.update_layout(
    hovermode="x unified",
    height=500,
    yaxis_title="Precio (USD)"
)
st.plotly_chart(fig, use_container_width=True)

# Gráficos de velas
st.header("Análisis Técnico - Gráficos de Velas")
def create_candlestick(ticker):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=filtered_df['Date'],
        open=filtered_df[f'Open_{ticker}'],
        high=filtered_df[f'High_{ticker}'],
        low=filtered_df[f'Low_{ticker}'],
        close=filtered_df[f'Close_{ticker}'],
        name=ticker
    ))
    fig.update_layout(
        title=f'Gráfico de Velas - {ticker}',
        xaxis_rangeslider_visible=False,
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

candle_cols = st.columns(len(selected_tickers))
for idx, ticker in enumerate(selected_tickers):
    with candle_cols[idx]:
        st.plotly_chart(create_candlestick(ticker), use_container_width=True)

# Volumen de trading
st.header("Volumen de Trading")
volume_fig = make_subplots(specs=[[{"secondary_y": True}]])
for ticker in selected_tickers:
    volume_fig.add_trace(go.Bar(
        x=filtered_df['Date'],
        y=filtered_df[f'Volume_{ticker}'],
        name=f'Volumen {ticker}',
        opacity=0.3
    ), secondary_y=False)
volume_fig.update_layout(
    barmode='stack',
    height=400,
    showlegend=True
)
st.plotly_chart(volume_fig, use_container_width=True)

# Sección de resumen
st.header('Resumen de Análisis')
col1, col2 = st.columns(2)
with col1:
    st.subheader('Historial de precios')
    st.dataframe(filtered_df[['Date'] + [f'Close_{ticker}' for ticker in selected_tickers]], use_container_width=True)
with col2:
    st.subheader('Volumen de trading')
    st.dataframe(filtered_df[['Date'] + [f'Volume_{ticker}' for ticker in selected_tickers]], use_container_width=True)

col3, col4 = st.columns(2)
with col3:
    st.subheader('Métricas de rendimiento')
    metrics_data = {
        'Ticker': selected_tickers,
        'Último Cierre': [filtered_df[f'Close_{ticker}'].iloc[-1] for ticker in selected_tickers],
        'Cambio (%)': [
            (filtered_df[f'Close_{ticker}'].iloc[-1] - filtered_df[f'Close_{ticker}'].iloc[-2]) / 
            filtered_df[f'Close_{ticker}'].iloc[-2] * 100 for ticker in selected_tickers
        ]
    }
    st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)
with col4:
    st.subheader('Análisis de tendencias')
    trend_data = {
        'Ticker': selected_tickers,
        'Tendencia': [
            'Al alza' if (filtered_df[f'Close_{ticker}'].iloc[-1] > filtered_df[f'Close_{ticker}'].iloc[-2]) else 'A la baja'
            for ticker in selected_tickers
        ]
    }
    st.dataframe(pd.DataFrame(trend_data), use_container_width=True)

# Análisis de tendencias
st.dataframe(pd.DataFrame(trend_data), use_container_width=True)

# --- Sección: Análisis Espectral por acción ---
st.header("Análisis Espectral por Acción")

st.sidebar.subheader("Opciones del Análisis Espectral")
price_fft_type = st.sidebar.selectbox("Tipo de precio para análisis espectral:", ['Close', 'Open', 'High', 'Low'], index=0)
log_scale = st.sidebar.checkbox("Usar escala logarítmica (Y)", value=False)

if len(selected_tickers) == 0:
    st.warning("Debe seleccionar al menos una acción para mostrar su análisis espectral.")
else:
    for ticker in selected_tickers:
        st.subheader(f"FFT de {ticker} ({price_fft_type})")
        
        # Obtener la señal (precio)
        serie = filtered_df[f'{price_fft_type}_{ticker}'].dropna().values
        N = len(serie)
        if N < 10:
            st.info(f"No hay suficientes datos para calcular FFT de {ticker}.")
            continue

        T = 1  # intervalo de muestreo: 1 día
        yf = fft(serie - np.mean(serie))
        xf = fftfreq(N, T)[:N // 2]
        amplitudes = 2.0 / N * np.abs(yf[0:N // 2])

        # Gráfico
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=xf,
            y=np.log(amplitudes) if log_scale else amplitudes,
            mode='lines',
            name=f'FFT {ticker}'
        ))
        fig.update_layout(
            title=f"Espectro de Frecuencia de {ticker}",
            xaxis_title='Frecuencia (1/día)',
            yaxis_title='Log(Amplitud)' if log_scale else 'Amplitud',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)


st.header("Gráfico de Velas con Patrones Detectados")

for ticker in selected_tickers:
    st.subheader(f"Patrones recientes en {ticker}")

    data = filtered_df[['Date',
                        f'Open_{ticker}', f'High_{ticker}',
                        f'Low_{ticker}', f'Close_{ticker}']].dropna().reset_index(drop=True)

    fig = go.Figure(data=[go.Candlestick(
        x=data['Date'],
        open=data[f'Open_{ticker}'],
        high=data[f'High_{ticker}'],
        low=data[f'Low_{ticker}'],
        close=data[f'Close_{ticker}'],
        increasing_line_color='green',
        decreasing_line_color='red',
        name='Precio'
    )])

    patrones = detectar_patrones(data, ticker)

    for p in patrones[-5:]:  # mostrar solo los últimos 5
        fig.add_trace(go.Scatter(
            x=[p["Date"]],
            y=[p["Precio"] * 1.01],  # ligera separación para visibilidad
            mode='markers',
            marker=dict(size=12, symbol="star", color="orange"),
            hovertext=[f"{p['Tipo']} ({p['Date']})"],
            hoverinfo="text",
            showlegend=False
        ))

    fig.update_layout(
        title=f"Gráfico de Velas + Patrones – {ticker}",
        xaxis_title="Fecha",
        yaxis_title="Precio (USD)",
        xaxis_rangeslider_visible=False,
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    # Mostrar tabla con últimos patrones detectados
    if patrones:
        df_patrones = pd.DataFrame(patrones[-5:])  # solo últimos 5
        st.markdown("### Últimos patrones detectados")
        st.dataframe(df_patrones, use_container_width=True)
    else:
        st.info("No se detectaron patrones en este intervalo.")

    
# Mostrar datos brutos
if st.checkbox("Mostrar datos brutos"):
    st.subheader("Datos Históricos")
    st.dataframe(filtered_df, use_container_width=True)
    
