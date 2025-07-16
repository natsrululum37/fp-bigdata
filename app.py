import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

st.set_page_config(page_title='Dashboard Analisis Harga Rumah', layout='wide')

# Load data dengan pengecekan file
@st.cache_data
def load_data():
    if not os.path.exists('housing_clean.csv'):
        return None
    return pd.read_csv('housing_clean.csv')

df = load_data()

st.title('Dashboard Analisis Regresi Harga Rumah')
st.markdown('**Dataset:** [California Housing Prices (Kaggle)](https://www.kaggle.com/datasets/camnugent/california-housing-prices)')
st.markdown('Sumber: Cam Nugent, Kaggle')
st.markdown('---')

if df is None:
    st.error('File housing_clean.csv tidak ditemukan. Silakan jalankan analysis.py terlebih dahulu.')
    st.stop()

# Sidebar
st.sidebar.header('Navigasi')
menu = st.sidebar.radio('Pilih Menu:', ['EDA & Visualisasi', 'Korelasi', 'Prediksi Harga Rumah', 'Evaluasi Model', 'Download'])

# EDA & Visualisasi
if menu == 'EDA & Visualisasi':
    st.header('Eksplorasi Data Awal')
    st.write('Statistik Deskriptif:')
    st.dataframe(df.describe().T)
    st.write('Distribusi Harga Rumah:')
    fig1, ax1 = plt.subplots()
    sns.histplot(data=df, x='median_house_value', bins=30, kde=True, color='skyblue', ax=ax1)
    ax1.set_xlabel('Harga Rumah (USD)')
    ax1.set_ylabel('Frekuensi')
    st.pyplot(fig1)
    st.write('Scatter Plot: Jumlah Kamar per Rumah Tangga vs Harga Rumah')
    fig2, ax2 = plt.subplots()
    sns.scatterplot(x='kamar_per_rumah_tangga', y='median_house_value', data=df, alpha=0.5, ax=ax2)
    ax2.set_xlabel('Jumlah Kamar per Rumah Tangga')
    ax2.set_ylabel('Harga Rumah (USD)')
    st.pyplot(fig2)
    st.write('Heatmap Korelasi Fitur Numerik')
    fig3, ax3 = plt.subplots(figsize=(10,8))
    corr = df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax3)
    st.pyplot(fig3)

# Korelasi
elif menu == 'Korelasi':
    st.header('Analisis Korelasi')
    corr = df.select_dtypes(include=[np.number]).corr()
    if 'median_house_value' in corr.columns:
        target_series = pd.Series(corr['median_house_value'])
        corr_target = target_series.sort_values(ascending=False)
        st.write('Korelasi fitur dengan harga rumah:')
        st.dataframe(corr_target)
        st.write('Fitur dengan korelasi tertinggi:')
        if len(corr_target) > 1:
            st.success(f"{corr_target.index[1]}: {corr_target.iloc[1]:.2f}")
    else:
        st.error('Kolom median_house_value tidak ditemukan pada matriks korelasi!')

# Prediksi Harga Rumah
elif menu == 'Prediksi Harga Rumah':
    st.header('Prediksi Harga Rumah')
    st.write('Masukkan nilai fitur untuk prediksi harga rumah:')
    median_income = st.number_input('Pendapatan Median (dalam puluhan ribu USD)', min_value=0.0, value=float(df['median_income'].median()))
    housing_median_age = st.number_input('Usia Median Rumah (tahun)', min_value=0.0, value=float(df['housing_median_age'].median()))
    total_rooms = st.number_input('Total Jumlah Kamar', min_value=0.0, value=float(df['total_rooms'].median()))
    total_bedrooms = st.number_input('Total Jumlah Kamar Tidur', min_value=0.0, value=float(df['total_bedrooms'].median()))
    kamar_per_rumah_tangga = st.number_input('Jumlah Kamar per Rumah Tangga', min_value=0.0, value=float(df['kamar_per_rumah_tangga'].median()))
    rasio_kamar_tidur_per_kamar = st.number_input('Rasio Kamar Tidur per Kamar', min_value=0.0, value=float(df['rasio_kamar_tidur_per_kamar'].median()))
    fitur = pd.DataFrame(
        [[median_income, housing_median_age, total_rooms, total_bedrooms, kamar_per_rumah_tangga, rasio_kamar_tidur_per_kamar]],
        columns=pd.Index(['median_income', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'kamar_per_rumah_tangga', 'rasio_kamar_tidur_per_kamar'])
    )
    # Model selection
    model_option = st.selectbox('Pilih Model:', ['Regresi Linier', 'Ridge', 'Lasso', 'Decision Tree'])
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.tree import DecisionTreeRegressor
    models = {
        'Regresi Linier': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=42)
    }
    X = df[['median_income', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'kamar_per_rumah_tangga', 'rasio_kamar_tidur_per_kamar']]
    y = df['median_house_value']
    model = models[model_option].fit(X, y)
    pred = model.predict(fitur)[0]
    st.success(f'Prediksi Harga Rumah ({model_option}): ${pred:,.2f}')

# Evaluasi Model
elif menu == 'Evaluasi Model':
    st.header('Evaluasi Model Regresi')
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.tree import DecisionTreeRegressor
    features = ['median_income', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'kamar_per_rumah_tangga', 'rasio_kamar_tidur_per_kamar']
    X = df[features]
    y = df['median_house_value']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models = {
        'Regresi Linier': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=42)
    }
    eval_table = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        eval_table.append({
            'Model': name,
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'CV R2 Rata-rata': np.mean(cv_scores),
            'CV R2 Std': np.std(cv_scores)
        })
    st.write('Tabel Evaluasi Model:')
    st.dataframe(pd.DataFrame(eval_table).set_index('Model').round(3))
    # Visualisasi Prediksi vs Aktual
    vis_model = st.selectbox('Pilih Model untuk Visualisasi:', ['Regresi Linier', 'Ridge', 'Lasso', 'Decision Tree'])
    img_map = {
        'Regresi Linier': 'regresilinier',
        'Ridge': 'ridge',
        'Lasso': 'lasso',
        'Decision Tree': 'decisiontree'
    }
    img_path = f'viz_prediksi_vs_aktual_{img_map[vis_model]}.png'
    try:
        st.image(img_path, caption=f'Prediksi vs Aktual ({vis_model})')
    except Exception:
        st.warning('Gambar visualisasi belum tersedia.')

# Download
elif menu == 'Download':
    st.header('Download Hasil')
    if os.path.exists('poster_regresi_housing.png'):
        with open('poster_regresi_housing.png', 'rb') as f:
            st.download_button('Download Poster (.png)', f, file_name='poster_regresi_housing.png')
    else:
        st.warning('Poster belum tersedia. Jalankan analysis.py untuk membuat poster.')
    if os.path.exists('housing_clean.csv'):
        with open('housing_clean.csv', 'rb') as f:
            st.download_button('Download Dataset Bersih (.csv)', f, file_name='housing_clean.csv')
    else:
        st.warning('Dataset bersih belum tersedia. Jalankan analysis.py untuk membuat dataset.')
    st.info('Poster dan dataset bersih siap diunduh untuk laporan.') 