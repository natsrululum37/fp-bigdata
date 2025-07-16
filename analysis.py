import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from PIL import Image, ImageDraw, ImageFont

# 1. Pengumpulan & Pembersihan Data
# ================================
file_path = 'dataset/housing.csv'
df = pd.read_csv(file_path)

print('Jumlah baris data:', len(df))
print('Info kolom:')
print(df.info())
print('Cek nilai hilang:')
print(df.isnull().sum())

# Imputasi nilai hilang (mean)
numeric_cols = df.select_dtypes(include=[np.number]).columns
imputer = SimpleImputer(strategy='mean')
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
print('Setelah imputasi, nilai hilang per kolom:')
print(df.isnull().sum())

# 2. Rekayasa Fitur
# =================
df['kamar_per_rumah_tangga'] = df['total_rooms'] / df['households']
df['rasio_kamar_tidur_per_kamar'] = df['total_bedrooms'] / df['total_rooms']

# Simpan dataset bersih & fitur baru
clean_path = 'housing_clean.csv'
df.to_csv(clean_path, index=False)

# 3. EDA & Visualisasi
# ====================
desc = df.describe()
print('Statistik deskriptif:')
print(desc)

# Visualisasi 1: Histogram distribusi harga rumah
plt.figure(figsize=(8,5))
sns.histplot(data=df, x='median_house_value', bins=30, kde=True, color='skyblue')
plt.title('Distribusi Harga Rumah (USD)')
plt.xlabel('Harga Rumah (USD)')
plt.ylabel('Frekuensi')
plt.tight_layout()
plt.savefig('viz1_hist_harga.png')
plt.close()

# Visualisasi 2: Scatter plot kamar per rumah tangga vs harga rumah
plt.figure(figsize=(8,5))
sns.scatterplot(x='kamar_per_rumah_tangga', y='median_house_value', data=df, alpha=0.5)
plt.title('Jumlah Kamar per Rumah Tangga vs Harga Rumah')
plt.xlabel('Jumlah Kamar per Rumah Tangga')
plt.ylabel('Harga Rumah (USD)')
plt.tight_layout()
plt.savefig('viz2_scatter_kamarph_harga.png')
plt.close()

# Visualisasi 3: Heatmap korelasi antar fitur numerik
plt.figure(figsize=(10,8))
corr = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Heatmap Korelasi Fitur')
plt.tight_layout()
plt.savefig('viz3_heatmap_korelasi.png')
plt.close()

# 4. Analisis Korelasi
# ====================
if 'median_house_value' in corr.columns:
    corr_target = corr['median_house_value']
    corr_target = pd.Series(corr_target, index=corr.index)
    corr_target = corr_target.sort_values(ascending=False)
    print('Korelasi fitur dengan harga rumah:')
    print(corr_target)
else:
    print('Kolom median_house_value tidak ditemukan pada matriks korelasi!')

# 5. Model Regresi & Evaluasi
# ===========================
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

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    results[name] = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'cv_r2_mean': np.mean(cv_scores),
        'cv_r2_std': np.std(cv_scores),
        'y_pred': y_pred
    }
    print(f'\nEvaluasi Model {name}:')
    print(f'MSE: {mse:.2f}')
    print(f'RMSE: {rmse:.2f}')
    print(f'MAE: {mae:.2f}')
    print(f'R2: {r2:.2f}')
    print(f'CV R2 Rata-rata: {np.mean(cv_scores):.2f} (std: {np.std(cv_scores):.2f})')

# Visualisasi Prediksi vs Aktual untuk semua model
for name, res in results.items():
    plt.figure(figsize=(8,5))
    plt.scatter(y_test, res['y_pred'], alpha=0.5)
    plt.xlabel('Harga Aktual')
    plt.ylabel('Harga Prediksi')
    plt.title(f'Prediksi vs Aktual ({name})')
    plt.tight_layout()
    img_name = f'viz_prediksi_vs_aktual_{name.lower().replace(" ","")}.png'
    plt.savefig(img_name)
    plt.close()

# 6. Poster Otomatis (PNG)
# ========================
def buat_poster():
    poster = Image.new('RGB', (1200, 2000), color='white')
    draw = ImageDraw.Draw(poster)
    try:
        font_title = ImageFont.truetype('arial.ttf', 60)
        font_sub = ImageFont.truetype('arial.ttf', 36)
        font_text = ImageFont.truetype('arial.ttf', 28)
    except:
        font_title = font_sub = font_text = None
    draw.text((50, 30), 'Analisis Regresi Harga Rumah', fill='black', font=font_title)
    draw.text((50, 120), 'Dataset: California Housing Prices (Kaggle)', fill='black', font=font_sub)
    draw.text((50, 180), 'Statistik Deskriptif:', fill='black', font=font_sub)
    draw.text((50, 230), str(desc[['median_house_value', 'median_income']].round(2)), fill='black', font=font_text)
    draw.text((50, 400), 'Korelasi Fitur:', fill='black', font=font_sub)
    draw.text((50, 450), str(corr_target.round(2)), fill='black', font=font_text)
    draw.text((50, 700), 'Evaluasi Model:', fill='black', font=font_sub)
    y_offset = 750
    for name, res in results.items():
        eval_str = f"{name}: RMSE={res['rmse']:.2f}, R2={res['r2']:.2f}, CV R2={res['cv_r2_mean']:.2f}"
        draw.text((50, y_offset), eval_str, fill='black', font=font_text)
        y_offset += 40
    # Tempel visualisasi
    try:
        img1 = Image.open('viz1_hist_harga.png').resize((500,300))
        img2 = Image.open('viz2_scatter_kamarph_harga.png').resize((500,300))
        img3 = Image.open('viz3_heatmap_korelasi.png').resize((500,300))
        poster.paste(img1, (650, 200))
        poster.paste(img2, (650, 520))
        poster.paste(img3, (650, 840))
        img4 = Image.open('viz_prediksi_vs_aktual_regresilinier.png').resize((500,300))
        poster.paste(img4, (650, 1160))
    except Exception as e:
        print('Gagal menempel gambar ke poster:', e)
    draw.text((50, 1900), 'Sumber data: California Housing Prices, Kaggle (Cam Nugent)', fill='black', font=font_text)
    poster.save('poster_regresi_housing.png')
    print('Poster berhasil disimpan sebagai poster_regresi_housing.png')

buat_poster()
print('Selesai. Semua hasil disimpan di file PNG dan CSV.') 