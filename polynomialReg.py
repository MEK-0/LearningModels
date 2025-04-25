import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

# Excel dosyasını yükleme
data_path = '/Users/mahmutesat/Desktop/Machine-learning/SATILIK_EV1.xlsx'
df = pd.read_excel(data_path, engine='openpyxl')

# Özellik (X) ve hedef değişken (y) seçimi
x = df[['Oda_Sayısı', 'Net_m2', 'Katı', 'Yaşı']]
y = df['Fiyat']

# Eğitim ve test verisi olarak ayırma
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Polinom derecesini belirlemek için döngü
rmses = []
degrees = np.arange(1, 10)
min_rmse, min_deg = 1e10, 0

for deg in degrees:
    poly_features = PolynomialFeatures(degree=deg, include_bias=False)
    x_poly_train = poly_features.fit_transform(x_train)

    poly_reg = LinearRegression()
    poly_reg.fit(x_poly_train, y_train)

    x_poly_test = poly_features.transform(x_test)
    poly_predict = poly_reg.predict(x_poly_test)

    poly_mse = mean_squared_error(y_test, poly_predict)
    poly_rmse = np.sqrt(poly_mse)
    rmses.append(poly_rmse)

    if min_rmse > poly_rmse:
        min_rmse = poly_rmse
        min_deg = deg

# En iyi derece ve hata skoru yazdırma
print('En iyi Model {:.2f} RMSE hata skorunu veren {} polinom derecesi ile sağlanıyor'.format(min_rmse, min_deg))

# RMSE grafiği çizdirme
plt.style.use('fivethirtyeight')
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.plot(degrees, rmses, marker='o')
ax.set_yscale('log')
ax.set_xlabel('Polinom Derecesi')
ax.set_ylabel('RMSE')
plt.text(min_deg + 0.5, min_rmse * 1.2, f'En iyi derece: {min_deg}\nRMSE: {min_rmse:.2f}',
         style='italic', bbox={'facecolor': 'orange', 'alpha': 0.5, 'pad': 10})
plt.title('Polinom Derecelerine Göre RMSE')
plt.grid(True)


# En iyi derece ile yeniden model eğitme
polinom_derecesi = PolynomialFeatures(degree=min_deg, include_bias=False)
x_train_polinom = polinom_derecesi.fit_transform(x_train)
x_test_polinom = polinom_derecesi.transform(x_test)

# Polinom Regresyon Modeli
polinom_regresyon = LinearRegression()
polinom_regresyon.fit(x_train_polinom, y_train)

# Doğrusal Regresyon Modeli (Karşılaştırmak için)
dogrusal_model = LinearRegression()
dogrusal_model.fit(x_train, y_train)

# Model Performansları
print('\n--- Model Performansları ---')
print('Doğrusal Regresyon - Eğitim R2:', dogrusal_model.score(x_train, y_train))
print('Doğrusal Regresyon - Test R2:', dogrusal_model.score(x_test, y_test))
print('Polinom Regresyon - Eğitim R2:', polinom_regresyon.score(x_train_polinom, y_train))
print('Polinom Regresyon - Test R2:', polinom_regresyon.score(x_test_polinom, y_test))

# Eğitim verisi üzerinde tahminler ve gerçek değer karşılaştırması
print("\n--- Eğitim Verisi: Tahmin vs Gerçek ---")
y_pred_train = polinom_regresyon.predict(x_train_polinom)
for i in range(len(y_pred_train)):
    print(f"Tahmin Edilen: ${y_pred_train[i]:.2f}, Gerçek: ${y_train.values[i]}")

# Test verisi üzerinde tahminler ve gerçek değer karşılaştırması
print("\n--- Test Verisi: Tahmin vs Gerçek ---")
y_pred_test = polinom_regresyon.predict(x_test_polinom)
for i in range(len(y_pred_test)):
    print(f"Tahmin Edilen: ${y_pred_test[i]:.2f}, Gerçek: ${y_test.values[i]}")

# Tüm veriyi kullanarak tahmin sütunu oluşturma
x_poly = polinom_derecesi.transform(x)
df['Fiyat_Tahmini'] = polinom_regresyon.predict(x_poly)

# Tahmin ve gerçek fiyatların karşılaştırmalı grafiği
plt.figure(figsize=(12, 6))
plt.style.use('fivethirtyeight')
plt.plot(df['Fiyat'].values, 'g^', label='Gerçek Fiyat')
plt.plot(df['Fiyat_Tahmini'].values, 'ro', label='Tahmin Edilen Fiyat')
plt.xlabel('İndekss')
plt.ylabel('Fiyat')
plt.title('Gerçek ve Tahmin Edilen Fiyatlar')
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()
