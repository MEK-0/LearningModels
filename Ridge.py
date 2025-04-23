################Ridge Regresyonu################
import math
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({'font.size':12})

#veri seti indirme
data_path = '/Users/mahmutesat/Desktop/Machine-learning/SATILIK_EV1.xlsx'
veri = pd.read_excel(data_path, engine='openpyxl')

#Hedef ve Öznitelikleri belirleme
x = veri[['Oda_Sayısı','Net_m2','Katı','Yaşı']]
y = veri['Fiyat']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Ridge (alpha=1)
ridge = Ridge(alpha=1)
ridge.fit(x_train, y_train)
train_score = ridge.score(x_train, y_train)
test_score = ridge.score(x_test, y_test)
coeff_used = np.sum(ridge.coef_ != 0)

print('Eğitim verisi için R2 (Ridge, alpha=1):', train_score)
print('Test verisi için R2 (Ridge, alpha=1):', test_score)
print('Kullanılan Öznitelik Sayısı (Ağırlığı sıfırdan büyük):', coeff_used)

# Ridge (alpha=0.01)
ridge001 = Ridge(alpha=0.01)
ridge001.fit(x_train, y_train)
train_score001 = ridge001.score(x_train, y_train)
test_score001 = ridge001.score(x_test, y_test)
coeff_used001 = np.sum(ridge001.coef_ != 0)

# Ridge (alpha=0.0001)
ridge00001 = Ridge(alpha=0.0001)
ridge00001.fit(x_train, y_train)
train_score00001 = ridge00001.score(x_train, y_train)
test_score00001 = ridge00001.score(x_test, y_test)
coeff_used00001 = np.sum(ridge00001.coef_ != 0)

# Linear Regression ile karşılaştıralım
lr = LinearRegression()
lr.fit(x_train, y_train)
lr_train_score = lr.score(x_train, y_train)
lr_test_score = lr.score(x_test, y_test)

print('Eğitim setinin R2 (Linear Reg):', lr_train_score)
print('Test Setinin R2 (Linear Reg):', lr_test_score)

# Grafikleştirme
plt.figure(figsize=(12, 6))

# Subplot 1 - Ridge (alpha=1) ve Linear Regression
plt.subplot(1, 2, 1)
plt.plot((1, 2, 3, 4), lr.coef_, alpha=0.7, linestyle='none', marker='o',
         markersize=15, color='orange', label='Doğrusal Reg', zorder=2)
plt.plot((1, 2, 3, 4), ridge.coef_, alpha=1, linestyle='none', marker='x',
         markersize=10, color='black', label=r'Ridge: $\alpha = 1$', zorder=7)
plt.plot(0, lr.intercept_, alpha=0.7, linestyle='none', marker='o',
         markersize=15, color='orange')
plt.plot(0, ridge.intercept_, alpha=1, linestyle='none', marker='x',
         markersize=10, color='black')
plt.xticks([0, 1, 2, 3, 4], ('Sabit_Terim', 'Oda_Sayısı', 'Net_m2', 'Katı', 'Yaşı'),
           rotation=70)
plt.xlabel('Öznitelikler', fontsize=13)
plt.ylabel('Öznitelik Katsayıları', fontsize=13)
plt.title('Ridge vs Linear Regression (alpha=1)', fontsize=14)
plt.legend(fontsize=11, loc='upper right')

# Subplot 2 - Ridge (alpha=0.01) ve Linear Regression
plt.subplot(1, 2, 2)
plt.plot((1, 2, 3, 4), lr.coef_, alpha=0.7, linestyle='none', marker='o',
         markersize=15, color='orange', label='Doğrusal Reg', zorder=2)
plt.plot((1, 2, 3, 4), ridge001.coef_, alpha=1, linestyle='none', marker='x',
         markersize=10, color='blue', label=r'Ridge: $\alpha = 0.01$', zorder=7)
plt.plot(0, lr.intercept_, alpha=0.7, linestyle='none', marker='o',
         markersize=15, color='orange')
plt.plot(0, ridge001.intercept_, alpha=1, linestyle='none', marker='x',
         markersize=10, color='blue')
plt.xticks([0, 1, 2, 3, 4], ('Sabit_Terim', 'Oda_Sayısı', 'Net_m2', 'Katı', 'Yaşı'),
           rotation=70)
plt.xlabel('Öznitelikler', fontsize=13)
plt.ylabel('Öznitelik Katsayıları', fontsize=13)
plt.title('Ridge vs Linear Regression (alpha=0.01)', fontsize=14)
plt.legend(fontsize=11, loc='upper right')

plt.tight_layout()
plt.show()

# Model parametreleri
print('ridge.coef_ (alpha=1):', ridge.coef_)
print('ridge00001.coef_ (alpha=0.0001):', ridge00001.coef_)
print('lr.coef_:', lr.coef_)
