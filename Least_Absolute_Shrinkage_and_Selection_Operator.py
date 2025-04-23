################Lasso Regresyonu################
import math
import numpy as np
import pandas as pd
import seaborn as seabornInstance
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({'font.size':12})
from sklearn.linear_model import Lasso

#veri seti indirme
data_path = '/Users/mahmutesat/Desktop/Machine-learning/SATILIK_EV1.xlsx'
veri = pd.read_excel(data_path, engine='openpyxl')

#Hedef ve Öznitelikleri belirleme
x = veri[['Oda_Sayısı','Net_m2','Katı','Yaşı']]
y = veri['Fiyat']

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#Lasso Algoritmasını eğitme kısmı
lasso = Lasso()
lasso.fit(x_train,y_train)
train_score =lasso.score(x_train,y_train)
test_score = lasso.score(x_test,y_test)
coeff_used = np.sum(lasso.coef_ !=0)

print('Eğitim verisi için R2:',train_score) #0.73
print('Test verisi için R2:',test_score) #0.68
print('Kullanılan Öznitelik Sayısı (Ağırlığı sıfırdan büyük):',coeff_used) # 4


#Lasso (alpha=0.01)
lasso001 = Lasso(alpha=0.01,max_iter=1000000)
lasso001.fit(x_train,y_train)

train_score001 = lasso001.score(x_train,y_train)
test_score001 = lasso001.score(x_test,y_test)
coeff_used001 = np.sum(lasso001.coef_ !=0)

#Lasso (alpha=0.0001)
lasso00001 = Lasso(alpha=0.0001,max_iter=1000000)
lasso00001.fit(x_train,y_train)

train_score00001 = lasso00001.score(x_train,y_train)
test_score00001 = lasso00001.score(x_test,y_test)
coeff_used00001 = np.sum(lasso00001.coef_ !=0)


#Bi de bunu Liner Reg ile kıyaslayalım
lr = LinearRegression()
lr.fit(x_train,y_train)
lr_train_score = lr.score(x_train,y_train)
lr_test_score = lr.score(x_test,y_test)

print('Eğitim setinin R2 :',lr_train_score)
print('Test Setinin R2 :',lr_test_score)

#Grafikleştirme
# Grafikleştirme
plt.figure(figsize=(12, 6))

# İlk subplot - Lasso (alpha=1) ve Doğrusal Regresyon
plt.subplot(1, 2, 1)
plt.plot((1, 2, 3, 4), lr.coef_, alpha=0.7, linestyle='none', marker='o',
         markersize=15, color='orange', label='Doğrusal Reg', zorder=2)
plt.plot((1, 2, 3, 4), lasso.coef_, alpha=1, linestyle='none', marker='x',
         markersize=10, color='black', label=r'Lasso: $\alpha = 1$', zorder=7)
plt.plot(0, lr.intercept_, alpha=0.7, linestyle='none', marker='o',
         markersize=15, color='orange')
plt.plot(0, lasso.intercept_, alpha=1, linestyle='none', marker='x',
         markersize=10, color='black')
plt.xticks([0, 1, 2, 3, 4], ('Sabit_Terim', 'Oda_Sayısı', 'Net_m2', 'Katı', 'Yaşı'),
           rotation=70)
plt.xlabel('Öznitelikler', fontsize=13)
plt.ylabel('Öznitelik Katsayıları', fontsize=13)
plt.title('Lasso vs Linear Regression (alpha=1)', fontsize=14)
plt.legend(fontsize=11, loc='upper right')

# İkinci subplot - Lasso (alpha=0.01) ve Doğrusal Regresyon
plt.subplot(1, 2, 2)
plt.plot((1, 2, 3, 4), lr.coef_, alpha=0.7, linestyle='none', marker='o',
         markersize=15, color='orange', label='Doğrusal Reg', zorder=2)
plt.plot((1, 2, 3, 4), lasso001.coef_, alpha=1, linestyle='none', marker='x',
         markersize=10, color='blue', label=r'Lasso: $\alpha = 0.01$', zorder=7)
plt.plot(0, lr.intercept_, alpha=0.7, linestyle='none', marker='o',
         markersize=15, color='orange')
plt.plot(0, lasso001.intercept_, alpha=1, linestyle='none', marker='x',
         markersize=10, color='blue')
plt.xticks([0, 1, 2, 3, 4], ('Sabit_Terim', 'Oda_Sayısı', 'Net_m2', 'Katı', 'Yaşı'),
           rotation=70)
plt.xlabel('Öznitelikler', fontsize=13)
plt.ylabel('Öznitelik Katsayıları', fontsize=13)
plt.title('Lasso vs Linear Regression (alpha=0.01)', fontsize=14)
plt.legend(fontsize=11, loc='upper right')

plt.tight_layout()
plt.show()


# NOT grafikte kullanılan alpha değeri şeffaflaştırma içindir
#Model parametresi değil yani

#lasso (alpha=1 ve 0,001) ile çoklu doğrusal reg. parametreleri
print('line 105 lasso.coef_ :',lasso.coef_)
print('line 106 lasso00001.coef_ :',lasso00001.coef_)
print('line 107 lr.coef_ :',lr.coef_)

