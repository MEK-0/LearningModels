import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from math import sqrt

#Dataset

data ={
  'Tarih': ['2024-01-05', '2024-01-06', '2024-01-08', '2024-01-09', '2024-01-10',
              '2024-01-12', '2024-01-14', '2024-01-15', '2024-01-17', '2024-01-18'],
  'Urun': ['Lavanta', 'Kahve', 'Vanilya', 'Lavanta', 'Vanilya',
             'Kahve', 'Lavanta', 'Kahve', 'Vanilya', 'Lavanta'],
  'Adet': [12, 7, 15, 10, 5, 11, 14, 9, 12, 8],
  'Birim_Fiyat': [40, 45, 38, 40, 38, 45, 40, 45, 38, 40]
 }

df = pd.DataFrame(data)

#Toplam gelir sütunu ekleme
df['Toplam_Gelir']=df['Adet']*df['Birim_Fiyat']

print('Veri Seti: \n',df)

#Model için X ve Y belirlememiz gerekiyor

x = df[['Adet']]
y =df[['Toplam_Gelir']]

#Liner Regresyon model

model= LinearRegression()
model.fit(x,y)

#Tahminler
y_pred=model.predict(x)

#Model Parametreleri
print("\n Eğim:",model.coef_[0])
print("Sabit(intercept)",model.intercept_)

#ilk olarak grafik üzerinde gösterim
plt.figure()
plt.title('Adet - Toplam Gelir Grafiği')
plt.xlabel('Adet')
plt.ylabel('Gelir')
plt.plot(x,y,'k.')
plt.axis([5, 16, 100, 700])
plt.grid(True)


#Model tahmin Doğrusu oluşturma
plt.figure()
plt.title('Model Tahmin Doğrusu grafiği')
plt.xlabel('Adet')
plt.ylabel('Gelir')
plt.plot(x,y,'k.')
plt.plot(x,model.predict(x),color='blue')
plt.axis([5, 16, 100, 700])
plt.grid(True)
plt.show()

#Analiz kısmı
R2 =r2_score(y,y_pred)
MSE = mean_squared_error(y,y_pred)
MAE = mean_absolute_error(y,y_pred)
print("Doğruluk oranı",(R2*100))
print("MSE",MSE)
print("MAE",MAE)


x = int(input("Tahmini satış adetini giriniz: "))  
test_x = np.array([[x]])  
predicted_price = model.predict(test_x)[0][0] 

print(f"{x} adet satıştan kazanılan tahmini gelir: {predicted_price:.2f} TL")
