import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from math import sqrt

#Veri seti ekleme

x = np.array([[6000],[8200],[9000],[14200],[16200]])
y = [86000,82000,78000,75000,70000]

#Data set Görselleştirme
plt.figure()
plt.title("Otomobil fiyat-Km Dağılım Grafiği")
plt.xlabel('KM')
plt.ylabel('Fiyat')
plt.plot(x,y,'k.')
plt.axis([3000,20000,60000,90000])
plt.grid(True)
plt.show()

model = LinearRegression()
model.fit(x,y)
#algoritma model parametrelerine bakalım
model.intercept_
model.coef_

#Model tahmin Doğrusu oluşturma
plt.figure()
plt.title('Otomobil Fiyat-KM Dağılım Grafiği')
plt.xlabel('KM')
plt.ylabel('Fiyat')
plt.plot(x,model.predict(x),color ='red')
plt.plot(x,y,'k.')
plt.grid(True)
plt.show()

# 12.000km lik bir araç fiyat tahmini
test_araba = np.array([[12000]])
predicted_price = model.predict(test_araba)[0]
print("12.000 km de olan arabanın tahmini fiyatı:$%.2f"% predicted_price)

# 9.000 km lik bir aracın fiyat tahmini
test_araba = np.array([[9000]])
predicted_price = model.predict(test_araba)[0]
print("9.000 km de olan arabanın tahmini fiyatı : $%.2f"%predicted_price)

#y_predictions isimli bir değişken oluşturup karşılaştırma yapma

y_predictions = model.predict(x)

for i ,y_prediction in enumerate(y_predictions):
    print('Tahmin Edilen Fiyat: $%.2f , Gerçek Fiyat: $%.2f'%(y_prediction,y[i]))

MAE=mean_absolute_error(y,y_predictions)
MSE=mean_squared_error(y,y_predictions)
R2=r2_score(y,y_predictions)
RMSE = sqrt(mean_squared_error(y,y_predictions))

print("MAE ",MAE)
print("MSE ",MSE)
print("R2 score ",R2)
print("RMSE ",RMSE)
print("Modelin doğruluk oranı %.2f%%" % (R2*100))

#Modelin test veri seti için ne kadar başarılı olduğunu test edelim. Bunun için yeni veri set oluşturalım

x_test = np.array([[1700],[2600],[11000],[14000],[17500]]).reshape(-1,1)
y_test = [94000,94400,73000,83000,75000]

#Graph-1 Eğitim veri seti grafiği
plt.figure()
plt.title('Otomobil fiyat-Km Serpilme Grafiği 1')
plt.xlabel('KM')
plt.ylabel('Fiyat')
plt.plot(x,y,'k.')
plt.axis([0,20000,60000,95000])
plt.grid(True)
plt.show()

#Graph-2 Eğitim Veri seti ve Tahmin Doğrusu
plt.figure()
plt.title('Otomobil fiyat-Km Serpilme Grafiği 2')
plt.xlabel('KM')
plt.ylabel('Fiyat')
plt.plot(x,model.predict(x),color='red')
plt.plot(x,y,'k.',color='black')
plt.axis([0,20000,60000,100000])
plt.grid(True)
plt.show()

#Graph-3 Eğitim ve Test Veri seti (Birlikte)
plt.figure()
plt.title('Otomobil fiyat-Km Serpilme Grafiği 3')
plt.xlabel('KM')
plt.ylabel('Fiyat')
plt.plot(x,y,'k.',color='black')
plt.plot(x_test,y_test,'x',color='black')
plt.axis([0,20000,60000,95000])
plt.grid(True)
plt.show()

#Graph-4 Eğitim ve Test Veri seti ile Eğitim veri setinin tahmin doğrusu
plt.figure()
plt.title('Otomobil fiyat-Km Serpilme Grafiği 4')
plt.xlabel('KM')
plt.ylabel('Fiyat')
plt.plot(x,model.predict(x),color='red')
plt.plot(x,y,'k.',color='black')
plt.plot(x_test,y_test,'x',color='black')
plt.axis([0,20000,60000,95000])
plt.grid(True)
plt.show()

#Graph-5 Tahmin Doğrusu Test Veri setini başarılı öngörebiliyor mu
plt.figure()
plt.title('Otomobil fiyat-Km Serpilme Grafiği 5')
plt.xlabel('KM')
plt.ylabel('Fiyat')
plt.plot(x,model.predict(x),color='red')
plt.plot(x_test,model.predict(x),'---',color='red')
plt.plot(x_test,y_test,'x',color='black')
plt.axis([0,20000,60000,95000])
plt.grid(True)
plt.show()



