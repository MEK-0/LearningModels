
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report , confusion_matrix
from matplotlib.colors import ListedColormap

#Veri seti yükleme
iris = load_iris()
x = iris.data
y = iris.target

#Veri kümeleme eğitim
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#özellik ölçeklendirme
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#KNN modelini oluşturma
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)

#tahmin
y_pred = knn.predict(x_test)


#performans değerlendirme
print("Sınıflandırma raporu: \n", classification_report(y_test,y_pred))
print("Karışıklık matrisi: \n",confusion_matrix(y_test,y_pred))

#Karar sınırı görselleştirme
x_vis = x[:,:2]
x_train_vis,x_test_vis,y_train_vis,y_test_vis = train_test_split(x_vis,y,test_size=0.2,random_state=42)
scaler = StandardScaler()
x_train_vis = scaler.fit_transform(x_train_vis)
x_test_vis = scaler.transform(x_test_vis)

# model
knn_vis = KNeighborsClassifier(n_neighbors=5)
knn_vis.fit(x_train_vis,y_train_vis)

#Meshgrid oluşturma
x_min , x_max = x_train_vis[:,0].min() - 1 , x_train_vis[:,0].max() + 1
y_min , y_max = x_train_vis[:,1].min() - 1 , x_train_vis[:,1].max() + 1
xx , yy = np.meshgrid(np.linspace(x_min,x_max,200),
                      np.linspace(y_min,y_max,200))

z = knn_vis.predict(np.c_[xx.ravel(),yy.ravel()])
z = z.reshape(xx.shape)

#plot
plt.figure(figsize=(10,6))
plt.contourf(xx,yy,z,cmap=ListedColormap(['#FFAAAA','#AAFFAA','#AAAAFF']), alpha=0.5)
plt.scatter(x_train_vis[:,0],x_train_vis[:,1], c=y_train_vis, edgecolors='k',cmap=ListedColormap(['#FF0000','#00FF00','#0000FF']))
plt.title("K-NN Karar sınırları - ilk 2 özellik")
plt.xlabel("özellik 1")
plt.ylabel("özellik 2")
plt.show()

