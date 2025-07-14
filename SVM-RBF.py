from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report , confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
# veri seti yükleme
iris = load_iris()
x = iris.data
y = iris.target

#veri ayırma
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2 , random_state=42)

# Özellik ölçekleme
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#SVM model oluşturma
svm_model = SVC(C=1.0,kernel='rbf',gamma='scale')
svm_model.fit(x_train,y_train)


#Tahmin
y_pred = svm_model.predict(x_test)

print("Sınıflandırma Raporu:\n",classification_report(y_test,y_pred))
print("Karmaşıklık matrisi:\n",confusion_matrix(y_test,y_pred))


x_vis = x[:,:2]
x_train_vis , x_test_vis , y_train_vis , y_test_vis = train_test_split(x_vis,y,test_size=0.2 ,random_state=42)
x_train_vis = scaler.fit_transform(x_train_vis)
x_test_vis = scaler.transform(x_test_vis)

svm_vis = SVC(kernel='rbf',C=1.0, gamma='scale')
svm_vis.fit(x_train_vis,y_train_vis)

#Meshgrid
x_min , x_max = x_train_vis[:,0].min() - 1, x_train_vis[:,0].max() + 1
y_min , y_max = x_train_vis[:,1].min() - 1, x_train_vis[:,1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max,300),
                     np.linspace(y_min,y_max,300))

Z = svm_vis.predict(np.c_[xx.ravel(),yy.ravel()])
Z = Z.reshape(xx.shape)

#PLOT

plt.figure(figsize=(10,6))
plt.contourf(xx,yy,Z ,alpha=0.3, cmap=ListedColormap(['#FFAAAA','#AAFFAA','#AAAAFF']))
plt.scatter(x_train_vis[:,0],x_train_vis[:,1], c=y_train_vis, edgecolors='k',cmap=ListedColormap(["#FF0000",'#00FF00','#0000FF']))
plt.title("SVM (RBF KERNEL) - Decision Limit")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
