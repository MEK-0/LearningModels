
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

#Veri yükleme
iris = load_iris()
x = iris.data[:,:2]
y = iris.target

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2 , random_state=42)

#Ölçekleme
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

#Random Forest modeli
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(x_train_scaled , y_train)

#Sınırları çizme Fonk.
def plot_rf_decision_boundary(model , x , y , title ):
    x_min , x_max = x[:,0].min() - 1 , x[:,0].max() + 1
    y_min , y_max = x[:,1].min() - 1 , x[:,1].max() + 1
    xx , yy = np.meshgrid(np.linspace(x_min,y_max,300),
                         np.linspace(y_min,y_max,300))
    z = model.predict(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)

    plt.figure(figsize=(10,6))
    plt.contourf(xx,yy,z , alpha=0.3, cmap=ListedColormap(['#FFAAAA','#AAFFAA','#AAAAFF']))
    plt.scatter(x[:,0],x[:,1], c=y ,edgecolor='k', cmap=ListedColormap(['#FF0000','#00FF00','#0000FF']))
    plt.xlabel("Feature 1 (Scaled)")
    plt.ylabel("Feature 2 (Scaled)")
    plt.title(title)


#Görselleştirme
plot_rf_decision_boundary(rf_model, x_train_scaled ,y_train,"Random Forest - Decision Boundary")

#Sınıflandırma performansı
print("Random Forest - Classification report:\n")
print(classification_report(y_test , rf_model.predict(x_test_scaled)))

# Özellik önemleri
importances = rf_model.feature_importances_
for name, score in zip(iris.feature_names[:2], importances):
    print(f"{name}: {score:.4f}")


plt.show()
