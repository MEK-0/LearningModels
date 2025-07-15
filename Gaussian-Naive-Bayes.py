from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report , confusion_matrix

#Veri yükleme
iris = load_iris()
x = iris.data
y = iris.target

x_train , x_test , y_train , y_test = train_test_split(x,y , test_size=0.2, random_state=42)

#Özellik Ölçeklendirme
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

#Gaussian Naive Bayes modeli
nb_model = GaussianNB()
nb_model.fit(x_train_scaled,y_train)

#Tahmin
y_pred = nb_model.predict(x_test_scaled)

#Performans değerlendirme
print("Classification Report:\n",classification_report(y_test,y_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test, y_pred))
