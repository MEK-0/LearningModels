import lightgbm as lgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import pandas as pd

iris = load_iris()
x = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

x_train , x_test , y_train , y_test = train_test_split(x,y, test_size=0.2 , random_state=42)

#Ölçekleme
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

#LightGBM modeli
lgb_model = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=3,
    boosting_type='gbdt',
    learning_rate=0.1,
    n_estimators=100,
    max_depth=3,
    random_state=42
)

#Eğitme
lgb_model.fit(x_train_scaled,y_train)

#Tahmin ve Rapor
y_pred = lgb_model.predict(x_test_scaled)
print("Classification Report:\n",classification_report(y_test,y_pred))

