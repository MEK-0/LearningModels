import pandas as pd
from catboost import CatBoostClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# 1. data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Ölçekleme
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

# 4. CatBoost modeli
cat_model = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=3,
    loss_function='MultiClass',
    verbose=0,
    random_state=42
)

# 5. Eğit
cat_model.fit(X_train_scaled, y_train)

# 6. Tahmin ve performans
y_pred = cat_model.predict(X_test_scaled)
print("Classification Report:\n", classification_report(y_test, y_pred))
