import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# 1. Veri setini yükle
iris = load_iris()
X = iris.data
y = iris.target

# 2. Train/test böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. (Opsiyonel) Özellik ölçekleme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. XGBoost sınıflandırıcı oluştur
xgb_model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=3,
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    eval_metric='mlogloss'
)

# 5. Eğit
xgb_model.fit(X_train_scaled, y_train)

# 6. Tahmin
y_pred = xgb_model.predict(X_test_scaled)

# 7. Sonuçları yazdır
print("Classification Report:\n", classification_report(y_test, y_pred))

