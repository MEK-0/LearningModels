from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , classification_report

messages = [
    "Win money now!!!",
    "Your loan is approved",
    "Hi, how are you?",
    "Call your mom",
    "Limited offer just for you",
    "Meeting at 10am",
    "Click here to claim your prize",
    "Don't forget the appointment",
    "Get cheap meds online",
    "Let's catch up soon"
]

labels = [1, 1, 0, 0, 1, 0, 1, 0, 1, 0]  # 1 = spam, 0 = not spam

#Metinleri sayısal verilere dönüştürme

vectorizer = CountVectorizer()
x = vectorizer.fit_transform(messages)

x_train ,x_test , y_train , y_test = train_test_split(x,labels ,test_size=0.3, random_state=42)

#Model oluşturma

model = MultinomialNB()
model.fit(x_train,y_train)

#Tahmin
y_pred = model.predict(x_test)

#Sonuçlar
print("Accuracy",accuracy_score(y_test,y_pred))
print("Classification Report:\n",classification_report(y_test,y_pred))

#Örnek tahmin
sample = ["Congratulations! You've won a $1000 Walmart gift card."]
sample_vec = vectorizer.transform(sample)
print("Sample Prediction:",model.predict(sample_vec))

#Düzgün çalışmöıyor veri seti kötü az 
