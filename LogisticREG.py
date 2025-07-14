import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

#Veri üretimi
x,y = make_classification(n_samples=100,n_features=2,n_classes=2,n_redundant=0,random_state=42)

#model oluşturma
model = LogisticRegression()
model.fit(x,y)

#karar sınırını çizme
x_min ,x_max = x[:,0].min() , x[:,0].max()
y_min ,y_max = x[:,1].min() , x[:,1].max()

xx,yy = np.meshgrid(np.linspace(x_min,x_max,100),
                    np.linspace(y_min,y_max,num=100))

z = model.predict(np.c_[xx.ravel(),yy.ravel()])
z = z.reshape(xx.shape)

plt.contourf(xx,yy,z,alpha=0.4)
plt.scatter(x[:,0],x[:,1],c=y)
plt.title("logistic regression decision boundary")
plt.show()
