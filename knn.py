import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics

names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
dataset = pd.read_csv('/home/yeom/Desktop/youngeon/code/study/torchtutorial/chap03/data/iris.data', names=names)

X = dataset.iloc[:, :-1].values # 모든 행과 마지막 열 바로 앞까지의 모든 열
y = dataset.iloc[:, 4].values # 모든 행과 4번째 열

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

from sklearn.neighbors import KNeighborsClassifier
'''
knn = KNeighborsClassifier(n_neighbors=50)
knn.fit(X_train,y_train)


y_pred = knn.predict(X_test)
print("정확도 : {}".format(accuracy_score(y_test,y_pred)))
'''
from sklearn.metrics import accuracy_score
k = 10
acc_array = np.zeros(k)
#print(type(acc_array)) 참고로 리스트는 넘파이 배열과는 다른 data type
for k in np.arange(1, k + 1):
      classifier = KNeighborsClassifier(n_neighbors=k).fit(X_train,y_train)
      y_pred = classifier.predict(X_test)
      acc = accuracy_score(y_test, y_pred)
      acc_array[k-1] = acc

max_acc = np.amax(acc_array)
acc_list = list(acc_array)
k = acc_list.index(max_acc)
print("정확도", max_acc, "으로 최적의 k는", k+1)