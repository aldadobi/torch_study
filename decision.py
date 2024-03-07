import pandas as pd

df = pd.read_csv('/home/yeom/Desktop/youngeon/code/study/torchtutorial/chap03/data/titanic/train.csv')

#print(df.head())
df = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']]
df['Sex'] = df['Sex'].map({'male' : 0, 'female' : 1})
df = df.dropna()
X = df.drop('Survived', axis = 1)
y = df['Survived']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score
acc_score = accuracy_score(y_test, y_pred)

from sklearn.metrics import confusion_matrix
r = pd.DataFrame(
      confusion_matrix(y_test, y_pred),
      columns = ['Predicted Not Survival', 'Predicted Survival'],
      index = ['True not Survival', 'True Survival']
)
print(r) 
