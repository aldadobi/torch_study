from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
digits = load_digits()
plt.figure(figsize=(20,4))

for index, (image,label) in enumerate(zip(digits.data[0:5], digits.target[0:5])):
      plt.subplot(1,5, index + 1)
      plt.imshow(np.reshape(image, (8,8)), cmap=plt.cm.gray)
      plt.title('Training: %i\n' % label, fontsize = 20)

#plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,test_size=0.25, random_state=0)

from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression()
logisticRegr.fit(X_train, y_train)
#print(X_test[0])
#print(X_test[0].reshape(1,-1))
logisticRegr.predict(X_test[0].reshape(1,-1))
array = logisticRegr.predict(X_test[0:10])
predictions = logisticRegr.predict(X_test)
score = logisticRegr.score(X_test,y_test)
#print(score)
cm = metrics.confusion_matrix(y_test,predictions)
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=False, fmt=".3f", linewidths=.5, square=True, cmap = 'Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size=15)
plt.show()


#print(digits.target.shape)
