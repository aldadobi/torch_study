import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

X = pd.read_csv('/home/yeom/Desktop/youngeon/code/study/torchtutorial/chap03/data/credit card.csv')
#print(X.head())
X = X.drop('CUST_ID', axis=1)
X.fillna(method='ffill', inplace=True)
#print(X.head())
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
#print(X_scaled)
X_normalized = normalize(X_scaled)

X_normalized = pd.DataFrame(X_normalized)


#print(X_normalized.shape)
pca = PCA(n_components=2)
X_principal = pca.fit_transform(X_normalized)
X_principal = pd.DataFrame(X_principal)
X_principal.columns = ['P1', 'P2']

db_default = DBSCAN(eps= 0.0375, min_samples= 50).fit(X_principal)
labels = db_default.labels_
colours = {}
colours[0] = 'r'
colours[1] = 'g'
colours[2] = 'b'
colours[3] = 'c'
colours[4] = 'y'
colours[5] = 'm'
colours[6] = 'black'
colours[-1] = 'black'
cvec = [colours[label] for label in labels]

r = plt.scatter(
    X_principal['P1'], X_principal['P2'], marker ='o', color = colours[0]) 
g = plt.scatter( 
    X_principal['P1'], X_principal['P2'], marker ='o', color = colours[1]) 
b = plt.scatter( 
    X_principal['P1'], X_principal['P2'], marker ='o', color = colours[2]) 
c = plt.scatter( 
    X_principal['P1'], X_principal['P2'], marker ='o', color = colours[3]) 
y = plt.scatter( 
    X_principal['P1'], X_principal['P2'], marker ='o', color = colours[4]) 
m = plt.scatter( 
    X_principal['P1'], X_principal['P2'], marker ='o', color = colours[5]) 
k = plt.scatter( 
    X_principal['P1'], X_principal['P2'], marker ='o', color = colours[6]) 

plt.figure(figsize =(9, 9)) 
plt.scatter(X_principal['P1'], X_principal['P2'], c = cvec) 

plt.legend((r, g, b, c, y, m, k), 
           ('Label 0', 'Label 1', 'Label 2', 'Label 3', 'Label 4', 'Label 5', 'Label -1'), 
           scatterpoints = 1, 
           loc ='upper left', 
           ncol = 3, 
           fontsize = 8) 

plt.show()
