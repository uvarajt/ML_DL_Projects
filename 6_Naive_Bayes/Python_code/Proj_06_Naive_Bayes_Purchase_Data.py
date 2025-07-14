import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

purchaseData = pd.read_csv('Purchase_Logistic.csv')
X = purchaseData.iloc[:, [2, 3]].values
Y = purchaseData.iloc[:, 4].values

scaler = StandardScaler();
X = scaler.fit_transform(X)

Xtrain, Xtest, Ytrain, Ytest \
    = train_test_split(X, Y, test_size = 0.25, random_state = 0)
    
cf = GaussianNB()
cf.fit(Xtrain, Ytrain)
Ypred = cf.predict(Xtest)

cmat = confusion_matrix(Ytest, Ypred)
print('Confusion matrix =\n', cmat)

plt.figure(1);  
plt.scatter(X[:, 0], X[:, 1], c = Y)
plt.suptitle('Purchase Data')
plt.xlabel('Scaled Age')
plt.ylabel('Scaled Income')
plt.grid(1,which='both')
plt.axis('tight')
plt.show()

col = cf.predict(X)

plt.figure(2);
plt.scatter(X[:, 0], X[:, 1], c = col)
plt.suptitle('Naive Bayes Purchase Data')
plt.xlabel('Scaled Age')
plt.ylabel('Scaled Income')
plt.grid(1,which='both')
plt.axis('tight')
plt.show()