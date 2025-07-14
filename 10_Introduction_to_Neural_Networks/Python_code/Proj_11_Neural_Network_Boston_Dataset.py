from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

BosData = pd.read_csv('BostonHousing.csv')
X = BosData.iloc[:,0:13]
y = BosData.iloc[:,13] # MEDV: Median value of owner-occupied homes in $1000s

ss = StandardScaler()
X = ss.fit_transform(X)

Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size = 0.2)

model = Sequential()
model.add(Dense(15, input_dim=13, activation="relu"))
#15 - No. of neurons, 13 features, 
model.add(Dense(1))
model.compile(loss="mean_squared_error")

history = model.fit(Xtrain, ytrain, epochs=150, batch_size=10)
ypred = model.predict(Xtest)
ypred = ypred[:,0]

error = np.sum(np.abs(ytest-ypred))/np.sum(np.abs(ytest))*100
print('Prediction Error is',error,'%')