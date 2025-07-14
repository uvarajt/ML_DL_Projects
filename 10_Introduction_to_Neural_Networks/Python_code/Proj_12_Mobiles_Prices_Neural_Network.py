from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
import tensorflow as tf
nc = 4 # Number of classes

df = pd.read_csv("mobile_prices.csv")
X = df.iloc[:,:20].values
y = df.iloc[:,20:21].values
ss = StandardScaler()
X = ss.fit_transform(X)

Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size = 0.1)
ytrain = tf.one_hot(ytrain[:,0], depth=nc)

model = Sequential()
model.add(Dense(20, input_dim=20, activation="relu"))
model.add(Dense(12, activation="relu"))
model.add(Dense(4, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

history = model.fit(Xtrain, ytrain, epochs=100, batch_size=64)
ypred = model.predict(Xtest)
ypred = np.argmax(ypred, axis=1)

score = accuracy_score(ypred,ytest)
print('Accuracy score is',100*score,'%')
