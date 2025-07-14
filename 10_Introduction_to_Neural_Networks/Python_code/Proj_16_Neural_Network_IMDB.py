from keras.datasets import imdb
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score

(Xtrain, ytrain), (Xtest, ytest) = imdb.load_data(num_words=10000)

def vectorize(sequences, dimension = 10000):
    results = np.zeros((len(sequences), dimension))
    for i in range(len(sequences)):
        results[i, sequences[i]] = 1
    return results
 
Xtrain = vectorize(Xtrain)

model = Sequential()
model.add(Dense(50, input_dim=10000, activation="relu"))
model.add(Dense(50, activation="relu"))
model.add(Dense(50, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model.fit(Xtrain, ytrain, epochs=10, batch_size=550)


Xtest = vectorize(Xtest)
ypred = model.predict(Xtest)
ypred = np.round(ypred)
score = accuracy_score(ypred,ytest)
print('Accuracy score is',100*score,'%')

