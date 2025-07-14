import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


irisset = datasets.load_iris()
X = irisset.data[:50,0:1]
y = irisset.data[:50,1]

reg = LinearRegression().fit(X,y)
w = reg.coef_
c = reg.intercept_

xpoints = np.linspace(4,6)
ypoints = w[0] * xpoints + c;
plt.plot(xpoints, ypoints, 'g-')
plt.scatter(X, y,s=10)
plt.suptitle('Linear Regression IRIS Data')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.grid(1,which='both')
plt.axis('tight')
plt.show()

yPredict = reg.predict(X)
rmse = (np.sqrt(mean_squared_error(y, yPredict)))
r2 = r2_score(y, yPredict)
print('Train RMSE =', rmse)
print('Train R2 score =', r2)
print("\n")