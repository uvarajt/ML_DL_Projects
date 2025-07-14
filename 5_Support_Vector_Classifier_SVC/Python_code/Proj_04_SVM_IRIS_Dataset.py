import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import datasets

irisset = datasets.load_iris()
X = irisset.data[:100,:2]
z = irisset.target[:100]

clf = svm.SVC(kernel='linear')
clf.fit(X,z)
w = clf.coef_[0]

xpoints = np.linspace(4,7)
ypoints = -w[0] / w[1] * xpoints -clf.intercept_[0]/ w[1]

plt.plot(xpoints, ypoints, 'g-')
plt.scatter(X[:, 0], X[:, 1], c = z)
plt.suptitle('SVM IRIS Data')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.grid(1,which='both')
plt.axis('tight')
plt.show()
