from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn import tree
from sklearn import datasets

irisset = datasets.load_iris();
X = irisset.data
Y = irisset.target

cf = DecisionTreeClassifier();
cf.fit(X,Y);


decPlot = plot_tree(decision_tree=cf, feature_names = ["sepal_length","sepal_width","petal_length","petal_width"], 
                     class_names =["setosa", "vercicolor", "verginica"] , filled = True , precision = 4, rounded = True)

text_representation = tree.export_text(cf, feature_names = ["sepal_length","sepal_width","petal_length","petal_width"])
print(text_representation)