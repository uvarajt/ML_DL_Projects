import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.tree import plot_tree
from sklearn import tree
import graphviz

purchaseData = pd.read_csv('Purchase_Logistic.csv')
X = purchaseData.iloc[:, [2, 3]]
Y = purchaseData.iloc[:, 4]

Xtrain, Xtest, Ytrain, Ytest \
= train_test_split(X, Y, test_size = 0.25, random_state = 0) 

cf=DecisionTreeClassifier(max_depth = 3);
cf.fit(Xtrain,Ytrain);
Ypred = cf.predict(Xtest)
cmat = confusion_matrix(Ytest, Ypred)

decPlot = plot_tree(decision_tree=cf, feature_names = ["Age", "Salary"], 
                     class_names =["No", "Yes"] , filled = True , precision = 4, rounded = True)

text_representation = tree.export_text(cf,  feature_names = ["Age","Salary"])
print(text_representation)

dot_data = tree.export_graphviz(cf, out_file=None, 
                     feature_names = ["Age", "Salary"],  
                     class_names =["No", "Yes"],  
                     filled=True, rounded=True,  
                     special_characters=True)  
graph = graphviz.Source(dot_data) 