from sklearn import tree
import urllib
import numpy as np
import urllib
import os

csv_filename = "winequality-red.csv"
result_filename = "result.dot"

try:
    os.remove(result_filename)
except OSError:
    pass

# URL for the Pima Indians Diabetes dataset (UCI Machine Learning Repository)
# load the CSV file as a numpy matrix
dataset = np.loadtxt(csv_filename, delimiter=";", skiprows=1)
print(dataset.shape)
# separate the data from the target attributes
X = dataset[:,0:11]
y = dataset[:,11]

#clf = tree.DecisionTreeClassifier()
#clf = clf.fit(X,y)

#print(clf.tree_.node_count)

#print(clf.score(X,y))

clf = tree.DecisionTreeClassifier(max_depth=3)
clf = clf.fit(X,y)
print (clf.tree_.node_count)
print (clf.score(X,y))

with open(result_filename, 'w') as f: f = tree.export_graphviz(clf, out_file=f)


