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

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,y)

other_filename = "result2.csv"
my_file = open(other_filename, 'w+')

for i in range(1,1599):
    clf = tree.DecisionTreeClassifier(max_depth=i).fit(X,y)
    my_file.write(str(clf.tree_.node_count)+","+str(clf.score(X,y))+"\n")


my_file.close()
#with open(result_filename, 'w') as f: f = tree.export_graphviz(clf, out_file=f)


