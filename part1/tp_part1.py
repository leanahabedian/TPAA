from sklearn import tree
import numpy as np
import os

csv_filename = "winequality-red.csv"
result_tree_filename = "original_tree.dot"
result_csv_filename = "accuracy.csv"

try:
    os.remove(result_tree_filename)
    os.remove(result_csv_filename)
except OSError:
    pass

# load the CSV file as a numpy matrix
dataset = np.loadtxt(csv_filename, delimiter=";", skiprows=1)
print(dataset.shape)
# separate the data from the target attributes
X = dataset[:,0:11]
y = dataset[:,11]

# build tree without restrictions
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,y)
with open(result_tree_filename, 'w') as f: f = tree.export_graphviz(clf, out_file=f)

# build different trees depending on max_leaf_nodes
result_csv_file = open(result_csv_filename, 'w+')
node_count = clf.tree_.node_count

for i in range(2,100):
    clf = tree.DecisionTreeClassifier(max_leaf_nodes=i).fit(X,y)
    result_csv_file.write(str(clf.tree_.node_count)+","+str(clf.score(X,y))+"\n")

result_csv_file.close()



