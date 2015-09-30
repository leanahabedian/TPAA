from sklearn import tree
from sklearn import cross_validation
import numpy as np
import os

csv_filename = "winequality-red.csv"
result_tree_filename = "original_tree.dot"
result_csv_filename = "accuracy.csv"
result_performance_filename = "performance.csv"
result_noisy_filename = "noisyPerformance.csv"

try:
    os.remove(result_tree_filename)
    os.remove(result_csv_filename)
    os.remove(result_performance_filename)
    os.remove(result_noisy_filename)
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

for i in range(2,100):
    clf = tree.DecisionTreeClassifier(max_leaf_nodes=i).fit(X,y)
    result_csv_file.write(str(clf.tree_.node_count)+","+str(clf.score(X,y))+"\n")

result_csv_file.close()

# split dataset in train set and test set
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3, random_state=0)

# analyse performance while increasing the amount of leaf nodes
result_performance_file = open(result_performance_filename, 'w+')
result_performance_file.write("treeSize (#nodes), Train Score, Test Score\n")

for i in range(2,100):
    clf = tree.DecisionTreeClassifier(max_leaf_nodes=i).fit(X_train,y_train)
    result_performance_file.write(str(clf.tree_.node_count)+","+str(clf.score(X_train,y_train))+","+str(clf.score(X_test,y_test))+"\n")

result_performance_file.close()


# analyse performance while increasing the amount of leaf nodes and with noisy samples
result_noisy_file = open(result_noisy_filename, 'w+')
result_noisy_file.write("treeSize (#nodes), Train Score, Test Score\n")

# adding noise to the sample
noise_idx = np.random.random(y_train.shape)
y_train_with_noise = y_train.copy()
y_train_with_noise[noise_idx<0.3] = np.floor(y_train_with_noise[noise_idx<0.3] - 1) * (-1)


for i in range(2,100):
    clf = tree.DecisionTreeClassifier(max_leaf_nodes=i).fit(X_train,y_train_with_noise)
    result_noisy_file.write(str(clf.tree_.node_count)+","+str(clf.score(X_train,y_train_with_noise))+","+str(clf.score(X_test,y_test))+"\n")

result_noisy_file.close()



