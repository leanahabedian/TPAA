from sklearn import tree
from sklearn import cross_validation
import numpy as np
import csv
import os
import matplotlib.pyplot as plt

csv_filename = "winequality-red.csv"
result_tree_filename = "base_tree.dot"

try:
    os.remove(result_tree_filename)
except OSError:
    pass


def load_data():
    # load the CSV file as a numpy matrix
    dataset = np.loadtxt(csv_filename, delimiter=";", skiprows=1)

    # separate the data from the target attributes
    X = dataset[:,0:11]
    y = dataset[:,11]

    return X, y


def analyze_base_tree(X, y):
    # build tree without restrictions
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X,y)

    print("Node count: " + str(clf.tree_.node_count))
    print("Score: " + str(clf.score(X,y)))
    
    max_node_count = clf.tree_.node_count
    x_coord = []
    node_count = []
    score = []

    #with open(result_tree_filename, 'w') as f: f = tree.export_graphviz(clf, out_file=f)

    # build different trees depending on max_leaf_nodes
    for i in range(2, max_node_count):
        clf = tree.DecisionTreeClassifier(max_leaf_nodes=i).fit(X,y)
        x_coord.append(i)
        node_count.append(clf.tree_.node_count)
        score.append(clf.score(X,y))

    # generate chart 1
    plt.plot(x_coord, node_count, '')
    plt.title('Variacion de la cantidad de nodos restringiendo max_leaf_nodes')
    plt.xlabel('max_leaf_nodes')
    plt.ylabel('node_count')
    plt.legend(["node_count"], loc="upper left")
    plt.show()

    # generate chart 2
    plt.plot(x_coord, score, '')
    plt.title('Variacion del score restringiendo max_leaf_nodes')
    plt.xlabel('max_leaf_nodes')
    plt.ylabel('score')
    plt.legend(["score"], loc="upper left")
    plt.show()


def split_sample_in_train_and_test(X, y):
    # split dataset: 30% for test and 70% for train
    return  cross_validation.train_test_split(X, y, test_size=0.3, random_state=0)


def analyse_performance(X_train, X_test, y_train, y_test):
    # initialize data and file
    performanceX = []
    performance_train = []
    performance_test = []
    
    for i in range(2,300):
        clf = tree.DecisionTreeClassifier(max_leaf_nodes=i).fit(X_train,y_train)
        performanceX.append(clf.tree_.node_count)
        performance_train.append(clf.score(X_train,y_train))
        performance_test.append(clf.score(X_test,y_test))
 
    # generate chart
    plt.plot(performanceX, performance_train, '', performanceX, performance_test, '')
    plt.title('Performance con cross_validation')
    plt.xlabel('node_count')
    plt.ylabel('score')
    plt.legend(["train","test"], loc="upper left")
    plt.show()


def analyse_performance_with_noise(X_train, X_test, y_train, y_test):

    def make_graph(y_train_with_noise, performance_train, performance_test):
        del performanceX[:]
        for i in range(2,300):
            clf = tree.DecisionTreeClassifier(max_leaf_nodes=i).fit(X_train,y_train_with_noise)
            performanceX.append(clf.tree_.node_count)
            performance_train.append(clf.score(X_train,y_train_with_noise))
            performance_test.append(clf.score(X_test,y_test))

       
    # initialize data and file
    performanceX = []
    performance_train_without_noise = []
    performance_train_noise_10 = []
    performance_train_noise_20 = []
    performance_train_noise_30 = []
    performance_train_noise_40 = []
    performance_train_noise_50 = []
    performance_test_without_noise = []
    performance_test_noise_10 = []
    performance_test_noise_20 = []
    performance_test_noise_30 = []
    performance_test_noise_40 = []
    performance_test_noise_50 = []

    # adding noise to the sample
    noise_idx = np.random.random(y_train.shape)
    y_train_with_noise_10 = y_train.copy()
    y_train_with_noise_20 = y_train.copy()
    y_train_with_noise_30 = y_train.copy()
    y_train_with_noise_40 = y_train.copy()
    y_train_with_noise_50 = y_train.copy()
    y_train_with_noise_10[noise_idx<0.1] = np.floor(y_train_with_noise_10[noise_idx<0.1] - 1) * (-1)
    y_train_with_noise_20[noise_idx<0.2] = np.floor(y_train_with_noise_20[noise_idx<0.2] - 1) * (-1)
    y_train_with_noise_30[noise_idx<0.3] = np.floor(y_train_with_noise_30[noise_idx<0.3] - 1) * (-1)
    y_train_with_noise_40[noise_idx<0.4] = np.floor(y_train_with_noise_40[noise_idx<0.4] - 1) * (-1)
    y_train_with_noise_50[noise_idx<0.5] = np.floor(y_train_with_noise_50[noise_idx<0.5] - 1) * (-1)

    make_graph(y_train, performance_train_without_noise, performance_test_without_noise)
    make_graph(y_train_with_noise_10, performance_train_noise_10, performance_test_noise_10)
    make_graph(y_train_with_noise_20, performance_train_noise_20, performance_test_noise_20)
    make_graph(y_train_with_noise_30, performance_train_noise_30, performance_test_noise_30)
    make_graph(y_train_with_noise_40, performance_train_noise_40, performance_test_noise_40)
    make_graph(y_train_with_noise_50, performance_train_noise_50, performance_test_noise_50)

    # generate chart
    #plt.plot(performanceX, performance_train_without_noise, '', performanceX, performance_test_without_noise, '', performanceX, performance_train_noise_10, '', performanceX, performance_test_noise_10, '', performanceX, performance_train_noise_20, '', performanceX, performance_test_noise_20, '', performanceX, performance_train_noise_30, '', performanceX, performance_test_noise_30, '', performanceX, performance_train_noise_40, '', performanceX, performance_test_noise_40, '', performanceX, performance_train_noise_50, '', performanceX, performance_test_noise_50, '')
    plt.plot(performanceX, performance_train_without_noise, '', performanceX, performance_test_without_noise, '', performanceX, performance_train_noise_20, '', performanceX, performance_test_noise_20, '', performanceX, performance_train_noise_40, '', performanceX, performance_test_noise_40, '', performanceX, performance_train_noise_50, '', performanceX, performance_test_noise_50, '')
    plt.title('Performance con ruido')
    plt.xlabel('node_count')
    plt.ylabel('score')
    #plt.legend(["train 0% noise","test 0% noise", "train 10% noise","test 10% noise", "train 20% noise","test 20% noise", "train 30% noise","test 30% noise", "train 40% noise","test 40% noise", "train 50% noise","test 50% noise"], loc="upper left")
    plt.legend(["train 0% noise","test 0% noise", "train 20% noise","test 20% noise", "train 40% noise","test 40% noise", "train 50% noise","test 50% noise"], loc="upper left")
    plt.show()
        

if __name__ == "__main__":

    X,y = load_data() 
    #analyze_base_tree(X,y) 
    X_train, X_test, y_train, y_test = split_sample_in_train_and_test(X, y)
    print "Train set: ", X_train.shape, y_train.shape
    print "Test set:", X_test.shape, y_test.shape
    #analyze_base_tree(X_train,y_train)
    #analyse_performance(X_train, X_test, y_train, y_test)
    analyse_performance_with_noise(X_train, X_test, y_train, y_test)
