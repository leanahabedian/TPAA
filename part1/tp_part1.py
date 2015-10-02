from sklearn import tree
from sklearn import cross_validation
import numpy as np
import csv
import os
import matplotlib.pyplot as plt

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

def load_data():

    # load the CSV file as a numpy matrix
    dataset = np.loadtxt(csv_filename, delimiter=";", skiprows=1)

    # separate the data from the target attributes
    X = dataset[:,0:11]
    y = dataset[:,11]
    return X, y

def build_different_trees(X, y):
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

def split_sample_in_train_and_test(X, y):
    return  cross_validation.train_test_split(X, y, test_size=0.3, random_state=0)

def analyse_performance(X_train, X_test, y_train, y_test):
    # initialize data and file
    performanceX = []
    performance_train = []
    performance_test = []
    result_performance_file = open(result_performance_filename, 'w+')
    result_performance_file.write("Cantidad de nodos, Score de entrenamiento, Score de test\n")

    for i in range(2,100):
        clf = tree.DecisionTreeClassifier(max_leaf_nodes=i).fit(X_train,y_train)
        performanceX.append(clf.tree_.node_count)
        performance_train.append(clf.score(X_train,y_train))
        performance_test.append(clf.score(X_test,y_test))
        result_performance_file.write(str(clf.tree_.node_count)+","+str(clf.score(X_train,y_train))+","+str(clf.score(X_test,y_test))+"\n")

    result_performance_file.close()
    
    # generate chart
    plt.plot(performanceX, performance_train, '', performanceX, performance_test, '')
    plt.title('performance')
    plt.xlabel('cantidad de nodos')
    plt.ylabel('score')
    plt.legend(["score de entrenamiento","score de test"], loc="upper left")
    plt.show()


def analyse_performance_with_noise(X_train, X_test, y_train, y_test):

    def make_graph(y_train_with_noise, percent):
        for i in range(2,250):
            clf = tree.DecisionTreeClassifier(max_leaf_nodes=i).fit(X_train,y_train_with_noise)
            performanceX.append(clf.tree_.node_count)
            performance_train.append(clf.score(X_train,y_train_with_noise))
            performance_test.append(clf.score(X_test,y_test))
#            result_noisy_file.write(str(clf.tree_.node_count)+","+str(clf.score(X_train,y_train_with_noise))+","+str(clf.score(X_test,y_test))+"\n")

        result_noisy_file.close()

        # generate chart
        plt.plot(performanceX, performance_train, '', performanceX, performance_test, '')
        plt.title('performance con '+percent+'% de ruido')
        plt.xlabel('cantidad de nodos')
        plt.ylabel('score')
        plt.legend(["score de entrenamiento","score de test"], loc="upper left")
        plt.show()
        del performanceX[:]
        del performance_train[:]
        del performance_test[:]


    # initialize data and file
    performanceX = []
    performance_train = []
    performance_test = []
    result_noisy_file = open(result_noisy_filename, 'w+')
    result_noisy_file.write("Cantidad de nodos, Score de entrenamiento, Score de test 0% error, Score de test \n")

    # adding noise to the sample
    noise_idx = np.random.random(y_train.shape)
    y_train_with_noise_10 = y_train.copy()
    y_train_with_noise_30 = y_train.copy()
    y_train_with_noise_50 = y_train.copy()
    y_train_with_noise_10[noise_idx<0.1] = np.floor(y_train_with_noise_10[noise_idx<0.1] - 1) * (-1)
    y_train_with_noise_30[noise_idx<0.3] = np.floor(y_train_with_noise_30[noise_idx<0.3] - 1) * (-1)
    y_train_with_noise_50[noise_idx<0.5] = np.floor(y_train_with_noise_50[noise_idx<0.5] - 1) * (-1)

    make_graph(y_train_with_noise_10, "10")
    make_graph(y_train_with_noise_30, "30")
    make_graph(y_train_with_noise_50, "50")

if __name__ == "__main__":
    X,y = load_data() 
    build_different_trees(X,y) 
    X_train, X_test, y_train, y_test = split_sample_in_train_and_test(X, y)
    analyse_performance(X_train, X_test, y_train, y_test)
    analyse_performance_with_noise(X_train, X_test, y_train, y_test)
