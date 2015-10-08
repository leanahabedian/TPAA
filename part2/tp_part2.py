import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets


csv_filename = "winequality-red.csv"

# constants
h = 1  # step size in the mesh
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

def load_data():

    # load the CSV file as a numpy matrix
    dataset = np.loadtxt(csv_filename, delimiter=';', skiprows=1)

    # separate the data from the target attributes
    X = dataset[:,:11]
    # reduce dimension of X with pca
    pca = PCA(n_components=2)
    pca.fit(X)
    print "Matriz de covariancia sin eliminar atributos"
    print pca.get_covariance()

    X = dataset[:, [0,1,2,3,4,7,8,9,10]]
    # reduce dimension of X with pca
    pca = PCA(n_components=2)
    pca.fit(X)
    print "Matriz de covariancia eliminando atributos 5 y 6"
    print pca.get_covariance()

    X = dataset[:, [1,2,3,4,7,8,9,10]]
    # reduce dimension of X with pca
    pca = PCA(n_components=2)
    pca.fit(X)
    print "Matriz de covariancia eliminando atributos 0, 5 y 6"
    print pca.get_covariance()

    X = pca.transform(X)
    y = dataset[:,11]
    return X, y


def split_sample_in_train_and_test(X, y):
    # split dataset: 30% for test and 70% for train
    return  cross_validation.train_test_split(X, y, test_size=0.1, random_state=0)

    
def plot_knn(X, y, weights, n_neighbors):
    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights).fit(X, y)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i, weights = '%s')" % (n_neighbors, weights))
        
    plt.show()


def analyze_score(X_train, y_train, X_test, y_test, weights):
    
    max_node_count = X_train.shape[0]
    x_coord = []
    score_test = []
    score_train = []

    # we create different instances of Neighbours Classifier
    for n_neighbors in range(1, max_node_count,10):
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights).fit(X_train, y_train)
        x_coord.append(n_neighbors)
        score_train.append(clf.score(X_train,y_train))
        score_test.append(clf.score(X_test,y_test))

    # generate chart
    plt.plot(x_coord, score_test, '')
    plt.title('Variacion del score en funcion del k')
    plt.xlabel('k')
    plt.ylabel('score')
    plt.legend(["test"], loc="upper left")
    plt.show()

def analyze_score_all_weghts(X_train, y_train, X_test, y_test):
    
    max_node_count = X_train.shape[0]
    x_coord = []
    score_uniform = []
    score_distance = []

    # we create different instances of Neighbours Classifier
    for n_neighbors in range(1, max_node_count,10):
        x_coord.append(n_neighbors)
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform').fit(X_train, y_train)
        score_uniform.append(clf.score(X_test,y_test))
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance').fit(X_train, y_train)
        score_distance.append(clf.score(X_test,y_test))

    # generate chart
    plt.plot(x_coord, score_uniform, '', x_coord, score_distance, '')
    plt.title('Variacion del score en funcion del k')
    plt.xlabel('k')
    plt.ylabel('score')
    plt.legend(["uniform", "distance"], loc="upper left")
    plt.show()

def analyse_performance_with_noise(X_train, X_test, y_train, y_test):

    def make_graph(y_train_with_noise, score_uniform, score_distance):
        del x_coord[:]
        for n_neighbors in range(1, X_train.shape[0],10):
            x_coord.append(n_neighbors)
            clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform').fit(X_train, y_train_with_noise)
            score_uniform.append(clf.score(X_test,y_test))
            clf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance').fit(X_train, y_train_with_noise)
            score_distance.append(clf.score(X_test,y_test))
       
    # initialize data and file
    x_coord = []
    score_uniform_0 = []
    score_distance_0 = []
    score_uniform_10 = []
    score_distance_10 = []
    score_uniform_20 = []
    score_distance_20 = []

    # adding noise to the sample
    noise_idx = np.random.random(y_train.shape)
    y_train_with_noise_10 = y_train.copy()
    y_train_with_noise_20 = y_train.copy()
    y_train_with_noise_10[noise_idx<0.1] = np.floor(y_train_with_noise_10[noise_idx<0.1] - 1) * (-1)
    y_train_with_noise_20[noise_idx<0.2] = np.floor(y_train_with_noise_20[noise_idx<0.2] - 1) * (-1)

    make_graph(y_train, score_uniform_0, score_distance_0)
    make_graph(y_train_with_noise_10, score_uniform_10, score_distance_10)
    make_graph(y_train_with_noise_20, score_uniform_20, score_distance_20)

    # generate chart
    #plt.plot(x_coord, score_uniform_0, '', x_coord, score_distance_0, '', x_coord, score_uniform_10, '', x_coord, score_distance_10, '', x_coord, score_uniform_20, '', x_coord, score_distance_20, '')
    plt.plot(x_coord, score_distance_0, '', x_coord, score_distance_10, '', x_coord, score_distance_20, '')
    plt.title('Variacion del score en funcion del k con ruido')
    plt.xlabel('k')
    plt.ylabel('score')
    #plt.legend(["uniform 0% ruido", "distance 0% ruido", "uniform 10% ruido", "distance 10% ruido", "uniform 20% ruido", "distance 20% ruido",], loc="upper left")
    plt.legend(["distance 0% ruido", "distance 10% ruido", "distance 20% ruido"], loc="upper left")
    plt.show()

if __name__ == "__main__":
    
    X, y = load_data()
    # k variation
    plot_knn(X, y, 'uniform',1)
    plot_knn(X, y, 'uniform',7)
    plot_knn(X, y, 'uniform',1599)
    # score over train
    analyze_score(X, y, X, y, 'uniform')

    X_train, X_test, y_train, y_test = split_sample_in_train_and_test(X, y)
    # score over test 
     analyze_score(X_train, y_train, X_test, y_test, 'uniform')
    # plot with distance
    plot_knn(X, y, 'distance',1599)
    analyze_score_all_weghts(X_train, y_train, X_test, y_test)
    # performance with noise
    analyse_performance_with_noise(X_train, X_test, y_train, y_test)   
   
