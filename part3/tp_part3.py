from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
import numpy as np
import os

csv_filename = "notas_ln.csv"


# load the CSV file as a numpy matrix
dataset = np.loadtxt(csv_filename, dtype=str, delimiter=",", skiprows=1)
print(dataset.shape)
# separate the data from the target attributes
X = dataset[:,[1,3,4]]
y = dataset[:,2]



