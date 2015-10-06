from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
import numpy as np
import os

csv_filename = "notas_ln_semicolons.csv"


# load the CSV file as a numpy matrix
dataset = np.genfromtxt(csv_filename, dtype=np.str_, delimiter=";", skip_header=1, )

# separate the data from the target attributes
X = dataset[:,[1,3,4]]
y = dataset[:,2]

print X
print y

