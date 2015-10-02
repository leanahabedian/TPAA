from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
import numpy as np
import os

csv_filename = "notas_ln.csv"


# load the CSV file as a numpy matrix
dataset = np.loadtxt(csv_filename, dtype=np.str_, delimiter=",")
X = []
y = []
with open("winequality-red.csv", 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=';')
    for i, row in enumerate(spamreader):
        if i != 0:
            row_x = row[:-1]
            X.append( row_x )
            y.append( str(row[11]) )


# separate the data from the target attributes
X = dataset[:,[1,3,4]]
y = dataset[:,2]



