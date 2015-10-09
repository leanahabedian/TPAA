from sklearn.naive_bayes import MultinomialNB
from sklearn import cross_validation
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction import text
from sklearn import tree, metrics
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import csv
import numpy as np
import os

csv_filename = "notas_ln.csv"

# loading the csv file
X = []
y = []
with open(csv_filename, 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for i, row in enumerate(spamreader):
        if i != 0:
            X.append( str(row[3]) )
            y.append( str(row[2]) )
#

# spliting the corpus in 20% for test and 80% for train
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2, random_state=0)

#vectorizer = CountVectorizer() ; run_type = "tokenizing stop words"
# Excercise 3.4
vectorizer = CountVectorizer(stop_words=stopwords.words('spanish')) ; run_type = "without tokenizing stop words"
#

X_train_counts = vectorizer.fit_transform(X_train)

"""
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)

tfidf_transformer = TfidfTransformer().fit(X_train_counts)
X_train_tfidf = tfidf_transformer.transform(X_train_counts)
"""

clf = MultinomialNB().fit(X_train_counts, y_train)

X_test_counts = vectorizer.transform(X_test)

predicted = clf.predict(X_test_counts)

#
#for doc, category in zip(X_test, predicted):
#    print('%r => %s' % (doc[0:20], category))
#

print "clasiffier precision: ",np.mean(predicted == y_test)
print metrics.classification_report(y_test, predicted)

# Excersice 3.3
cm = metrics.confusion_matrix(y_test, predicted)
print cm
# Show confusion matrix in a separate window
plt.matshow(cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
#
