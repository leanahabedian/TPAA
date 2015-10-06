from sklearn.naive_bayes import MultinomialNB
from sklearn import cross_validation
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn import tree, metrics
import csv
import numpy as np
import os

csv_filename = "notas_ln_semicolons.csv"

STOP_WORDS = ["a","aca","ahi","ajena","ajenas","ajeno","ajenos","al","algo","algun","alguna","algunas","alguno","algunos","alla","alli","alli","ambos","ampleamos","ante","antes","aquel","aquella","aquellas","aquello","aquellos","aqui","aqui","arriba","asi","atras","aun","aunque","bajo","bastante","bien","cabe","cada","casi","cierta","ciertas","cierto","ciertos","como","como","con","conmigo","conseguimos","conseguir","consigo","consigue","consiguen","consigues","contigo","contra","cual","cuales","cualquier","cualquiera","cualquieras","cuan","cuan","cuando","cuanta","cuanta","cuantas","cuantas","cuanto","cuanto","cuantos","cuantos","de","dejar","del","demas","demas","demasiada","demasiadas","demasiado","demasiados","dentro","desde","donde","dos","el","el","ella","ellas","ello","ellos","empleais","emplean","emplear","empleas","empleo","en","encima","entonces","entre","era","eramos","eran","eras","eres","es","esa","esas","ese","eso","esos","esta","estaba","estado","estais","estamos","estan","estar","estas","este","esto","estos","estoy","etc","fin","fue","fueron","fui","fuimos","gueno","ha","hace","haceis","hacemos","hacen","hacer","haces","hacia","hago","hasta","incluso","intenta","intentais","intentamos","intentan","intentar","intentas","intento","ir","jamas","junto","juntos","la","largo","las","lo","los","mas","mas","me","menos","mi","mia","mia","mias","mientras","mio","mio","mios","mis","misma","mismas","mismo","mismos","modo","mucha","muchas","muchisima","muchisimas","muchisimo","muchisimos","mucho","muchos","muy","nada","ni","ningun","ninguna","ningunas","ninguno","ningunos","no","nos","nosotras","nosotros","nuestra","nuestras","nuestro","nuestros","nunca","os","otra","otras","otro","otros","para","parecer","pero","poca","pocas","poco","pocos","podeis","podemos","poder","podria","podriais","podriamos","podrian","podrias","por","por que","porque","primero","primero desde","puede","pueden","puedo","pues","que","que","querer","quien","quien","quienes","quienesquiera","quienquiera","quiza","quizas","sabe","sabeis","sabemos","saben","saber","sabes","se","segun","ser","si","si","siempre","siendo","sin","sin","sino","so","sobre","sois","solamente","solo","somos","soy","sr","sra","sres","sta","su","sus","suya","suyas","suyo","suyos","tal","tales","tambien","tambien","tampoco","tan","tanta","tantas","tanto","tantos","te","teneis","tenemos","tener","tengo","ti","tiempo","tiene","tienen","toda","todas","todo","todos","tomar","trabaja","trabajais","trabajamos","trabajan","trabajar","trabajas","trabajo","tras","tu","tu","tus","tuya","tuyo","tuyos","ultimo","un","una","unas","uno","unos","usa","usais","usamos","usan","usar","usas","uso","usted","ustedes","va","vais","valor","vamos","van","varias","varios","vaya","verdad","verdadera","vosotras","vosotros","voy","vuestra","vuestras","vuestro","vuestros","y","ya","yo"]


X = []
y = []
with open(csv_filename, 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=';')
    for i, row in enumerate(spamreader):
        if i != 0:
            X.append( str(row[3]) )
            y.append( str(row[2]) )

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3, random_state=0)

#vectorizer = CountVectorizer()
# Excercise 3.4
vectorizer = CountVectorizer(stop_words=STOP_WORDS)
#

X_train_counts = vectorizer.fit_transform(X_train)


tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
print X_train_tf.shape

tfidf_transformer = TfidfTransformer().fit(X_train_counts)
X_train_tfidf = tfidf_transformer.transform(X_train_counts)
print X_train_tfidf.shape


# Excercise 3.5
clf = MultinomialNB().fit(X_train_counts, y_train)
#clf = MultinomialNB().fit(X_train_tf, y_train) # for coding with TF (term frequencies)
#clf = MultinomialNB().fit(X_train_tfidf, y_train) # for coding with TFIDF (invers document frequency)
#

X_test_counts = vectorizer.transform(X_test)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

predicted = clf.predict(X_test_tfidf)

for doc, category in zip(X_test, predicted):
    print('%r => %s' % (doc[0:20], category))

print np.mean(predicted == y_test)
print metrics.classification_report(y_test, predicted)

# Excersice 3.3
print metrics.confusion_matrix(y_test, predicted)
#
