from __future__ import division
import utils.mnist_reader as mnist_reader
from sklearn.svm import SVC
from sklearn import metrics

X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

print('Init model')
clf = SVC()

print('Training model...')
clf.fit(X_train[0:60000, :], y_train[0:60000])
print('Done')

print('Testing model...')
res = clf.predict(X_test)
print('Done')

accuracy = metrics.accuracy_score(y_test, res)
print(accuracy)

