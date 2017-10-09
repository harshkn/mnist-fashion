from __future__ import division
import utils.mnist_reader as mnist_reader
from sklearn import metrics
from xgboost import XGBClassifier

X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

print('Init model')
clf = XGBClassifier()

print('Training model')
clf.fit(X_train, y_train)
# clf.fit(X_train[0:5000, :], y_train[0:5000])
print('Done')

print('Testing model')
res = clf.predict(X_test)
print('Done')

accuracy = metrics.accuracy_score(y_test, res)
print(accuracy)

