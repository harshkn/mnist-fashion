from __future__ import division
from sklearn import metrics
import utilities as utils
import numpy as np
np.random.seed(123)  # for reproducibility
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
import keras

# floyd run --gpu --data harsha/datasets/mnist-fashion/1:data "python lenet.py"

images_path = "data/train-images-idx3-ubyte.gz"
labels_path = "data/train-labels-idx1-ubyte.gz"
X_train, y_train = utils.load_data(images_path, labels_path)

images_path = "data/t10k-images-idx3-ubyte.gz"
labels_path = "data/t10k-labels-idx1-ubyte.gz"
X_test, y_test = utils.load_data(images_path, labels_path)

# floyd run --data harsha/datasets/mnist-fashion/1:data --mode jupyter

# define data preparation
datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
# fit parameters from data


# utils.display_img(X_train, y_train, 1)
# utils.display_multiple_img(X_train, 0, 20)

split_size = int(X_train.shape[0]*0.7)

# X_train, X_val = X_train[:split_size], X_train[split_size:]
# y_train, y_val = y_train[:split_size], y_train[split_size:]

print("Training set size " + str(X_train.shape[0]))
# print("Validation set size " + str(X_val.shape[0]))
print("Test set size " + str(X_test.shape[0]))

#
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)

X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

datagen.fit(X_train)

# Convert 1-dimensional class arrays to 10-dimensional class matrices
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

print(X_train.shape)
print(Y_train.shape)

model = Sequential()
model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(28, 28, 1)))
print(model.output_shape)

model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

batch_size = 100
output_folder = 'output/new/'
tbCallBack = keras.callbacks.TensorBoard(log_dir=output_folder, histogram_freq=1,
          write_graph=True, write_images=True)

# model.fit(X_train, Y_train,
#           batch_size=batch_size, epochs=20, validation_split=0.2, verbose=1, callbacks=[tbCallBack])

# model.fit(X_train, Y_train,
#           batch_size=batch_size, nb_epoch=2, verbose=1, callbacks=[tbCallBack])

# model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
#                     steps_per_epoch=X_train.shape[0] // batch_size,
#                     epochs=2,
#                     validation_data=(X_test, y_test))
print(X_train.shape[0])
data = datagen.flow(X_train, Y_train, batch_size=batch_size)

count = 0
for dat in data:
    count = count+1

print(count)

# print(sum([1 for dat in data]))
# model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
#                     samples_per_epoch=80000,
#                     nb_epoch=2)
#
#
# score = model.evaluate(X_test, Y_test, verbose=0)
#
# model.save(output_folder + 'lenet-noreg-epoch250-b10000.h5')
#
# print(model.summary())
# print(score)
# print("Accuracy: %.2f%%" % (score[1]*100))
# print("done")
#
