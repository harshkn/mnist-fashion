# import matplotlib.pyplot as plt
import struct
import gzip
import numpy as np
import cv2


# def display_img(x_train, y_train, num):
#     print(y_train[num])
#     label = y_train[num].argmax(axis=0)
#     image = x_train[num].reshape([28,28])
#     plt.title('Example: %d  Label: %d' % (num, label))
#     plt.imshow(image, cmap=plt.get_cmap('gray_r'))
#     plt.show()


def load_data(images_path, labels_path):
    with gzip.open(labels_path, 'rb') as lbpath:
        struct.unpack('>II', lbpath.read(8))
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8)

    with gzip.open(images_path, 'rb') as imgpath:
        struct.unpack(">IIII", imgpath.read(16))
        images = np.frombuffer(imgpath.read(), dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


def display_multiple_img(x_train, start, stop):
    images = x_train[start].reshape([28,28])
    for i in range(start+1,stop):
        # images = np.concatenate((images, x_train[i].reshape([28,28])))
        images = np.hstack((images, x_train[i].reshape([28,28])))
        # print(images.shape)

    cv2.imshow("window", images)
    cv2.waitKey(0)
    # # plt.imshow(images, cmap=plt.get_cmap('gray_r'))
    # # plt.show()
