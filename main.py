import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.keras.optimizers import SGD, Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from PIL import Image
from tensorflow import keras

word_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
             12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W',
             23: 'X', 24: 'Y', 25: 'Z'}


def main():
    # file stuff
    dirname = os.path.dirname(__file__)
    path_train = os.path.join(dirname, 'train')
    path_test = os.path.join(dirname, 'test')

    img_list_train = os.listdir(path_train)
    img_list_test = os.listdir(path_test)
    alphabets = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    # get image train
    train_x = []
    train_y = []
    for img in img_list_train:
        train_x.append(np.array(Image.open(os.path.join(path_train, img))))
        label = []
        for i in alphabets:
            label.append(1 if i == img[0] else 0)
        train_y.append(label)

    train_x = np.array(train_x)
    train_yOHE = np.array(train_y)
    print("Train data shape: ", train_x.shape)

    # get image test
    test_x = []
    test_y = []
    for img in img_list_test:
        test_x.append(np.array(Image.open(os.path.join(path_test, img))))
        label = []
        for i in alphabets:
            label.append(1 if i == img[0] else 0)
        test_y.append(label)

    test_x = np.array(test_x)
    test_yOHE = np.array(test_y)
    print("Test data shape: ", test_x.shape)

    # Reshaping the training & test dataset so that it can be put in the model...
    train_X = train_x.reshape(train_x.shape[0], train_x.shape[1], train_x.shape[2], 1)
    print("New shape of train data: ", train_X.shape)
    test_X = test_x.reshape(test_x.shape[0], test_x.shape[1], test_x.shape[2], 1)
    print("New shape of test data: ", test_X.shape)

    # Converting the labels to categorical values...
    print("New shape of train labels: ", train_yOHE.shape)
    print("New shape of test labels: ", test_yOHE.shape)

    CNNModel(train_X, test_X, train_yOHE, test_yOHE, alphabets)


def neural_network_example():
    # Read the data...
    data = pd.read_csv("A_Z Handwritten Data.csv").astype('float32')

    # Split data the X - Our data , and y - the prdict label
    X = data.drop('0', axis=1)
    y = data['0']

    # Reshaping the data in csv file so that it can be displayed as an image...

    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2)
    train_x = np.reshape(train_x.values, (train_x.shape[0], 28, 28))
    test_x = np.reshape(test_x.values, (test_x.shape[0], 28, 28))

    print("Train data shape: ", train_x.shape)
    print("Test data shape: ", test_x.shape)

    # Dictionary for getting characters from index values...
    # word_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
    #              12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W',
    #              23: 'X', 24: 'Y', 25: 'Z'}

    alphabets = []
    for i in word_dict.values():
        alphabets.append(i)

    # Plotting the number of alphabets in the dataset...
    """ 
    train_yint = np.int0(y)
    count = np.zeros(26, dtype='int')
    for i in train_yint:
        count[i] += 1



    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.barh(alphabets, count)

    plt.xlabel("Number of elements ")
    plt.ylabel("Alphabets")
    plt.grid()
    plt.show()

    # Shuffling the data ...
    shuff = shuffle(train_x[:100])

    fig, ax = plt.subplots(3, 3, figsize=(10, 10))
    axes = ax.flatten()

    for i in range(9):
        axes[i].imshow(np.reshape(shuff[i], (28, 28)), cmap="Greys")
    plt.show() """

    # Reshaping the training & test dataset so that it can be put in the model...

    train_X = train_x.reshape(train_x.shape[0], train_x.shape[1], train_x.shape[2], 1)
    print("New shape of train data: ", train_X.shape)

    test_X = test_x.reshape(test_x.shape[0], test_x.shape[1], test_x.shape[2], 1)
    print("New shape of test data: ", test_X.shape)

    # Converting the labels to categorical values...

    train_yOHE = to_categorical(train_y, num_classes=26, dtype='int')
    print("New shape of train labels: ", train_yOHE.shape)

    test_yOHE = to_categorical(test_y, num_classes=26, dtype='int')
    print("New shape of test labels: ", test_yOHE.shape)

    CNNModel(train_X, test_X, train_yOHE, test_yOHE, alphabets)


def CNNModel(train_X, test_X, train_yOHE, test_yOHE, alphabets):
    # CNN model...
    if (os.path.exists(r'model_hand.h5')):

        model = keras.models.load_model(r'model_hand.h5')
    else:
        model = Sequential()

        model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(MaxPool2D(pool_size=(2, 2), strides=2))

        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(MaxPool2D(pool_size=(2, 2), strides=2))

        model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='valid'))
        model.add(MaxPool2D(pool_size=(2, 2), strides=2))

        model.add(Flatten())

        model.add(Dense(64, activation="relu"))
        model.add(Dense(128, activation="relu"))

        model.add(Dense(len(alphabets), activation="softmax"))

        model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0001)
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')

        history = model.fit(train_X, train_yOHE, epochs=1, callbacks=[reduce_lr, early_stop],
                            validation_data=(test_X, test_yOHE))
        model.summary()
        model.save(r'model_hand.h5')

        # Displaying the accuracies & losses for train & validation set...

        print("The validation accuracy is :", history.history['val_accuracy'])
        print("The training accuracy is :", history.history['accuracy'])
        print("The validation loss is :", history.history['val_loss'])
        print("The training loss is :", history.history['loss'])

    # Making model predictions...

    pred = model.predict(test_X[:9])
    print(test_X.shape)

    # Displaying some of the test images & their predicted labels...

    fig, axes = plt.subplots(3, 3, figsize=(8, 9))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        img = np.reshape(test_X[i], (28, 28))
        ax.imshow(img, cmap="Greys")
        pred = alphabets[np.argmax(test_yOHE[i])]
        ax.set_title("Prediction: " + pred)
        ax.grid()

    # Prediction on external image...
    img = cv2.imread('./highres_handwritten_b.png')
    img_copy = img.copy()


    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (400, 440))

    img_copy = cv2.GaussianBlur(img_copy, (7, 7), 0)
    img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    _, img_thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)

    img_final = cv2.resize(img_thresh, (28, 28))
    img_final = np.reshape(img_final, (1, 28, 28, 1))

    img_pred = alphabets[np.argmax(model.predict(img_final))]



    #Baseline 1: Guessing Randomly
    print("Baseline 1: " + word_dict[random.randint(0, 25)])

    #Baseline 2: Pixel Checker Decision Tree
    baseline2(img_filename)



    cv2.putText(img, "Dataflair _ _ _ ", (20, 25), cv2.FONT_HERSHEY_TRIPLEX, 0.7, color=(0, 0, 230))
    cv2.putText(img, "Prediction: " + img_pred, (20, 410), cv2.FONT_HERSHEY_DUPLEX, 1.3, color=(255, 0, 30))
    cv2.imshow('Dataflair handwritten character recognition _ _ _ ', img)

    while (1):
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()
    pass



def baseline2(filename):
    im = Image.open(filename)
    pix = im.load()
    #print(im.size[0])
    width = im.size[0]
    height = im.size[0]
    pass

def baseline3(filename):
    pass


if __name__ == '__main__':
    # main()
    neural_network_example()

    #img_filename = './test/R-935.png'
    #baseline2(img_filename)
