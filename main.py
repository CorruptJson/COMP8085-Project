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


    



    #Baseline 1: Guessing Randomly
    print("Baseline 1: " + word_dict[random.randint(0, 25)])

    #Baseline 2: Pixel Checker Decision Tree
    #baseline2(img_filename)




    # Prediction on external image...
    exp1_a = pred_img('./experiments/exp1_a.png', model, alphabets, "Experiment 1: A")
    exp1_b = pred_img('./experiments/exp1_b.png', model, alphabets, "Experiment 1: B")
    exp1_c = pred_img('./experiments/exp1_c.png', model, alphabets, "Experiment 1: C")

    exp_1 = np.hstack((exp1_a, exp1_b, exp1_c))
    cv2.imshow('Experiment 1', exp_1)

    exp2_a = pred_img('./experiments/exp2_a.png', model, alphabets, "Experiment 2: A")
    exp2_b = pred_img('./experiments/exp2_b.png', model, alphabets, "Experiment 2: B")
    exp2_c = pred_img('./experiments/exp2_c.png', model, alphabets, "Experiment 2: C")

    exp_2 = np.hstack((exp2_a, exp2_b, exp2_c))
    cv2.imshow('Experiment 2', exp_2)

    exp3_a = pred_img('./experiments/exp3_a.png', model, alphabets, "Experiment 3: A")
    exp3_b = pred_img('./experiments/exp3_b.png', model, alphabets, "Experiment 3: B")
    exp3_c = pred_img('./experiments/exp3_c.png', model, alphabets, "Experiment 3: C")

    exp_3 = np.hstack((exp3_a, exp3_b, exp3_c))
    cv2.imshow('Experiment 3', exp_3)

    exp4_a = pred_img('./experiments/exp4_a.png', model, alphabets, "Experiment 4: A")
    exp4_b = pred_img('./experiments/exp4_b.png', model, alphabets, "Experiment 4: B")
    exp4_c = pred_img('./experiments/exp4_c.png', model, alphabets, "Experiment 4: C")

    exp_4 = np.hstack((exp4_a, exp4_b, exp4_c))
    cv2.imshow('Experiment 4', exp_4)

    exp5_a = pred_img('./experiments/exp5_a.png', model, alphabets, "Experiment 5: A")
    exp5_b = pred_img('./experiments/exp5_b.png', model, alphabets, "Experiment 5: B")
    exp5_c = pred_img('./experiments/exp5_c.png', model, alphabets, "Experiment 5: C")

    exp_5 = np.hstack((exp5_a, exp5_b, exp5_c))
    cv2.imshow('Experiment 5', exp_5)

    exp6_a = pred_img('./experiments/exp6_a.png', model, alphabets, "Experiment 6: A")
    exp6_b = pred_img('./experiments/exp6_b.png', model, alphabets, "Experiment 6: B")
    exp6_c = pred_img('./experiments/exp6_c.png', model, alphabets, "Experiment 6: C")

    exp_6 = np.hstack((exp6_a, exp6_b, exp6_c))
    cv2.imshow('Experiment 6', exp_6)

    exp7_a = pred_img('./experiments/exp7_a.png', model, alphabets, "Experiment 7: A")
    exp7_b = pred_img('./experiments/exp7_b.png', model, alphabets, "Experiment 7: B")
    exp7_c = pred_img('./experiments/exp7_c.png', model, alphabets, "Experiment 7: C")

    exp_7 = np.hstack((exp7_a, exp7_b, exp7_c))
    cv2.imshow('Experiment 7', exp_7)

    exp8_ab = pred_img('./experiments/exp8_ab.png', model, alphabets, "Experiment 8: AB")
    exp8_ba = pred_img('./experiments/exp8_ba.png', model, alphabets, "Experiment 8: BA")
    exp8_ac = pred_img('./experiments/exp8_ac.png', model, alphabets, "Experiment 8: AC")
    exp8_bc = pred_img('./experiments/exp8_bc.png', model, alphabets, "Experiment 8: BC")

    exp_8 = np.hstack((exp8_ab, exp8_ba, exp8_ac, exp8_bc))
    cv2.imshow('Experiment 8', exp_8)

    exp9_black = pred_img('./experiments/exp9_black.png', model, alphabets, "Experiment 9: Black")
    exp9_white = pred_img('./experiments/exp9_white.png', model, alphabets, "Experiment 9: White")
    exp9_face = pred_img('./experiments/exp9_face.png', model, alphabets, "Experiment 9: Face")
    exp9_field = pred_img('./experiments/exp9_field.png', model, alphabets, "Experiment 9: Field")

    exp_9 = np.hstack((exp9_black, exp9_white, exp9_face, exp9_field))
    cv2.imshow('Experiment 9', exp_9)




    while (1):
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()



def pred_img(filename, model, alphabets, header):
    img = cv2.imread(filename)
    img_copy = img.copy()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (400, 440))

    img_copy = cv2.GaussianBlur(img_copy, (7, 7), 0)
    img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    _, img_thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)

    img_final = cv2.resize(img_thresh, (28, 28))
    img_final = np.reshape(img_final, (1, 28, 28, 1))

    img_pred = alphabets[np.argmax(model.predict(img_final))]
    cv2.putText(img, header, (20, 25), cv2.FONT_HERSHEY_TRIPLEX, 0.7, color=(0, 0, 230))
    cv2.putText(img, "Prediction: " + img_pred, (20, 410), cv2.FONT_HERSHEY_DUPLEX, 1.3, color=(255, 0, 30))


    return img



def baseline2(filename):
    ##Baseline that checks specific pixels and passes the values through a "decision tree"
    im = Image.open(filename, "r")
    pix = im.load()
    width, height = im.size
    print(im.size)
    if im.mode == "RGB":
        channels = 3
        print(channels)
    elif im.mode =="L":
        channels = 1
        print(channels)
    top = pix[0, width/2]
    middle = pix[height/2, width/2]
    left = pix[height/2, 0]
    right = pix[height/2, width-1]

    letter_predict = 0
    if top >= 250:
        if middle >= 250:
            if left >= 250:
                if right >= 250:
                    letter_predict= 0
                else:
                    letter_predict = 1

            elif right >= 250:
                letter_predict = 2
            else:
                letter_predict = 3
        elif left >= 250:
            if right >= 250:
                letter_predict = 4
            else:
                letter_predict = 5
        elif right >= 250:
            letter_predict = 6
        else:
            letter_predict = 7
    elif middle >= 250:
        if left >= 250:
            if right >= 250:
                letter_predict= 8
            else:
                letter_predict = 9
        elif right >= 250:
            letter_predict = 10
        else:
            letter_predict = 11
    elif left >= 250:
        if right >= 250:
            letter_predict = 12
        else:
            letter_predict = 13
    else:
        letter_predict = 14

    print("Baseline 1 predicted: " + word_dict[letter_predict])

    pass

def baseline3(filename):
    ##baseline checks pixels in a column and in a row through the center

    im = Image.open(filename, "r")
    pix = im.load()
    width, height = im.size
    half_width = width/2
    begin = False
    count_height = 0
    count_width = 0
    for h in range(0, height):
        if pix[h, half_width] <= 200: #goes dark
            begin = True
        elif begin and pix[h, half_width] >= 250: #becomes white
            count_height += 1
            begin = False
    half_height = height/2
    for w in range(0, width):
        if pix[half_height, w] <=200: #Goes dark
            begin = True
        elif begin and pix[half_height, w] >= 250: #Becomes white
            count_width += 1
            begin = False
    letter_predict = 0
    if count_height == 0:
        if count_width == 0:
            letter_predict = 0
        elif count_width == 1:
            letter_predict = 1
        elif count_width == 2:
            letter_predict = 2
        elif count_width == 3:
            letter_predict = 3
        else:
            print("too many")

    elif count_height == 1:
        if count_width == 0:
            letter_predict = 4
        elif count_width == 1:
            letter_predict = 5
        elif count_width == 2:
            letter_predict = 6
        elif count_width == 3:
            letter_predict = 7
        else:
            print("too many")

    elif count_height == 2:
        if count_width == 0:
            letter_predict = 8
        elif count_width == 1:
            letter_predict = 9
        elif count_width == 2:
            letter_predict = 10
        elif count_width == 3:
            letter_predict = 11
        else:
            print("too many")

    elif count_height == 3:
        if count_width == 0:
            letter_predict = 12
        elif count_width == 1:
            letter_predict = 13
        elif count_width == 2:
            letter_predict = 14
        elif count_width == 3:
            letter_predict = 15
        else:
            print("too many")

    else:
        print("Theres too many height crossings")

        print("Baseline 2 predicted: " + word_dict[letter_predict])
    pass


if __name__ == '__main__':
    #main()
    neural_network_example()

    img_filename = './test/R-935.png'
    baseline2(img_filename)
