import os

import numpy as np
from PIL import Image


def main():
    # file stuff
    dirname = os.path.dirname(__file__)
    path_train = os.path.join(dirname, 'train')
    path_test = os.path.join(dirname, 'test')

    img_list_train = os.listdir(path_train)
    img_list_test = os.listdir(path_test)

    # get image
    for img in img_list_train:
        im = Image.open(os.path.join(path_train, img))
        na = np.array(im)
        print(f"Image Name: {img}\nImage Data:\n{na}")
        break


if __name__ == '__main__':
    main()

