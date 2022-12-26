import keras.utils
import tensorflow as tf
from resnet import *


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    resnet50 = Resnet50(1000)
    resnet50.net.summary()



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
