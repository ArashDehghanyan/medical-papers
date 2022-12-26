import keras.utils
import tensorflow as tf
from resnet import *


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    resnet50 = Resnet50(1000, dropout_rate=0.5)
    X = tf.random.normal((1, 224, 224, 3))
    y = resnet50(X)
    resnet50.summary()



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
