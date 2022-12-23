import keras.utils
import tensorflow as tf
from resnet import *


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # blk = Residual(3)
    # X = tf.random.normal((4, 6, 6, 3))
    # y = blk(X)
    # print(y.shape)
    # blk1 = Residual(6, use1x1conv=True, strides=2)
    # y = blk1(X)
    # print(y.shape)
    resnet50 = Resnet50(1000)
    x = tf.random.normal((5, 224, 224, 3))
    y = resnet50(x)
    for i, module in enumerate(resnet50.net.layers):
        print(module.__class__.__name__ + "_" + str(i+1))
        for layer in module.layers:
            x = layer(x)
            print(layer.__class__.__name__, "Output shape:\t", x.shape)
    resnet50.net.summary()
    # keras.utils.plot_model(resnet50.net, 'resnet50.png', show_shapes=True, show_layer_names=True)



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
