from abc import ABC

import tensorflow as tf
from tensorflow import keras
from keras.models import Model, Sequential
from keras.layers import Layer
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import AveragePooling2D, Flatten, Dropout, Dense
from keras.activations import relu


class Residual(Layer):
    """The Residual block of ResNet."""

    def __init__(self, num_channels, use1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = Conv2D(num_channels, kernel_size=3, padding='same', strides=strides)
        self.conv2 = Conv2D(num_channels, kernel_size=3, padding='same')
        self.conv3 = None

        if use1x1conv:
            self.conv3 = Conv2D(num_channels, kernel_size=1, strides=strides)

        self.bn1 = BatchNormalization()
        self.bn2 = BatchNormalization()

    def call(self, X, training=None, mask=None):
        y = relu(self.bn1(self.conv1(X)))
        y = self.bn2(self.conv2(y))

        if self.conv3:
            X = self.conv3(X)
        y += X
        return relu(y)


class Bottleneck(Layer):
    """bottleneck block for resnet50/101/152"""

    def __init__(self, num_channels, use_projection_shortcut=False, strides=1):
        super().__init__()
        self.conv1 = Conv2D(num_channels, kernel_size=1, padding='valid', strides=strides)
        self.conv2 = Conv2D(num_channels, kernel_size=3, padding='same')
        self.conv3 = Conv2D(num_channels*4, kernel_size=1, padding='valid')
        self.conv4 = None
        if use_projection_shortcut:
            self.conv4 = Conv2D(num_channels*4, kernel_size=1, strides=strides)

        self.bn1 = BatchNormalization()
        self.bn2 = BatchNormalization()
        self.bn3 = BatchNormalization()
        self.bn4 = BatchNormalization()

    def call(self, inputs, training=None, mask=None):
        y = relu(self.bn1(self.conv1(inputs)))
        y = relu(self.bn2(self.conv2(y)))
        y = self.bn3(self.conv3(y))
        if self.conv4:
            inputs = self.bn4(self.conv4(inputs))
        y += inputs
        return relu(y)


class ResNet(Model, ABC):
    """ResNet model"""

    def __init__(self, arch, lr=0.1, num_classes=10):
        super().__init__()
        self.net = Sequential(self.b1())
        for i, b in enumerate(arch):
            self.net.add(self.block(*b, first_block=(i == 0)))
        self.net.add(Sequential([
            GlobalAveragePooling2D(),
            Dense(units=num_classes)
        ]))

    def b1(self):
        return Sequential([
            Conv2D(64, kernel_size=7, padding='same', strides=2),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=3, strides=2, padding='same')
        ])

    def block(self, num_residuals, num_channels, first_block=False):
        blk = Sequential()
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.add(Residual(num_channels, use1x1conv=True, strides=2))
            else:
                blk.add(Residual(num_channels))
        return blk

    def call(self, X, training=None, mask=None):
        y = self.net(X)
        return y


class FirstLayer(Layer):
    """GoogleNet and ResNet first layers"""
    def __init__(self):
        super(FirstLayer, self).__init__()
        self.conv = Conv2D(64, kernel_size=7, padding='same', strides=2, input_shape=(224, 224, 3))
        self.bn = BatchNormalization()
        self.max_pool = MaxPooling2D(pool_size=3, padding='same', strides=2)

    def call(self, inputs, *args, **kwargs):
        y = self.bn(self.conv(inputs))
        return self.max_pool(y)


class Classifier(Layer):
    """Classifier layer"""
    def __init__(self, num_classes, activation=None, dropout_rate=0.1):
        super(Classifier, self).__init__()
        self.global_pool = GlobalAveragePooling2D()
        self.dropout = Dropout(dropout_rate)
        self.fc = Dense(num_classes, activation=activation)

    def call(self, inputs, *args, **kwargs):
        x = self.global_pool(inputs)
        x = self.dropout(x)
        y = self.fc(x)
        return y


class ResnetModule(Layer):
    """Create deep residual modules"""
    def __init__(self, num_residuals, num_channels, first_block=False):
        super(ResnetModule, self).__init__()
        self.module = []
        for j in range(num_residuals):
            if j == 0 and not first_block:
                self.module.append(Bottleneck(num_channels, use_projection_shortcut=True, strides=2))
            elif j == 0 and first_block:
                self.module.append(Bottleneck(num_channels, use_projection_shortcut=True))
            else:
                self.module.append(Bottleneck(num_channels))

    def call(self, x, *args, **kwargs):
        for b in self.module:
            x = b(x)
        return x


class DeepResNet(Model):
    """Deeper Resnet model for 50 layers and more"""

    def __init__(self, arch, num_classes, dropout_rate):
        super(DeepResNet, self).__init__()
        self.first_layer = FirstLayer()
        self.classifier = Classifier(num_classes=num_classes, activation='softmax', dropout_rate=dropout_rate)
        self.modules = []

        for i, b in enumerate(arch):
            self.modules.append(ResnetModule(*b, first_block=(i == 0)))
        
    def call(self, inputs, training=None, mask=None):
        x = self.first_layer(inputs)
        for m in self.modules:
            x = m(x)
        return self.classifier(x)


class ResNet18(ResNet, ABC):
    """
    ResNet-18 is a subclass of ResNet contains 18 layers: 4 convolutional layers in each module together with
    the first 7x7 convolutional layer and the final fully connected layer.
    """

    def __init__(self, lr=0.1, num_classes=10):
        super().__init__(((2, 64), (2, 128), (2, 256), (2, 512)),
                         lr, num_classes)


class Resnet50(DeepResNet):
    """Create Resnet-50 using bottleneck building blocks"""
    
    def __init__(self, num_classes=2, dropout_rate=0.2):
        super(Resnet50, self).__init__(((3, 64), (4, 128), (6, 256), (3, 512)),
                                       num_classes=num_classes, dropout_rate=dropout_rate)
