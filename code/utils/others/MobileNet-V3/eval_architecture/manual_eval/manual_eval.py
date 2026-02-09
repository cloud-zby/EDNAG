import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import keras
from keras.datasets import cifar10
from keras import backend as K
from keras.layers import (
    Input,
    Conv2D,
    GlobalAveragePooling2D,
    Dense,
    BatchNormalization,
    Activation,
)
from keras.models import Model
from keras.layers import DepthwiseConv2D

from keras import optimizers, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.initializers import he_normal
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint

num_classes = 10
batch_size = 64
epochs = 500
iterations = 782
USE_BN = True
DROPOUT = 0.2  # keep 80%
CONCAT_AXIS = 3
weight_decay = 1e-4
DATA_FORMAT = "channels_last"  # Theano:'channels_first' Tensorflow:'channels_last'

log_filepath = os.path.join(os.getcwd(), "eval_architecture", "manual_eval", "logs")
# os.makedirs(log_filepath, exist_ok=True)  # Create the directory if it doesn't exist


def color_preprocessing(x_train, x_test):
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    mean = [125.307, 122.95, 113.865]
    std = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        x_train[:, :, :, i] = (x_train[:, :, :, i] - mean[i]) / std[i]
        x_test[:, :, :, i] = (x_test[:, :, :, i] - mean[i]) / std[i]
    return x_train, x_test


def scheduler(epoch):
    if epoch < 100:
        return 0.01
    if epoch < 200:
        return 0.001
    return 0.0001


# load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
x_train, x_test = color_preprocessing(x_train, x_test)


def depthwise_separable(ks, x, params):
    # f1/f2 filter size, s1 stride of conv
    (s1, f2) = params
    x = DepthwiseConv2D(
        (ks, ks),
        strides=(s1[0], s1[0]),
        padding="same",
        depthwise_initializer="he_normal",
        depthwise_regularizer=regularizers.l2(weight_decay),
    )(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation("relu")(x)
    x = Conv2D(
        int(f2[0]),
        (1, 1),
        strides=(1, 1),
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(weight_decay),
    )(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation("relu")(x)
    return x


def MobileNet(img_input, classes=10):
    """
    通过修改MobileNet类的每层depthwise_separable的卷积核大小和输出channel数，
    可以实现手动修改kernel_size和expand_list！
    通过修改MobileNet层数，可以实现手动修改depth_list！
    """
    # change stride
    x = Conv2D(
        int(32),
        (3, 3),
        strides=(1, 1),
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(weight_decay),
    )(
        img_input
    )  # change stride
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = depthwise_separable(3, x, params=[(1,), (64,)])
    x = depthwise_separable(3, x, params=[(1,), (128,)])  # change stride
    x = depthwise_separable(3, x, params=[(1,), (128,)])
    x = depthwise_separable(3, x, params=[(1,), (128,)])
    x = depthwise_separable(3, x, params=[(1,), (256,)])  # change stride
    x = depthwise_separable(3, x, params=[(1,), (256,)])
    x = depthwise_separable(3, x, params=[(1,), (256,)])
    x = depthwise_separable(3, x, params=[(2,), (512,)])
    for _ in range(5):
        x = depthwise_separable(3, x, params=[(1,), (512,)])

    x = depthwise_separable(3, x, params=[(2,), (1024,)])
    x = depthwise_separable(3, x, params=[(1,), (1024,)])

    x = GlobalAveragePooling2D()(x)
    out = Dense(classes, activation="softmax")(x)
    return out


class TestAccuracyCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        test_loss, test_acc = self.model.evaluate(x_test, y_test, verbose=0)
        print(f"\nEpoch {epoch + 1} - Test accuracy: {test_acc:.4f}")


def main():
    os.makedirs(log_filepath, exist_ok=True)
    img_input = Input(shape=(32, 32, 3))
    output = MobileNet(img_input)
    model = Model(img_input, output)
    model.summary()
    # set optimizer
    sgd = optimizers.SGD(learning_rate=0.1, momentum=0.9, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
    # set callback
    test_accuracy_cb = TestAccuracyCallback()
    tb_cb = TensorBoard(log_dir=log_filepath, histogram_freq=0)
    change_lr = LearningRateScheduler(scheduler)
    cbks = [change_lr, tb_cb, test_accuracy_cb]
    # set data augmentation
    datagen = ImageDataGenerator(
        horizontal_flip=True,
        width_shift_range=0.125,
        height_shift_range=0.125,
        fill_mode="constant",
        cval=0.0,
    )
    datagen.fit(x_train)
    # start training
    model.fit(
        datagen.flow(x_train, y_train, batch_size=batch_size),
        steps_per_epoch=iterations,
        epochs=epochs,
        callbacks=cbks,
        validation_data=(x_test, y_test),
    )
    model.save("mobilenet.h5")


# main()
