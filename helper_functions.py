import keras.backend
import numpy as np
import os
import glob
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage import transform
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation, \
    Conv2DTranspose, Concatenate, Input, SeparableConv2D, add, UpSampling2D, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


# This function will return the images and labels within the given folder
# labels are derived from the first pixel in the label img
# images are maximum value normalized so they are from 0-1 seems to be issues with this so this part is commented out
# param foldername: name of folder where files are located
# returns tuple of numpy arrays of imgs and labels
def dataset_reader(foldername):
    dirname = os.path.join(os.getcwd(), 'Data', foldername)
    images_path = glob.glob(dirname + "/images/*.tif")
    labels_path = glob.glob(dirname + "/labels/*.tif")

    numfiles = len(images_path)
    numlabels = len(labels_path)
    # Checking to make sure that there is the same number of labels and images
    assert numlabels == numfiles

    img = []
    label = []
    print("reading in %d images" % numfiles)

    for i in range(numfiles):
        # works like a percent bar to make sure function did not hang
        if i % 100 == 0:
            print("percent complete: {:.0%}".format((i / numfiles)), end="\r")
        im_temp = cv2.imread(images_path[i], cv2.IMREAD_UNCHANGED)
        im_temp = cv2.cvtColor(im_temp, cv2.COLOR_BGRA2RGBA)
        lbl_temp = cv2.imread(labels_path[i], cv2.IMREAD_UNCHANGED)
        im_norm = (im_temp - im_temp.min()) / (im_temp.max() - im_temp.min())  # max normalizing the image
        img.append(im_temp)
        label.append(lbl_temp[0, 0])  # label is made with the (0,0) pixel of label image

    img = np.asarray(img)
    label = np.asarray(label)

    return img, label


# This is a function similar to dataset_reader but ignores invalid labels for NAIP data.
def dataset_reader_naip(foldername):
    dirname = os.path.join(os.getcwd(), 'Data', foldername)
    images_path = glob.glob(dirname + "/images/*.tif")
    labels_path = glob.glob(dirname + "/labels/*.tif")
    numfiles = len(images_path)
    numlabels = len(labels_path)
    # Checking to make sure that there is the same number of labels and images

    assert numlabels == numfiles
    img = []
    label = []
    print("reading in %d images" % numfiles)

    for i in range(numfiles):
        # works like a percent bar to make sure function did not hang
        if i % 100 == 0:
            print("percent complete: {:.0%}".format((i / numfiles)), end="\r")

        # Read in label first to make sure corresponding image and label are valid.
        lbl_temp = cv2.imread(labels_path[i], cv2.IMREAD_UNCHANGED)
        if (lbl_temp[0, 0] != 0):
            label.append(lbl_temp[0, 0])

            im_temp = cv2.imread(images_path[i], cv2.IMREAD_UNCHANGED)
            im_temp = cv2.cvtColor(im_temp, cv2.COLOR_BGRA2RGBA)
            img.append(im_temp)

    print(f"read in {len(img)} valid images and labels")

    img = np.asarray(img)
    label = np.asarray(label)

    return img, label


def segmented_dataset_reader(foldername):
    dirname = os.path.join(os.getcwd(), 'Data', foldername)
    images_path = glob.glob(dirname + "/images/*.tif")
    labels_path = glob.glob(dirname + "/labels/*.tif")
    numfiles = len(images_path) // 5
    numlabels = len(labels_path) // 5
    # Checking to make sure that there is the same number of labels and images
    assert numlabels == numfiles
    img = []
    label = []
    print("reading in %d images" % numfiles)
    for i in range(numfiles):
        # works like a percent bar to make sure function did not hang
        if i % 100 == 0:
            print("percent complete: {:.0%}".format((i / numfiles)), end="\r")
        im_temp = cv2.imread(images_path[i], cv2.IMREAD_UNCHANGED)
        im_temp = cv2.cvtColor(im_temp, cv2.COLOR_BGRA2RGBA)
        im_temp = im_temp / 255.0
        lbl_temp = cv2.imread(labels_path[i], cv2.IMREAD_UNCHANGED)
        lbl_temp = cv2.cvtColor(lbl_temp, cv2.COLOR_BGRA2RGBA)
        lbl_temp = lbl_temp
        img.append(im_temp)
        label.append(lbl_temp)  # label is an image
    img = np.asarray(img)
    label = np.asarray(label)
    return img, label


def get_simple_model(input_shape):
    """
    This function should build a Sequential model according to the above specification. Ensure the 
    weights are initialised by providing the input_shape argument in the first layer, given by the
    function argument.
    """
    model = Sequential([
        Conv2D(filters=50, input_shape=input_shape, kernel_size=(5, 5), activation='relu', padding='SAME'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(filters=30, kernel_size=(5, 5), activation='relu', padding='SAME'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(units=100, activation='relu'),
        BatchNormalization(),
        Dense(units=50, activation='relu'),
        BatchNormalization(),
        Dense(units=3, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def double_conv_block(x, n_filters):
    # Conv2D then ReLU activation
    leaky_relu = keras.layers.LeakyReLU(alpha=0.2)
    x = layers.Conv2D(n_filters, 3, padding="same", activation=leaky_relu, kernel_initializer="he_normal",
                      kernel_regularizer="l1_l2", bias_regularizer="l1_l2")(x)
    # Conv2D then ReLU activation
    x = layers.Conv2D(n_filters, 3, padding="same", activation='elu', kernel_initializer="he_normal",
                      kernel_regularizer="l1_l2", bias_regularizer="l1_l2")(x)
    x = BatchNormalization()(x)
    return x


def downsample_block(x, n_filters):
    f = double_conv_block(x, n_filters)
    p = layers.MaxPool2D(2)(f)
    p = layers.Dropout(0.3)(p)
    return f, p


def upsample_block(x, conv_features, n_filters):
    # upsample
    x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
    # concatenate
    x = layers.concatenate([x, conv_features])
    # dropout
    x = layers.Dropout(0.3)(x)
    # Conv2D twice with ReLU activation
    x = double_conv_block(x, n_filters)
    return x


def build_unet_model():
    keras.backend.clear_session()
    # inputs
    inputs = layers.Input(shape=(128, 128, 3))

    # encoder: contracting path - downsample
    # 1 - downsample
    f1, p1 = downsample_block(inputs, 64)
    # 2 - downsample
    f2, p2 = downsample_block(p1, 128)
    # 3 - downsample
    f3, p3 = downsample_block(p2, 256)
    # 4 - downsample
    f4, p4 = downsample_block(p3, 512)

    # 5 - bottleneck
    bottleneck = double_conv_block(p4, 1024)

    # decoder: expanding path - upsample
    # 6 - upsample
    u6 = upsample_block(bottleneck, f4, 512)
    # 7 - upsample
    u7 = upsample_block(u6, f3, 256)
    # 8 - upsample
    u8 = upsample_block(u7, f2, 128)
    # 9 - upsample
    u9 = upsample_block(u8, f1, 64)

    # outputs
    outputs = layers.Conv2D(4, 1, padding="same", activation="softmax")(u9)

    # unet model with Keras Functional API
    unet_model = tf.keras.Model(inputs, outputs, name="U-Net")
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    unet_model.compile(optimizer=opt,
                       loss=loss,
                       metrics=['accuracy'])

    return unet_model


def reduce_lr():
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        min_delta=0.01,
        patience=2,
        min_lr=0.0000001,
        verbose = 1,
    )
    return reduce_lr

def early_stop():
    early_stopping = EarlyStopping(
                               monitor='val_loss',
                               patience = 8,
                               mode = 'max',
                               restore_best_weights=True,
                               verbose = 1
    )
    return early_stopping

def get_test_accuracy(model, test_images, test_labels):
    """Test model classification accuracy"""
    test_loss, test_acc = model.evaluate(x=test_images, y=test_labels, verbose=0)
    print('accuracy: {acc:0.3f}'.format(acc=test_acc))
    return test_acc


def get_train_accuracy(model, train_images, train_labels):
    """Train model classification accuracy"""
    train_loss, train_acc = model.evaluate(x=train_images, y=train_labels, verbose=0)
    print('accuracy: {acc:0.3f}'.format(acc=train_acc))
    return train_acc


def plot_accuracy(history):
    try:
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
    except KeyError:
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
    plt.title('Accuracy vs. epochs')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='lower right')
    plt.show()


def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss vs. epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper right')
    plt.show()
