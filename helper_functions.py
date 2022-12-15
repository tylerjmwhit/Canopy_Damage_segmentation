import keras.backend
import numpy as np
import os
import glob
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import gc
import pandas as pd
import seaborn as sns
from skimage import transform
import random
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation, \
    Conv2DTranspose, Concatenate, Input, SeparableConv2D, add, UpSampling2D, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow_examples.models.pix2pix import pix2pix
from skimage.morphology import disk
from skimage import morphology
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation, Conv2DTranspose, Concatenate, Input, SeparableConv2D, add, UpSampling2D
from sklearn.metrics import confusion_matrix
from tensorflow.keras import layers, regularizers
from tensorflow.keras.layers import Dropout
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

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
        im_temp = im_temp / 255.0
        lbl_temp = cv2.imread(labels_path[i], cv2.IMREAD_UNCHANGED)
        lbl_temp = cv2.cvtColor(lbl_temp, cv2.COLOR_BGRA2RGBA)
        img.append(im_temp)
        label.append(lbl_temp)  # label is an image
    img = np.asarray(img).astype(np.float32)
    label = np.asarray(label).astype('uint8')
    return img, label


def get_simple_model(input_shape):
    """
    This function should build a Sequential model according to the above specification. Ensure the 
    weights are initialised by providing the input_shape argument in the first layer, given by the
    function argument.
    """
    model = Sequential([
        Conv2D(filters=50, input_shape=input_shape, kernel_size=(5, 5), activation='relu', padding='SAME',
               kernel_initializer="he_normal",
               kernel_regularizer="l1_l2", bias_regularizer="l1_l2"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(filters=30, kernel_size=(5, 5), activation='relu', padding='SAME', kernel_initializer="he_normal",
               kernel_regularizer="l1_l2", bias_regularizer="l1_l2"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(units=100, activation='relu', kernel_initializer="he_normal",
              kernel_regularizer="l1_l2", bias_regularizer="l1_l2"),
        BatchNormalization(),
        Dense(units=50, activation='relu', kernel_initializer="he_normal",
              kernel_regularizer="l1_l2", bias_regularizer="l1_l2"),
        BatchNormalization(),
        Dense(units=3, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def double_conv_block(x, n_filters):
    # Conv2D then ReLU activation
    x = layers.SeparableConv2D(n_filters, 3, padding="same", activation='relu', depthwise_initializer="he_normal",
                               pointwise_initializer="he_normal")(x)
    # Conv2D then ReLU activation
    x = layers.SeparableConv2D(n_filters, 3, padding="same", activation='relu', depthwise_initializer="he_normal",
                               pointwise_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    return x


def downsample_block(x, n_filters):
    f = double_conv_block(x, n_filters)
    p = layers.MaxPool2D(2)(f)
    p = layers.Dropout(0.3)(p)
    return f, p


def upsample_block(x, conv_features, n_filters):
    # upsample
    x = pix2pix.upsample(n_filters, 3)(x)
    # x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
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
    f1, p1 = downsample_block(inputs, 32)
    # 2 - downsample
    f2, p2 = downsample_block(p1, 64)
    # 3 - downsample
    f3, p3 = downsample_block(p2, 128)
    # 4 - downsample
    f4, p4 = downsample_block(p3, 256)

    f5, p5 = downsample_block(p4, 512)

    # 5 - bottleneck
    bottleneck = double_conv_block(p5, 1024)

    # decoder: expanding path - upsample
    u5 = upsample_block(bottleneck, f5, 512)
    # 6 - upsample
    u6 = upsample_block(u5, f4, 256)
    # 7 - upsample
    u7 = upsample_block(u6, f3, 128)
    # 8 - upsample
    u8 = upsample_block(u7, f2, 64)
    # 9 - upsample
    u9 = upsample_block(u8, f1, 32)

    # outputs
    outputs = layers.Conv2DTranspose(4, 1, padding="same")(u9)
    #outputs = layers.Conv2D(4, 3, padding="same",activation='softmax')(u9)

    # unet model with Keras Functional API
    unet_model = tf.keras.Model(inputs, outputs, name="U-Net")
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    met = tf.keras.metrics.MeanIoU(num_classes=4, sparse_y_pred=False)
    unet_model.compile(optimizer=opt,
                       loss=loss,
                       metrics=['accuracy', met])

    return unet_model


def mobile_unet_model(output_channels: int):
    keras.backend.clear_session()
    inputs = tf.keras.layers.Input(shape=[128, 128, 3])
    base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

    # Use the activations of these layers
    layer_names = [
        'block_1_expand_relu',  # 64x64
        'block_3_expand_relu',  # 32x32
        'block_6_expand_relu',  # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',  # 4x4
    ]
    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

    down_stack.trainable = True
    for layer in down_stack.layers[:-7]:
        layer.trainable = False
        # Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])
    up_stack = [
        pix2pix.upsample(512, 3),  # 4x4 -> 8x8
        pix2pix.upsample(256, 3),  # 8x8 -> 16x16
        pix2pix.upsample(128, 3),  # 16x16 -> 32x32
        pix2pix.upsample(64, 3),  # 32x32 -> 64x64
    ]
    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])
        # Trying this commented out next
        x = BatchNormalization()(x)

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
        filters=output_channels, kernel_size=3, strides=2,
        padding='same', activation='softmax')  # 64x64 -> 128x128

    x = last(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    met = tf.keras.metrics.MeanIoU(num_classes=4, sparse_y_pred=False)
    model.compile(optimizer=opt,
                  loss=loss,
                  metrics=['accuracy', met])
    return model


def convolution_block(
        block_input,
        num_filters=256,
        kernel_size=3,
        dilation_rate=1,
        padding="same",
        use_bias=False,
):
    x = layers.SeparableConv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=keras.initializers.HeNormal(),
    )(block_input)
    x = layers.BatchNormalization()(x)
    return tf.nn.relu(x)


def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output

def DeeplabV3Plus(image_size, num_classes):
    keras.backend.clear_session()
    model_input = keras.Input(shape=(image_size, image_size, 3))
    resnet50 = keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_tensor=model_input
    )
    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPooling(x)

    input_a = layers.UpSampling2D(
        size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = layers.UpSampling2D(
        size=(image_size // x.shape[1], image_size // x.shape[2]),
        interpolation="bilinear",
    )(x)
    model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same")(x)
    model = keras.Model(inputs=model_input, outputs=model_output)
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    met = tf.keras.metrics.MeanIoU(num_classes=4, sparse_y_pred=False)
    model.compile(optimizer=opt,
                  loss=loss,
                  metrics=['accuracy', met])
    return model

def reduce_lr():
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.4,
        min_delta=0.01,
        patience=3,
        min_lr=0.0000001,
        verbose=1,
    )
    return reduce_lr


def early_stop():
    early_stopping = EarlyStopping(
        monitor='val_mean_io_u',
        min_delta=0.001,
        patience=6,
        mode='max',
        restore_best_weights=True,
        verbose=1,

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


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask


def display(display_list, titles=None):
    plt.figure(figsize=(20, 20))
    if titles is None:
        title = ["Input Image", "True Mask", "DeepLab", "MobileNet", "Simple U-Net", "Soft Voting Mask"]
    else:
        title = ["Input Image", "True Mask", titles]
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(display_list[i])
        plt.axis("off")
    plt.show()


def show_predictions(mod, img=None, label=None, num=1, titles=None):
    if img is not None:
        for i in range(num):
            pred_mask = mod.predict(img, verbose = 0)
            footprint = disk(4)
            mask = np.array(create_mask(pred_mask[i])).reshape(128,128)
            mask = morphology.closing(mask, footprint)
            display([img[i], label[i], mask], titles)


def voting(model_names, t_images, t_labels, offset=10, num=3):
    soft = np.empty(t_labels.shape + (4,))
    hard = []
    miou = tf.keras.metrics.MeanIoU(num_classes=4)
    for model_name in model_names:
        keras.backend.clear_session()
        gc.collect()
        model = keras.models.load_model(model_name)
        preds = model.predict(t_images, verbose=0)
        mask = create_mask(preds)
        miou.reset_state()
        miou.update_state(t_labels, mask)
        print(model_name)
        print(miou.result().numpy())
        soft = soft + preds
        hard.append(mask)
    s_vote = create_mask(soft)
    hard = np.array(hard)
    miou.reset_state()
    miou.update_state(t_labels, s_vote)
    print('s_voting')
    print(miou.result().numpy())
    footprint = disk(4)
    for i in range(num):
        hard0 = morphology.closing(hard[0, i + offset].reshape(128,128), footprint)
        hard1 = morphology.closing(hard[1, i + offset].reshape(128,128), footprint)
        hard2 = morphology.closing(hard[2, i + offset].reshape(128,128), footprint)
        s_vote1 = morphology.closing(np.array(s_vote[i + offset]).reshape(128,128), footprint)
        display([t_images[i + offset], t_labels[i + offset], hard0, hard1,hard2 ,s_vote1])
