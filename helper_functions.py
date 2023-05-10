
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
import rasterio
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation, Conv2DTranspose, Concatenate, Input, SeparableConv2D, add, UpSampling2D, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow_examples.models.pix2pix import pix2pix
from skimage.morphology import disk
from skimage import morphology
from focal_loss import SparseCategoricalFocalLoss
from tensorflow.keras.applications import mobilenet_v2 , resnet50

#functions to read in all the segmentation dataset
def segmented_dataset_reader(foldername, train_bool= False):
    dirname = os.path.join(os.getcwd(), 'Data', foldername)
    images_path = glob.glob(dirname + "/images/*.tif")
    labels_path = glob.glob(dirname + "/labels/*.tif")
    numfiles = len(images_path)
    numlabels = len(labels_path)
    # Checking to make sure that there is the same number of labels and images
    assert numlabels == numfiles
    img = []
    img_geo = []
    label = []
    print("reading in %d images" % numfiles)
    for i in range(numfiles):
        # works like a percent bar to make sure function did not hang
        if i % 100 == 0:
            print("percent complete: {:.0%}".format((i / numfiles)), end="\r")
        dataset = rasterio.open(images_path[i])
        im_temp = dataset.read([1,2,3]).swapaxes(2,0)
        lbl_temp = cv2.imread(labels_path[i], cv2.IMREAD_UNCHANGED)
        lbl_temp = lbl_temp.astype(np.uint8)
        # if train_bool == True and tf.random.uniform(()) > 0.5:
        #      # Random flipping of the image and mask
        #     img.append((np.fliplr(im_temp)))
        #     label.append((np.fliplr(lbl_temp)))
        img.append(im_temp)
        label.append(lbl_temp)# label is an image
        if train_bool == False:
            im_temp_geo = dataset.transform
            img_geo.append(im_temp_geo)
    img = np.asarray(img).astype(np.float32) 
    label = np.asarray(label).astype('uint8')
    if train_bool == False:
        return img, img_geo, label
    return img, label

def double_conv_block(x, n_filters, act, act2):
# functions to generate two Separableconv2d layers
# uses Separableconv2d to save space and use less parameters
    # Conv2D then ReLU activation
    x = layers.SeparableConv2D(n_filters, 3, padding="same", activation=act, depthwise_initializer="he_normal",
                               pointwise_initializer="he_normal")(x)
    # Conv2D then ReLU activation
    x = layers.SeparableConv2D(n_filters, 3, padding="same", activation=act2, depthwise_initializer="he_normal",
                               pointwise_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    return x

#function to downsample the images using maxpooling
def downsample_block(x, n_filters, drop, act, act2):
    f = double_conv_block(x, n_filters, act, act2)

    p = layers.MaxPool2D(2)(f)
    p = layers.Dropout(drop)(p)
    return f, p

def upsample_block(x, conv_features, n_filters, drop, act, act2,  dropout = False):
#function to upsample the images
#uses pix2pix tensorflow model
    # upsample
    x = pix2pix.upsample(n_filters, 3)(x)
    # concatenate
    x = layers.concatenate([x, conv_features])
    # dropout
    if dropout:
        x = layers.Dropout(drop)(x)
    # Conv2D twice with ReLU activation
    x = double_conv_block(x, n_filters, act=act, act2=act2)
    return x


#function to build unet from component parts
def build_unet_model(num_class, weights, act2,  act='elu',
                      drop=0.3, drop2=0.2, drop_bool=True, drop_bool2=False,
                     filter=48, gamma=1, lr=0.003):
    keras.backend.clear_session()
    gc.collect()
    # inputs
    inputs = layers.Input(shape=(128, 128, 3))
    # encoder: contracting path - downsample
    # 1 - downsample
    f1, p1 = downsample_block(inputs, filter, drop=drop, act=act, act2=act2)
    # 2 - downsample
    f2, p2 = downsample_block(p1, filter * 2, drop=drop, act=act, act2=act2)
    # 3 - downsample
    f3, p3 = downsample_block(p2, filter * 4, drop=drop, act=act, act2=act2)
    # 4 - downsample
    f4, p4 = downsample_block(p3, filter * 8, drop=drop2, act=act, act2=act2)

    f5, p5 = downsample_block(p4, filter * 16, drop=drop2, act=act, act2=act2)

    # 5 - bottleneck
    bottleneck = double_conv_block(p5, 1024, act=act, act2=act2)

    # decoder: expanding path - upsample

    u5 = upsample_block(bottleneck, f5, filter * 16, drop=drop2, act=act, act2=act2)
    # 6 - upsample
    u6 = upsample_block(u5, f4, filter * 8, drop=drop2, act=act, act2=act2)
    # 7 - upsample
    u7 = upsample_block(u6, f3, filter * 4, drop=drop, dropout=drop_bool2, act=act, act2=act2)
    # 8 - upsample
    u8 = upsample_block(u7, f2, filter * 2, drop=drop, dropout=drop_bool2, act=act, act2=act2)
    # 9 - upsample
    u9 = upsample_block(u8, f1, filter, drop=drop, dropout=drop_bool, act=act, act2=act2)

    # outputs
    outputs = layers.Conv2DTranspose(num_class, 1, padding="same")(u9)

    # unet model with Keras Functional API
    unet_model = tf.keras.Model(inputs, outputs, name="U-Net")
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = SparseCategoricalFocalLoss(gamma=gamma, class_weight=weights,from_logits=True)  # true since not using softmax
    met = tf.keras.metrics.MeanIoU(num_classes=num_class, sparse_y_pred=False)
    unet_model.compile(optimizer=opt,
                       loss=loss,
                       metrics=['accuracy', met])

    return unet_model


#functions for building the mobilenet version of the unet
def mobile_unet_model(output_channels: int, weights,
                       trainable = 2, batch_bool = True, lr = 0.004, gamma = 1):
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
    for layer in down_stack.layers[:-trainable]:
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
        if batch_bool:
            x = BatchNormalization()(x)

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
        filters=output_channels, kernel_size=3, strides=2,
        padding='same', activation='softmax')  # 64x64 -> 128x128

    x = last(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = SparseCategoricalFocalLoss(gamma=gamma, class_weight=weights,from_logits=False)
    met = tf.keras.metrics.MeanIoU(num_classes=output_channels, sparse_y_pred=False)
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
        batch_bool = True,
        dropout = 0.5
):
    x = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=keras.initializers.HeNormal(),
    )(block_input)
    if batch_bool:
        x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    return tf.nn.relu(x)


def DilatedSpatialPyramidPooling(dspp_input, drop_rate = 0.5, batch_bool = True):
    dims = dspp_input.shape
    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True, dropout=drop_rate, batch_bool = batch_bool)
    out_pool = layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1, dropout=drop_rate, batch_bool = batch_bool)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6, dropout=drop_rate, batch_bool = batch_bool)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12, dropout=drop_rate, batch_bool = batch_bool)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18, dropout=drop_rate, batch_bool = batch_bool)

    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1, dropout=drop_rate, batch_bool = batch_bool)
    return output


def DeeplabV3Plus(image_size, num_classes, weight,
                   drop_rate= 0.4, drop_rate2=0.2, batch_bool = True, batch_bool2 = True, lr = 0.005, gamma = 1):
#DeeplabV3 model
    keras.backend.clear_session()
    model_input = keras.Input(shape=(image_size, image_size, 3))
    resnet50 = keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_tensor=model_input
    )
    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPooling(x, drop_rate = drop_rate, batch_bool = batch_bool)

    input_a = layers.UpSampling2D(
        size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1, dropout=drop_rate2, batch_bool = batch_bool2)

    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x, dropout=drop_rate2, batch_bool = batch_bool2)
    x = convolution_block(x, dropout=drop_rate2, batch_bool = batch_bool2)
    x = layers.UpSampling2D(
        size=(image_size // x.shape[1], image_size // x.shape[2]),
        interpolation="bilinear",
    )(x)
    model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same")(x)
    model = keras.Model(inputs=model_input, outputs=model_output)
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = SparseCategoricalFocalLoss(from_logits=True, class_weight=weight, gamma=gamma)
    met = tf.keras.metrics.MeanIoU(num_classes=num_classes, sparse_y_pred=False)
    model.compile(optimizer=opt,
                  loss=loss,
                  metrics=['accuracy', met])
    return model


###callbacks
def reduce_lr():
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_mean_io_u",
        factor=0.5,
        min_delta=0.001,
        patience=3,
        min_lr=0.000001,
        verbose=1,
    )
    return reduce_lr


def early_stop():
    early_stopping = EarlyStopping(
        monitor='val_mean_io_u',
        min_delta=0.001,
        patience=8,
        mode='max',
        restore_best_weights=True,
        verbose=1,

    )
    return early_stopping


###result display
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

def plot_meaniou(history):
    plt.plot(history.history['mean_io_u'])
    plt.plot(history.history['val_mean_io_u'])
    plt.title('mean_iou vs. epochs')
    plt.ylabel('mean_iou')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper right')
    plt.show()

def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss vs. epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper right')
    plt.show() 

def conf_mat(model, test_images, test_labels):
    categories = ["1", "2", "3"]

    plt.figure(figsize=(15, 5))

    rounded_predictions = np.argmax(model.predict(test_images, batch_size=128, verbose=0), axis=1)
    rounded_labels = np.argmax(test_labels, axis=1)

    cm = confusion_matrix(rounded_labels, rounded_predictions)
    df_cm = pd.DataFrame(cm, index=categories, columns=categories)

    plt.title("Confusion matrix\n")
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="YlGnBu")
    plt.ylabel("Predicted")
    plt.xlabel("Actual")
    plt.show()

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return np.array(pred_mask)
    
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
            pred_mask = mod.predict(img, verbose=0)
            footprint = disk(4)
            mask = np.array(create_mask(pred_mask[i])).reshape(128, 128)
            mask = morphology.closing(mask, footprint)
            display([img[i], label[i], mask], titles)

def voting(model_names, t_images, t_labels, offset=10, num=3, numclasses=6):
    soft = np.empty(t_labels.shape + (6,))
    hard = []
    miou = tf.keras.metrics.MeanIoU(num_classes=numclasses)
    footprint = morphology.disk(radius=4)
    for _model_name in model_names:
        print(_model_name)
        keras.backend.clear_session()
        gc.collect()
        temp_images = np.copy(t_images)
        if 'deeplab' in _model_name:
            temp_images /= 255.0
        elif "mobile" in _model_name:
            temp_images = mobilenet_v2.preprocess_input(temp_images)
        else:
            temp_images /= 255.0
        model = keras.models.load_model(_model_name)
        preds = model.predict(temp_images, verbose=0)
        mask = create_mask(preds)
        soft = soft + preds
        hard.append(mask)
        del model
        del mask
    keras.backend.clear_session()
    gc.collect()
    s_vote = create_mask(soft)
    for i in range(len(s_vote)):
        if i % 100 == 0:
            print("percent complete: {:.0%}".format((i / len(s_vote))), end="\r")
        x = morphology.closing(s_vote[i].reshape(128, 128), footprint).reshape(128,128,1)
        s_vote[i] = x
    hard = np.array(hard)
    miou.reset_state()
    miou.update_state(t_labels, s_vote)
    print('s_voting')
    print(miou.result().numpy())
    for i in range(num):
        hard0 = morphology.closing(hard[0, i + offset].reshape(128, 128), footprint)
        hard1 = morphology.closing(hard[1, i + offset].reshape(128, 128), footprint)
        hard2 = morphology.closing(hard[2, i + offset].reshape(128, 128), footprint)
        s_vote1 = np.array(s_vote[i + offset])
        display([t_images[i + offset]/255.0, t_labels[i + offset], hard0, hard1, hard2, s_vote1])
    return s_vote

def save_geo(masks, georef):
    for i in range(len(georef)):
        file_name = 'predictions/' + str(i) +'.tif'
        #print(file_name)
        temp_data = rasterio.open(
            file_name,
            'w',
            driver = 'GTiff',
            height = 128,
            width = 128,
            count = 1,
            dtype = masks.dtype,
            crs = 'EPSG:32610',
            transform = georef[i]
        )
        temp_data.write(masks[i].reshape(1,128,128))
        temp_data.close()
