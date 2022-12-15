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
# images are maximum value normalized so they are from 0-1
# param foldername: name of folder where files are located
# returns tuple of numpy arrays of imgs and labels
def dataset_reader_planetscope(foldername):
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
        im_temp = cv2.imread(images_path[i],cv2.IMREAD_UNCHANGED)
        im_temp = cv2.cvtColor(im_temp, cv2.COLOR_BGRA2RGBA)
        
        # Resize images from 30x30 to 32x32
        im_temp = cv2.resize(im_temp, (32, 32), interpolation=cv2.INTER_LINEAR)
        
        lbl_temp = cv2.imread(labels_path[i], cv2.IMREAD_UNCHANGED)
        #im_norm = im_temp / im_temp.max() # max normalizing the image
        img.append(im_temp)
        label.append(lbl_temp[0,0]) # label is made with the (0,0) pixel of label image

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
        if (lbl_temp[0,0] != 0):
            label.append(lbl_temp[0,0])

            im_temp = cv2.imread(images_path[i],cv2.IMREAD_UNCHANGED)
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
        im_temp = cv2.imread(images_path[i],cv2.IMREAD_UNCHANGED)
        im_temp = cv2.cvtColor(im_temp, cv2.COLOR_BGRA2RGBA)
        lbl_temp = cv2.imread(labels_path[i], cv2.IMREAD_UNCHANGED)
        lbl_temp = cv2.cvtColor(lbl_temp, cv2.COLOR_BGRA2RGBA)
        im_temp = im_temp/255.0
        #im_norm = im_temp / im_temp.max() # max normalizing the image
        img.append(im_temp)
        label.append(lbl_temp) # label is an image
    img = np.asarray(img)
    label = np.asarray(label)
    label = label[: , : , : ,0]/255
    return img, label

def get_simple_model(input_shape):
    """
    This function should build a Sequential model according to the above specification. Ensure the 
    weights are initialised by providing the input_shape argument in the first layer, given by the
    function argument.
    """
    wd = 0.0001
    rate = 0.2

    model = Sequential([
        Conv2D(filters = 50, input_shape = input_shape, kernel_size = (5, 5), activation = 'relu', padding = 'SAME', kernel_initializer = tf.keras.initializers.HeNormal(), bias_initializer=tf.keras.initializers.Constant(1.), kernel_regularizer = regularizers.l2(wd)),
        BatchNormalization(),
        Dropout(rate),
        MaxPooling2D(pool_size = (2,2)),
        Conv2D(filters = 30, kernel_size = (5, 5), activation = 'relu', padding = 'SAME'),
        BatchNormalization(),
        Dropout(rate),
        MaxPooling2D(pool_size = (2,2)),
        Flatten(),
        Dense(units = 100, activation = 'relu'),
        BatchNormalization(),
        Dense(units = 50, activation = 'relu'),
        BatchNormalization(),
        Dropout(rate),
        Dense(units = 3, activation = 'softmax')
    ])
    model.compile(optimizer = 'adam',
                 loss = 'categorical_crossentropy',
                 metrics = ['accuracy'])
    return model

def get_simple_unet_model(img_size):
    inputs = tf.keras.Input(shape=img_size)

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = Activation("relu")(x)
        x = SeparableConv2D(filters, 3, padding="same")(x)
        x = BatchNormalization()(x)

        x = Activation("relu")(x)
        x = SeparableConv2D(filters, 3, padding="same")(x)
        x = BatchNormalization()(x)

        x = MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = Activation("relu")(x)
        x = Conv2DTranspose(filters, 3, padding="same")(x)
        x = BatchNormalization()(x)

        x = Activation("relu")(x)
        x = Conv2DTranspose(filters, 3, padding="same")(x)
        x = BatchNormalization()(x)

        x = UpSampling2D(2)(x)

        # Project residual
        residual = UpSampling2D(2)(previous_block_activation)
        residual = Conv2D(filters, 1, padding="same")(residual)
        x = add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = Conv2D(1, 7, activation="softmax", padding="valid")(x)

    # Define the model
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer = 'rmsprop',
                 loss = 'categorical_crossentropy',
                 metrics = ['accuracy'])
    return model

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





"""
Stuff for oversampling and data augmentation.
"""
def rotate(img, angle):
    angle = int(random.uniform(-angle, angle))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))
    return img

def horizontal_flip(img):
    img = cv2.flip(img, 1)
    return img

def vertical_flip(img):
    img = cv2.flip(img, 0)
    return img

def augment_label_1(label_1_rows):
    new_L1_images = []
    new_L1_labels = []
    for index, row in label_1_rows.iterrows():
        img = row["images"]

        new_L1_images.append(img)
        new_L1_labels.append(1)

        rotate_90 = rotate(img, 90)
        new_L1_images.append(rotate_90)
        new_L1_labels.append(1)

        h_flip = horizontal_flip(img)
        new_L1_images.append(h_flip)
        new_L1_labels.append(1)

        v_flip = vertical_flip(img)
        new_L1_images.append(v_flip)
        new_L1_labels.append(1)
    new_L1_dict = {"images": new_L1_images, "labels": new_L1_labels}
    return pd.DataFrame(new_L1_dict)

def augment_label_2(label_2_rows):
    new_L2_images = []
    new_L2_labels = []
    for index, row in label_2_rows.iterrows():
        img = row["images"]

        new_L2_images.extend([img, img, img, img, img, img, img, img, img, img])
        new_L2_labels.extend([2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

        rotate_90 = rotate(img, 90)
        new_L2_images.extend([rotate_90, rotate_90, rotate_90, rotate_90, rotate_90, rotate_90, rotate_90, rotate_90, rotate_90, rotate_90])
        new_L2_labels.extend([2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

        rotate_180 = rotate(img, 180)
        new_L2_images.extend([rotate_180, rotate_180, rotate_180, rotate_180, rotate_180, rotate_180, rotate_180, rotate_180, rotate_180, rotate_180])
        new_L2_labels.extend([2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
        new_L2_images.extend([rotate_180, rotate_180, rotate_180, rotate_180, rotate_180, rotate_180, rotate_180, rotate_180, rotate_180, rotate_180])
        new_L2_labels.extend([2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

        h_flip = horizontal_flip(img)
        new_L2_images.extend([h_flip, h_flip, h_flip, h_flip, h_flip, h_flip, h_flip, h_flip, h_flip, h_flip])
        new_L2_labels.extend([2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
        new_L2_images.extend([h_flip, h_flip, h_flip, h_flip, h_flip, h_flip, h_flip, h_flip, h_flip, h_flip])
        new_L2_labels.extend([2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

        v_flip = vertical_flip(img)
        new_L2_images.extend([v_flip, v_flip, v_flip, v_flip, v_flip, v_flip, v_flip, v_flip, v_flip, v_flip])
        new_L2_labels.extend([2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
        new_L2_images.extend([v_flip, v_flip, v_flip, v_flip, v_flip, v_flip, v_flip, v_flip, v_flip, v_flip])
        new_L2_labels.extend([2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

    new_L2_dict = {"images": new_L2_images, "labels": new_L2_labels}
    return pd.DataFrame(new_L2_dict)





"""
Stuff for TL models.
"""
def get_TL_model(input_shape):
    featureExtractor = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape, pooling='avg')
    model = Sequential([
        featureExtractor,
        layers.Dense(64, activation = 'relu'),
        layers.Dropout(0.5),
        layers.Dense(3, activation = 'softmax')
    ]) 
    model.layers[0].trainable = False
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model


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

