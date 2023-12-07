##TODO clean up imports
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
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow_examples.models.pix2pix import pix2pix
from skimage.morphology import disk
from skimage import morphology
from focal_loss import SparseCategoricalFocalLoss
from tensorflow.keras.applications import mobilenet_v2, resnet50

#functions to read in all the segmentation dataset
def segmented_dataset_reader(foldername, train_bool= False, factor = 1):
    # Reads in a set of segmented images and their labels.
    # Args:
    #     foldername (str): The name of the folder containing the images and labels.
    #                           
    #                       expects the folder to be in a folder called Data and
    #                       contain a subfolder called images and another called labels
    #                       images and labels should be stored as tifs
    #     train_bool (bool): A flag indicating whether the function is called for training or testing.
    #     factor (int): an int that causing only a certain factor of dataset to be read in. Useful for only testing
    #                   on parts of the dataset

    # Returns:
    #     tuple: If train_bool is False, returns a tuple containing the image data, image geo-reference data, and label data.
    #         The image data is a 4D array of shape (numfiles, height, width, channels),
    #         the image geo-reference is a list of affine transformations corresponding to each image,
    #         and the label data is a 3D array of shape (numfiles, height, width).
    #         If train_bool is True, returns a tuple containing the image data and label data.

    # Raises:
    #     AssertionError: If the number of labels does not match the number of images.

    # Example:
    #     img_data, img_geo, lbl_data = segmented_dataset_reader('train', False)

    # This function works as follows:
    # 1. Constructs the path to the dataset folder.
    # 2. Retrieves the paths to image and label files using glob.
    # 3. Calculates the number of files in the dataset based on the factor.
    # 4. Performs an assertion check to ensure the number of labels matches the number of images.
    # 5. Initializes empty lists to store image data, image geo-reference data, and label data.
    # 6. Loops over the files in the dataset.
    #     - Opens each image file using rasterio and reads the first three bands.
    #     - Reshapes the image array to (channels, height, width).
    #     - Reads the label image using cv2.imread with the flag cv2.IMREAD_UNCHANGED.
    #     - Converts the label data type to uint8.
    #     - Appends the image and label data to the respective lists.
    #     - If not in training mode, retrieves and stores the image geo-reference data.
    # 7. Converts the lists to NumPy arrays.
    # 8. Returns the appropriate data based on the train_bool flag.

    dirname = os.path.join(os.getcwd(), 'Data', foldername)
    images_path = glob.glob(dirname + "/images/*.tif")
    labels_path = glob.glob(dirname + "/labels/*.tif")
    numfiles = len(images_path) // factor
    numlabels = len(labels_path) // factor
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
        ## can change to 1, 2, 4 to read in NIR
        im_temp = dataset.read([1,2,3]).swapaxes(2,0) ## reads in as BGR this swap to RGB
        lbl_temp = cv2.imread(labels_path[i], cv2.IMREAD_UNCHANGED)
        lbl_temp = lbl_temp.astype(np.uint8)
        img.append(im_temp)
        label.append(lbl_temp)
        if not train_bool:
            im_temp_geo = dataset.transform
            img_geo.append(im_temp_geo)
    img = np.asarray(img).astype(np.float32) 
    label = np.asarray(label).astype('uint8')
    if  not train_bool:
        return img, img_geo, label
    return img, label


##functions for simple UNET
def double_conv_block(x, n_filters, act, act2):
    #
    #     Constructs a double convolutional block consisting of two separable convolution layers.

    #     Args:
    #         x (tensor): Input tensor to the double conv block.
    #         n_filters (int): Number of filters for each convolution layer.
    #         act (str or callable): Activation function for the first convolution layer.
    #         act2 (str or callable): Activation function for the second convolution layer.

    #     Returns:
    #         tensor: Output tensor from the double conv block.

    #     Example:
    #         >>> x = double_conv_block(x, 64, 'relu', 'relu')
    # 
    # The function works as follows:
    # 1. Performs the first separable convolution layer using the provided n_filters, a kernel size
    #    of 3x3, "same" padding, and the specified activation function act.
    # 2. Performs the second separable convolution layer in a similar manner, using the activation function act2.
    # 3. Applies batch normalization to the output tensor.
    # 4. Returns the resulting tensor as the output of the double conv block

    x = layers.SeparableConv2D(n_filters, 3, padding="same", activation=act,
                               depthwise_initializer="he_normal", pointwise_initializer="he_normal")(x)
   
    x = layers.SeparableConv2D(n_filters, 3, padding="same", activation=act2,
                               depthwise_initializer="he_normal", pointwise_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    return x

def downsample_block(x, n_filters, drop, act, act2):
    #
    # Constructs a downsample block that performs downsampling using max pooling.

    # Args:
    #     x (tensor): Input tensor to the downsample block.
    #     n_filters (int): Number of filters for the double conv block.
    #     drop (float): Dropout rate applied to the max pooled tensor.
    #     act (str or callable): Activation function for the double conv block's first convolution layer.
    #     act2 (str or callable): Activation function for the double conv block's second convolution layer.

    # Returns:
    #     tuple: A tuple containing the output tensor from the double conv block and the downsampled tensor.

    # Example:
    #     >>> feature_map, downsampled_map = downsample_block(x, 64, 0.2, 'relu', 'relu')
    # 

    # The function works as follows:
    # 1. Constructs a double conv block using double_conv_block().
    # 2. Performs max pooling on the output tensor of the double conv block.
    # 3. Applies dropout to the max-pooled tensor.
    # 4. Returns a tuple containing the output tensor from the double conv block and the downsampled tensor.
    
    f = double_conv_block(x, n_filters, act, act2)
    p = layers.MaxPool2D(2)(f)
    p = layers.Dropout(drop)(p)
    return f, p

def upsample_block(x, conv_features, n_filters, drop, act, act2, dropout=False):
    # 
    # Constructs an upsample block that performs upsampling using the pix2pix TensorFlow model.

    # Args:
    #     x (tensor): Input tensor to the upsample block.
    #     conv_features (tensor): Tensor representing the features from the corresponding downsample block.
    #     n_filters (int): Number of filters for the double conv block.
    #     drop (float): Dropout rate applied to the upsampled tensor.
    #     act (str or callable): Activation function for the double conv block's first convolution layer.
    #     act2 (str or callable): Activation function for the double conv block's second convolution layer.
    #     dropout (bool, optional): Flag indicating whether to apply dropout. Defaults to False.

    # Returns:
    #     tensor: Output tensor from the upsample block.

    # Example:
    #     >>> upsampled_features = upsample_block(x, conv_features, 64, 0.2, 'relu', 'relu', dropout=True)
    # 
    # How the function works step by step:
    # 1. Performs upsampling using the pix2pix TensorFlow model, increasing the spatial dimensions of the input tensor.
    # 2. Concatenates the upsampled tensor with the convolutional features from the corresponding downsample block.
    # 3. If the `dropout` flag is set to True, applies dropout to the concatenated tensor.
    # 4. Passes the concatenated tensor through a double conv block with the specified number of filters and activation functions.
    # 5. Returns the resulting tensor as the output of the upsample block.

    x = pix2pix.upsample(n_filters, 3)(x)

    x = layers.concatenate([x, conv_features])

    if dropout:
        x = layers.Dropout(drop)(x)

    x = double_conv_block(x, n_filters, act=act, act2=act2)

    return x

def build_unet_model(num_class, weights = [1,1,1,1,1,1], act2 = keras.layers.LeakyReLU(),  act='elu',
                      drop=0.3, drop2=0.2, drop_bool=True, drop_bool2=False,
                     filter=48, gamma=1, lr=0.003):
    # Builds a U-Net model using the specified parameters.

    # Args:
    #     num_class (int): Number of output classes.
    #     weights: Weights for each class.
    #     act2 (str): Activation function for the second activation block (default: 'Leaky_relu').
    #     act (str): Activation function for the other blocks (default: 'elu').
    #     drop (float): Dropout rate for downsample blocks (default: 0.3).
    #     drop2 (float): Dropout rate for bottleneck and upsample blocks (default: 0.2).
    #     drop_bool (bool): Boolean value indicating whether to apply dropout in downsample blocks (default: True).
    #     drop_bool2 (bool): Boolean value indicating whether to apply dropout in downsample and upsample blocks (default: False).
    #     filter (int): Filter size (default: 48).
    #     gamma (int): Gamma value for SparseCategoricalFocalLoss (default: 1).
    #     lr (float): Learning rate for the optimizer (default: 0.003).

    # Returns:
    #     tf.keras.Model: U-Net model.
    # Example usage:
    # unet_model = build_unet_model(num_class, weights, act2=keras.layers.LeakyReLU(), act='elu',
    #                               drop=0.3, drop2=0.2, drop_bool=True, drop_bool2=False,
    #                               filter=48, gamma=1, lr=0.003)

    # The function works as follows:
    # 1. Clears the Keras session and performs garbage collection to free up memory.
    # 2. Defines the input layer with shape (128, 128, 3).
    # 3. Constructs the encoder (contracting path) by calling the downsample_block() function multiple times.
    # 4. Constructs the bottleneck layer using double_conv_block().
    # 5. Constructs the decoder (expanding path) by calling the upsample_block() function multiple times.
    # 6. Defines the output layer using Conv2DTranspose.
    # 7. Constructs the U-Net model using the defined layers and connections.
    # 8. Defines the optimizer, loss function, and metrics for the model.
    # 9. Compiles the model.
    # 10. Returns the U-Net model.



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

    unet_model = tf.keras.Model(inputs, outputs, name="U-Net")
    opt = tf.keras.optimizers.RMSprop(learning_rate=lr)
    # true since not using softmax in output layer
    loss = SparseCategoricalFocalLoss(gamma=gamma, class_weight=weights, from_logits=True)  
    met = tf.keras.metrics.MeanIoU(num_classes=num_class, sparse_y_pred=False)
    unet_model.compile(optimizer=opt,
                       loss=loss,
                       metrics=['accuracy', met])

    return unet_model


##Function for mobile net
def mobile_unet_model(output_channels: int, weights = [1,1,1,1,1],
                       trainable = 2, batch_bool = True, lr = 0.004, gamma = 1):
     
    #     Constructs a Mobile UNet model for semantic segmentation tasks.

    #     The Mobile UNet model combines the popular MobileNetV2 architecture with the U-Net architecture,
    #     creating an efficient and effective model for pixel-wise segmentation. The MobileNetV2 serves as
    #     the feature extraction backbone, capturing hierarchical features, while the U-Net-like structure
    #     allows for precise localization of objects.

    #     Args:
    #         output_channels (int): Number of output channels, which corresponds to the number of classes to predict.
    #         weights: Weights for each class to be used during training.
    #         trainable (int): Number of layers to be trained in the MobileNetV2 backbone (default: 2).
    #         batch_bool (bool): Boolean value indicating whether to apply batch normalization (default: True).
    #         lr (float): Learning rate for the optimizer (default: 0.004).
    #         gamma (int): Gamma value for the SparseCategoricalFocalLoss (default: 1).

    #     Returns:
    #         tf.keras.Model: Mobile UNet model for semantic segmentation.
    #     

    # Example usage:
    # model = mobile_unet_model(output_channels, weights, trainable=2, batch_bool=True, lr=0.004, gamma=1)

    # The function works as follows:
    # 1. Clears the Keras session.
    # 2. Defines the input layer.
    # 3. Creates the MobileNetV2 model as the feature extraction backbone.
    # 4. Specifies the layer names from which to extract activations.
    # 5. Retrieves the outputs of the specified layers from the MobileNetV2 model.
    # 6. Constructs the feature extraction model.
    # 7. Sets the trainability of the feature extraction model.
    # 8. Applies the downsampling through the feature extraction model.
    # 9. Initializes the upsample stack.
    # 10. Performs upsampling and establishes skip connections.
    # 11. Defines the last layer of the model.
    # 12. Constructs the Mobile UNet model using the defined layers and connections.
    # 13. Defines the optimizer, loss function, and metrics for the model.
    # 14. Compiles the model.
    # 15. Returns the Mobile UNet model.


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


#Functions for DeepLabNet
def convolution_block(block_input, num_filters=256, kernel_size=3, dilation_rate=1,
        padding="same", use_bias=False, batch_bool = True, dropout = 0.5 ):
    # 
    # Applies a convolution block to the given input.

    # Args:
    #     block_input: Input tensor.
    #     num_filters (int): Number of filters in the convolutional layer (default: 256).
    #     kernel_size (int): Size of the convolutional kernel (default: 3).
    #     dilation_rate (int): Dilation rate for the convolution (default: 1).
    #     padding (str): Padding mode for the convolution (default: 'same').
    #     use_bias (bool): Boolean value indicating whether to include a bias term in the convolutional layer (default: False).
    #     batch_bool (bool): Boolean value indicating whether to apply batch normalization (default: True).
    #     dropout (float): Dropout rate (default: 0.5).

    # Returns:
    #     Tensor: Output tensor after applying the convolution block.
    # 
    # Here's how the function works step by step:
    # 1. Apply a separable convolution operation to the block_input tensor using the SeparableConv2D layer from Keras.
    # 2. If batch_bool is set to True, apply batch normalization using the BatchNormalization layer from Keras.
    # 3. Apply dropout to the normalized tensor using the Dropout layer from Keras.
    # 4. Apply the ReLU activation function to the tensor.

    x = layers.SeparableConv2D( num_filters, kernel_size=kernel_size, dilation_rate=dilation_rate,
        padding=padding, use_bias=use_bias, kernel_initializer=keras.initializers.HeNormal())(block_input)
    if batch_bool:
        x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    return tf.nn.relu(x)

def DilatedSpatialPyramidPooling(dspp_input, drop_rate = 0.5, batch_bool = True):
    # 
    # Applies Dilated Spatial Pyramid Pooling to the given input.

    # Args:
    #     dspp_input: Input tensor.
    #     drop_rate (float): Dropout rate (default: 0.5).
    #     batch_bool (bool): Boolean value indicating whether to apply batch normalization (default: True).

    # Returns:
    #     Tensor: Output tensor after applying Dilated Spatial Pyramid Pooling.
    # Example:
    #     >>> output = DilatedSpatialPyramidPooling(dspp_input, drop_rate=0.5, batch_bool=True)

    #  Here's an overview of how the function works step by step:
    # 1. Obtains the dimensions of the input tensor.
    # 2. Performs average pooling on the input tensor
    # 3. Applies a convolution block to the pooled tensor, using a 1x1 kernel size and the specified dropout and batch normalization settings.
    # 4. Upsamples the output of the convolution block to match the original input tensor's spatial dimensions.
    # 5. Applies separate convolution blocks to the input tensor with different dilation rates (1, 6, 12, and 18), preserving the input tensor's spatial dimensions.
    # 6. Concatenates the upsampled tensor, along with the outputs from the convolution blocks at different dilation rates, along the channel dimension.
    # 7. Applies a final convolution block to the concatenated tensor, using a 1x1 kernel size and the specified dropout and batch normalization settings.
    # 8. Returns the resulting tensor as the output of the Dilated Spatial Pyramid Pooling.



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

def DeeplabV3Plus(image_size, num_classes, weight = [1,1,1,1,1],
                   drop_rate= 0.4, drop_rate2=0.2, batch_bool = False, batch_bool2 = True, lr = 0.005, gamma = 1):
    #    
    #     Constructs a DeepLabV3+ model for semantic segmentation tasks.

    #     DeepLabV3+ is a state-of-the-art model for pixel-wise semantic segmentation. It combines the
    #     powerful feature extraction capabilities of the ResNet50 backbone with dilated convolutions and
    #     spatial pyramid pooling to achieve precise object segmentation. The model utilizes skip connections
    #     to fuse features at different scales, enabling detailed object localization.

    #     Args:
    #         image_size (int): The input image size (both width and height).
    #         num_classes (int): Number of output classes to predict.
    #         weight: Weights for each class to be used during training.
    #         drop_rate (float): Dropout rate for regularization in the model (default: 0.4).
    #         drop_rate2 (float): Additional dropout rate for regularization in the model (default: 0.2).
    #         batch_bool (bool): Boolean value indicating whether to apply batch normalization (default: False).
    #         batch_bool2 (bool): Additional boolean value indicating whether to apply batch normalization (default: True).
    #         lr (float): Learning rate for the optimizer (default: 0.005).
    #         gamma (int): Gamma value for the SparseCategoricalFocalLoss (default: 1).

    #     Returns:
    #         tf.keras.Model: DeepLabV3+ model for semantic segmentation.

    #  Example:
    #     >>> model = DeeplabV3Plus(image_size=256, num_classes=10, weight=class_weights, drop_rate=0.4,
    #                                drop_rate2=0.2, batch_bool=False, batch_bool2=True, lr=0.005, gamma

    # Here's an overview of how the function works step by step:
    # 1. Clears the Keras backend session to ensure a fresh start.
    # 2. Defines the model input as a Keras input tensor with the specified image size and 3 color channels.
    # 3. Loads the ResNet50 model pre-trained on ImageNet, excluding the top classification layer, with the input tensor as the input.
    # 4. Extracts the output of the "conv4_block6_2_relu" layer from the ResNet50 backbone.
    # 5. Applies Dilated Spatial Pyramid Pooling to the extracted feature maps, with the specified dropout rate and batch normalization settings.
    # 6. Upsamples the pooled features to match the spatial dimensions of the original input.
    # 7. Retrieves the output from the "conv2_block3_2_relu" layer of the ResNet50 backbone.
    # 8. Applies a convolution block to the retrieved features, using a smaller number of filters, a 1x1 kernel size, and the specified dropout and batch normalization settings.
    # 9. Concatenates the upsampled and convolved features along the channel dimension.
    # 10. Applies two more convolution blocks to the concatenated features, using the specified dropout and batch normalization settings.
    # 11. Upsamples the resulting features to match the original input's spatial dimensions.
    # 12. Applies a 1x1 convolution layer to produce the final output predictions with the specified number of classes.
    # 13. Creates the DeepLabV3+ model by specifying the model inputs and outputs.
    # 14. Configures the model's optimizer, loss function, and metrics for training.
#     
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
    # 
    # Create a ReduceLROnPlateau callback for learning rate reduction during training.

    # ReduceLROnPlateau is a callback in Keras that reduces the learning rate when a monitored metric
    # has stopped improving. It is commonly used to fine-tune the learning rate during training to
    # improve model performance.
    # our monitored metric is val_mean_io_u

    # Returns:
    #     tf.keras.callbacks.ReduceLROnPlateau: ReduceLROnPlateau callback for learning rate reduction.
    # 
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
    # 
    # Create an EarlyStopping callback for early stopping during training.

    # EarlyStopping is a callback in Keras that stops the training process early when a monitored
    # metric has stopped improving. It is commonly used to prevent overfitting and save training time
    # by stopping the training if the model performance does not improve over a certain number of
    # epochs.
    # Our monitored metric is val_mean_io_u. This will also restore to the model with best 
    # val_mean_io_u value as stopping

    # Returns:
    #     tf.keras.callbacks.EarlyStopping: EarlyStopping callback for early stopping during training.
    # 
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
def plot_accuracy(history):
    # 
    # Plot the accuracy of a model during training.

    # This function plots the accuracy values achieved by a model during training and validation
    # across different epochs. It provides a visual representation of how the accuracy changes over
    # the course of training, allowing for easy interpretation of the model's performance.

    # Args:
    #     history (tf.keras.callbacks.History): The history object obtained from model training.

    # Returns:
    #     None
    # 
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
    # 
    # Plot the mean Intersection over Union (IoU) of a model during training.

    # This function plots the mean Intersection over Union (IoU) values achieved by a model during
    # training and validation across different epochs. The mean IoU is a commonly used metric for
    # evaluating the performance of image segmentation models. This function provides a visual
    # representation of how the mean IoU changes over the course of training.

    # Args:
    #     history (tf.keras.callbacks.History): The history object obtained from model training.

    # Returns:
    #     None
    # 
    plt.plot(history.history['mean_io_u'])
    plt.plot(history.history['val_mean_io_u'])
    plt.title('mean_iou vs. epochs')
    plt.ylabel('mean_iou')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper right')
    plt.show()

def plot_loss(history):
    # 
    # Plot the loss of a model during training.

    # This function plots the loss values achieved by a model during training and validation across
    # different epochs. The loss is a commonly used metric for evaluating the performance of machine
    # learning models. This function provides a visual representation of how the loss changes over
    # the course of training.

    # Args:
    #     history (tf.keras.callbacks.History): The history object obtained from model training.

    # Returns:
    #     None
    # 
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss vs. epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper right')
    plt.show() 

def create_mask(pred_mask):
    # 
    # Create a mask from the predicted mask.
    #
    # This function takes the predicted mask generated by a model and performs post-processing to convert
    # it into a binary mask. The predicted mask is typically a probability distribution over different
    # classes, and this function selects the class with the highest probability as the final mask.
    #
    # Args:
    #     pred_mask (tf.Tensor): The predicted mask generated by a model.
    #
    # Returns:
    #     np.ndarray: The binary mask obtained from the predicted mask.
    # 
    #
    #  How it works:
    # 1. Computes the class with the highest probability by taking the argmax along the last axis.
    # 2. Adds an extra axis to the mask.
    # 3. Converts the mask to a NumPy array.

    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return np.array(pred_mask)
    
def display(display_list, titles=None):
    # 
    # Display a list of images or masks with optional titles.

    # This function takes a list of images or masks and displays them in a grid layout. Each image or mask is
    # shown as a subplot with an optional title. The function is useful for visualizing input images, true masks,
    # and predicted masks generated by different models.

    # Args:
    #     display_list (list): A list of images or masks to be displayed.
    #     titles (list, optional): A list of titles for each image or mask. If not provided, default titles are used.

    # Returns:
    #     None
    # 
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
    # 
    # Display predictions made by a model on input images and corresponding labels.

    # This function takes a trained model (`mod`) and optional input images (`img`) and corresponding labels (`label`).
    # It predicts masks using the model and displays the input image, true mask, and predicted mask for a specified number
    # of samples (`num`). The function also supports providing custom titles for each displayed item.

    # Args:
    #     mod (tf.keras.Model): A trained model used for prediction.
    #     img (ndarray, optional): Input images. If provided, the function will predict masks for these images.
    #     label (ndarray, optional): Corresponding true masks. If provided, the true mask will be displayed alongside
    #                                the predicted mask.
    #     num (int, optional): The number of samples to display predictions for.
    #     titles (list, optional): A list of titles for each displayed item. If not provided, default titles are used.

    # Returns:
    #     None
    # 
    if img is not None:
        temp_images = np.copy(img)
        if "mobile" in titles:
            temp_images = mobilenet_v2.preprocess_input(temp_images)
        elif "deep" in titles:
            temp_images = resnet50.preprocess_input(temp_images)
        else:
            temp_images /= 255.0
    
        for i in range(num):
            pred_mask = mod.predict(temp_images, verbose=0)
            footprint = disk(4)
            mask = np.array(create_mask(pred_mask[i])).reshape(128, 128)
            mask = morphology.closing(mask, footprint)
            display([temp_images[i], label[i], mask], titles)


## Soft Voting
def voting(model_names, t_images, t_labels, offset=10, num=3, numclasses=6):
    # 
    # This function performs soft voting ensemble prediction on a list of
    # pre-trained models. Soft voting is a technique where the predicted
    # probabilities from multiple models are averaged to obtain the final
    # prediction. The function takes a list of model names, test images, and
    # ground truth labels as inputs, and returns the soft voting ensemble
    # predicted masks.

    # Args:
    #     model_names (list): List of model names to load and use for prediction.
    #     t_images (ndarray): Array of test images.
    #     t_labels (ndarray): Array of ground truth labels for the test images.
    #     offset (int): Offset value to start the visualization from (default: 10).
    #     num (int): Number of samples to visualize (default: 3).
    #     numclasses (int): Number of classes in the classification task (default: 6).

    # Returns:
    #     ndarray: Soft voting ensemble predicted masks.

    # The function iterates over the `model_names` list and performs the
    # following steps for each model:
    # 1. Loads the model using Keras `load_model` function.
    # 2. Preprocesses the test images based on the model's requirements. For
    #    example, if the model expects input images to be normalized or
    #    preprocessed in a specific way, the function applies the necessary
    #    transformations.
    # 3. Performs predictions on the preprocessed test images using the loaded
    #    model.
    # 4. Accumulates the predicted probabilities in the `soft` array by adding
    #    them element-wise.

    # After processing all models, the function performs post-processing on the
    # accumulated soft predictions to obtain the final soft voting ensemble
    # predicted masks. It applies a morphological closing operation to smoothen
    # the boundaries of the predicted masks.

    # The function then calculates the mean Intersection over Union (IoU) between
    # the ground truth labels and the soft voting ensemble predicted masks using
    # the MeanIoU metric.

    # Finally, the function prints the mean IoU value and returns the soft voting
    # ensemble predicted masks as an ndarray of shape (num_samples,
    # image_height, image_width).\

    
    soft = np.empty(t_labels.shape + (6,))
    #hard = []
    miou = tf.keras.metrics.MeanIoU(num_classes=numclasses)
    footprint = morphology.disk(radius=4)
    for _model_name in model_names:
        _ = gc.collect()
        print(_model_name)
        keras.backend.clear_session()
        temp_images = np.copy(t_images)
        if "mobile" in _model_name:
            temp_images = mobilenet_v2.preprocess_input(temp_images)
        elif "deep" in _model_name:
            temp_images = resnet50.preprocess_input(temp_images)
        else:
            temp_images /= 255.0
        model = keras.models.load_model(_model_name)
        _ = gc.collect()
        temp1 , temp2 = np.array_split(temp_images,2)
        preds1 = model.predict(temp1, batch_size = 8, verbose=0)
        preds2 = model.predict(temp2, batch_size = 8, verbose=0)
        preds = np.concatenate((preds1, preds2), axis = 0)
     #   mask = create_mask(preds)
        soft = soft + preds
     #   hard.append(mask)
        del model
    keras.backend.clear_session()
    gc.collect()
    s_vote = create_mask(soft)
    for i in range(len(s_vote)):
        if i % 100 == 0:
            print("percent complete: {:.0%}".format((i / len(s_vote))), end="\r")
        x = morphology.closing(s_vote[i].reshape(128, 128), footprint).reshape(128,128,1)
        s_vote[i] = x
    #hard = np.array(hard)
    miou.reset_state()
    miou.update_state(t_labels, s_vote)
    print('s_voting')
    print(miou.result().numpy())
    # for i in range(num):
    #     hard0 = morphology.closing(hard[0, i + offset].reshape(128, 128), footprint)
    #     hard1 = morphology.closing(hard[1, i + offset].reshape(128, 128), footprint)
    #     hard2 = morphology.closing(hard[2, i + offset].reshape(128, 128), footprint)
    #     s_vote1 = np.array(s_vote[i + offset])
    #     display([t_images[i + offset]/255.0, t_labels[i + offset], hard0, hard1, hard2, s_vote1])
    return s_vote


## Saving Georefrence data
def save_geo(masks, georef, foldername):
    # 
    # Saves the predicted masks as GeoTIFF files with there Geo-refrenceing data as well.

    # Args:
    #     masks (ndarray): Array of predicted masks.
    #     georef (list): List of geographic references corresponding to each mask.
    #     foldername (string): What the name of the output folder should be called

    # Returns:
    #     None

    # Overview:
    # 1. Iterates through each mask and its corresponding georeference.
    # 2. Constructs the file name for the GeoTIFF file.
    # 3. Opens a new GeoTIFF file in write mode with the specified parameters.
    # 4. Writes the mask data to the GeoTIFF file.
    # 5. Closes the GeoTIFF file.
    for i in range(len(georef)):
        if i % 100 == 0:
            print("percent complete: {:.0%}".format((i / len(georef))), end="\r")
        file_name = foldername +'/'+ str(i) +'.tif'
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
