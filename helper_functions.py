import numpy as np
import tifffile as tiff
import os
import glob


# This function will return the images and labels within the given folder
# labels are derived from the first pixel in the label img
# images are maximum value normalized so they are from 0-1
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
    print("reading ", numfiles, " images")
    for i in range(numfiles):
        # works like a percent bar to make sure function did not hang
        if i % 100 == 0:
            print("percent complete: {:.0%}".format((i / numfiles)), end="\r")

        im_temp = tiff.imread(images_path[i])  # tifffile library command
        lbl_temp = tiff.imread(labels_path[i])
        im_norm = im_temp / im_temp.max() # max normalizing the image
        img.append(im_norm)
        label.append(lbl_temp[0, 0]) # label is made with the (0,0) pixel of label image
    img = np.asarray(img)
    label = np.asarray(label)
    return img, label