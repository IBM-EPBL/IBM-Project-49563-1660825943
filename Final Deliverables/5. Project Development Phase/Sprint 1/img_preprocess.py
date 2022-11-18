# Importing the Necessary Libraries
from sklearn.preprocessing import LabelEncoder
from skimage import feature 
from imutils import paths
import numpy as np
import cv2
import os

def quantify_image(img):

    '''
    Function to compute the histogram of oriented gradients feature vector of the given input img
    
    Parameters: img - Image file

    Output: Returns feature vector
    '''

    features = feature.hog(
                            image=img,
                            orientations=9,
                            pixels_per_cell=(10,10),
                            cells_per_block=(2,2),
                            transform_sqrt=True,
                            block_norm="L1")

    return features

def load_split(path):
    
    '''

    Takes the list of images from the input directory and then initialize the the images with the labels (i.e) Labelling Image.
    Here before labelling the image is preprocessed with the following steps: Grayscaling, Resizing, Thresholding and finally the image after is thresholded it is quantified to obtain the histogram of gradient features

    Parameters: path - Image Directory Source path

    Returns:
    A tuple containing two numpy arrays:
    data - an array of hog features computed on images
    labels - an array of image labels

    '''

    imagePaths = list(paths.list_images(path))
    data = []
    labels = []
    for imagePath in imagePaths:
        label = imagePath.split(os.path.sep)[-2]
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        image = cv2.resize(image, (200, 200))
        image = cv2.threshold (image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU) [1]
        features = quantify_image(image)
        data.append(features)
        labels.append(label)
        
    return (np.array(data), np.array(labels))

def load_data():
    # Loading Training and Testing data
    trainingPath = r"dataset\training"
    testingPath = r"dataset\testing"
    # loading the training and testing data 
    print("[INFO] loading data...")
    (X_train, y_train) = load_split(trainingPath) 
    (X_test, y_test) = load_split(testingPath)

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test) 
    print(X_train.shape, y_train.shape)
    return (X_train, y_train), (X_test, y_test)
