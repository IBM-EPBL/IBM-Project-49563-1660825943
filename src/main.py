from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from skimage import feature 
from imutils import build_montages
from imutils import paths
import numpy as np
import cv2
import os
import pickle
from img_preprocess import load_data, quantify_image

def test(model):
    testingPath = r"dataset\testing"
    # randomly select a few images and then initialize the output images # for the montage
    testingPaths = list(paths.list_images(testingPath))
    idxs = np.arange(0, len (testingPaths))
    idxs = np.random.choice (idxs, size=(25,), replace=False)
    images = []

    # loop over the testing samples 
    for i in idxs:
    # load the testing image, clone it, and resize it
        image = cv2.imread(testingPaths[i])
        output = image.copy()
        output = cv2.resize(output, (128, 128))
        # pre-process the image in the same manner we did earlier
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (200, 200))
        image = cv2.threshold (image, 0, 255,
        cv2.THRESH_BINARY_INV | cv2. THRESH_OTSU)[1]

    # quantify the image and make predictions based on the extracted features using the last trained Random Forest 
        features = quantify_image(image)
        preds = model.predict([features])
        le = LabelEncoder()
        if preds[0] == 0:
            label = "healthy"
        else:
            label = "parkinson"
    # draw the colored class label on the output image and add it to # the set of output images
        color = (0, 255, 0) if label == "healthy" else (0, 0, 255)
        cv2.putText(output, label, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        images.append(output)
    print(len(images))
    # create a montage using 128x128 "tiles" with 5 rows and 5 columns
    montage = build_montages (images, (128, 128), (5, 5))[0]
    # show the output montage
    cv2.imshow("Output", montage) 
    cv2.waitKey(0)
    


if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = load_data()
    print("[INFO] training model")
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    test(model)
    print("[INFO] Model Testing Completed")
    # make predictions on the testing data
    predictions = model.predict(X_test) # compute the confusion matrix and and use it to derive the raw
    # accuracy
    cm = confusion_matrix(y_test, predictions).flatten() 
    print(cm)
    (tn, fp, fn, tp) = cm
    accuracy  = (tp + tn) / float(cm.sum())
    print(accuracy)
    pickle.dump(model, open('./artifacts/parkinson.pkl', 'wb'))
    print("[INFO] Model Saving Completed")
