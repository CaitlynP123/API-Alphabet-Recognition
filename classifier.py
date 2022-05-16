import cv2 
import numpy as np 
import pandas as pd 

from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score 

from PIL import Image 
import PIL.ImageOps 
import os, ssl, time

x = np.load('./image.npz')['arr_0']
y = pd.read_csv("./labels.csv")["labels"]

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J','K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T','U', 'V', 'W', 'X', 'Y', 'Z']
nclasses = len(classes)

xTrain, xTest, yTrain, yTest = train_test_split(x, y, train_size=3500, test_size=500, random_state=9) 

xTrain_scaled = xTrain/255.0 
xTest_scaled = xTest/255.0

clf = LogisticRegression(solver='saga', multi_class='multinomial').fit(xTrain_scaled, yTrain)

yPred = clf.predict(xTest_scaled) 
accuracy = accuracy_score(yTest, yPred) 

def alphabet_detection(image):
    img_pil = Image.open(image)
    img_bw = img_pil.convert('L')

    img_bw_resized = img_bw.resize((22, 30))
    pixel_filter = 20
    min_pixel = np.percentile(img_bw_resized, pixel_filter)
    img_bw_resized_inverted_scaled = np.clip(img_bw_resized-min_pixel, 0, 255)
    max_pixel = np.max(img_bw_resized)
    img_bw_resized_inverted_scaled = np.asarray(img_bw_resized_inverted_scaled)/max_pixel

    test_sample = np.array(img_bw_resized_inverted_scaled).reshape(1, 660)
    test_pred = clf.predict(test_sample)

    return test_pred[0]