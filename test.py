import cv2
import numpy as np
import math
import tensorflow as tf
from cvzone.ClassificationModule import Classifier
from cvzone.HandTrackingModule import HandDetector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
offset = 20
imgSize = 300
# labels = ["fist", "ind_thumb", "index", "palm", "pinky", "thumb", "victory", "Yo"]
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3),np.uint8)
        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]
        
        if h>w:
            k = imgSize/h
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            WidthGap = math.ceil((imgSize-wCal)/2)
            imgWhite[:, WidthGap:wCal+WidthGap] = imgResize
            pridiction, index = classifier.getPrediction(imgWhite)
            print(pridiction, index)
        if w>h:
            k = imgSize/w
            hCal = math.ceil(k*h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            heightGap = math.ceil((imgSize-hCal)/2)
            imgWhite[heightGap:hCal+heightGap, :] = imgResize
            pridiction, index = classifier.getPrediction(imgWhite)
            print(pridiction, index)
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImgWhite", imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    