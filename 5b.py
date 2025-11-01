"""Identification of face using opencv library """

import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
img = cv2.imread('people.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 5)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2) 
    roi_gray = gray[y:y + h, x:x + w]  
    roi_color = img[y:y + h, x:x + w]    

cv2.imshow('Detected Faces', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""Using open cv library of Neural Networks, faces are detected."""
