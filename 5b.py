"""Identification of face using opencv library """


import numpy as np
import cv2

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Read the input image
img = cv2.imread('people.png')

# Convert the image to grayscale (Haar cascades work on grayscale images)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
# Parameters:
#   scaleFactor = 1.1 → specifies how much the image size is reduced at each image scale
#   minNeighbors = 5 → specifies how many neighbors each rectangle should have to retain it
faces = face_cascade.detectMultiScale(gray, 1.1, 5)

# Draw rectangles around detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue rectangle
    roi_gray = gray[y:y + h, x:x + w]    # Region of interest (gray)
    roi_color = img[y:y + h, x:x + w]    # Region of interest (color)

# Display the output image with detected faces
cv2.imshow('Detected Faces', img)

# Wait for a key press to close the window
cv2.waitKey(0)
cv2.destroyAllWindows()

"""Using open cv library of Neural Networks, faces are detected."""