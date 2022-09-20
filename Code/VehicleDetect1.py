# OpenCV Python program to detect cars in video frame
# import libraries of python OpenCV 
import cv2
import numpy as np
from PIL import Image
  
# capture frames from a video
cap = cv2.VideoCapture('C:/Users/wkeenan14/Documents/CHSTrafficCapstone/Media/video.avi')
  
# Trained XML classifiers describes some features of some object we want to detect
car_cascade = cv2.CascadeClassifier('C:/Users/wkeenan14/Documents/CHSTrafficCapstone/cars.xml')
  
# loop runs if capturing has been initialized.
while True:
    # reads frames from a video
    ret, frames = cap.read()
    #print(len(frames))  
    # convert to gray scale of each frames
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    dilated = cv2.dilate(frames, cv2.dilate(blur,np.ones((3,3))))
  
    # Detects cars of different sizes in the input image
    cars = car_cascade.detectMultiScale(gray, dilated, 1.1, 1)
      
    # To draw a rectangle in each cars
    for (x,y,w,h) in cars:
        cv2.rectangle(frames,(x,y),(x+w,y+h),(0,0,255),2)
  
   # Display frames in a window 
    cv2.imshow('video', frames)
      
# Wait for Esc key to stop
    if cv2.waitKey(33) == 27:
        break
  
# De-allocate any associated memory usage
cv2.destroyAllWindows()