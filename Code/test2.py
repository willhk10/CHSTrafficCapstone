# Import libraries
from PIL import Image
import cv2
import numpy as np
import requests

image = Image.open(requests.get('https://a57.foxnews.com/media.foxbusiness.com/BrightCove/854081161001/201805/2879/931/524/854081161001_5782482890001_5782477388001-vs.jpg', stream=True).raw)
image = image.resize((450,250))
image_arr = np.array(image)
image

grey = cv2.cvtColor(image_arr,cv2.COLOR_BGR2GRAY)
Image.fromarray(grey)
blur = cv2.GaussianBlur(grey,(5,5),0)
Image.fromarray(blur)
dilated = cv2.dilate(blur,np.ones((3,3)))
Image.fromarray(dilated)
