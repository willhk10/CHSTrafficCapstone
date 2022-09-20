import cv2 as cv

# Read image from your local file system
original_image = cv.imread('C:/Users/wkeenan14/Documents/CHSTrafficCapstone/abba.png')

# Convert color image to grayscale for Viola-Jones
grayscale_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)

face_cascade = cv.CascadeClassifier("C:/Users/wkeenan14/Documents/CHSTrafficCapstone/haarcascade_frontalface_default")
detected_faces = face_cascade.detectMultiScale(grayscale_image)
for (column, row, width, height) in detected_faces:
    cv.rectangle(
        original_image,
        (column, row),
        (column + width, row + height),
        (0, 255, 0),
        2
    )
cv.imshow('download.jpg', original_image)
cv.waitKey(0)
cv.destroyAllWindows()