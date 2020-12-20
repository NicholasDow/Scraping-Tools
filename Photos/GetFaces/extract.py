import cv2
import sys

imagePath = sys.argv[1]

image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faceCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
faces = faceCascade.detectMultiScale(
    gray, scaleFactor=1.5, minNeighbors=1, minSize=(30, 30)
)


for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    roi_color = image[y : y + h, x : x + w]
    roi_color = cv2.resize(roi_color, (256, 256))
    cv2.imwrite(str(w) + str(h) + "_faces.jpg", roi_color)

status = cv2.imwrite("faces_detected.jpg", image)
print("written to filesystem: ", status)