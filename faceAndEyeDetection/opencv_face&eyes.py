from concurrent.futures import __dir__

import cv2
from pathlib import Path

faceDataPath = Path(__file__).parent.parent/"data/haarcascade_frontalface_default.xml"
eyeDataPath = Path(__file__).parent.parent/"data/haarcascade_eye.xml"

# with faceDataPath.open() as f:
print(faceDataPath, eyeDataPath)
face_data = cv2.CascadeClassifier(faceDataPath.open().name)
eyes_data = cv2.CascadeClassifier(eyeDataPath.open().name)

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = eyes_data.detectMultiScale(gray)
    faces = face_data.detectMultiScale(gray, 1.35,5)

    for x,y,w,h in eyes:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,20,10), 5)

    for x,y,w,h in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,20,10), 5)

    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k== 27:
        break

cap.release()
cv2.destroyAllWindows()