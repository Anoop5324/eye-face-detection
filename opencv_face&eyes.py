import cv2

face_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyes_data = cv2.CascadeClassifier("haarcascade_eye.xml")

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
