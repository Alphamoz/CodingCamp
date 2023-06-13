import cv2
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
red =  (0, 0, 255)
green = (0, 255, 0)
while True:
    ret, frame = cap.read()
    # grey image
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect face, returning bounding box pos
    faces = face_cascade.detectMultiScale(gray_image)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (w+x, y+h), red, 3)
        cv2.putText(frame,'ITS ME!!!', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, green, 2)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

# 