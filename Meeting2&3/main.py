import cv2
import time
import DetectHand as hd
start_time = time.time()
lastTime = 0
top_idx = [8,12,16,20]
handDetector = hd.HandDetector(detection_confidence=0.8)
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)
    # feed the frame to the detection hand module
    frame = handDetector.findHand(frame)
    landmarksLoc = handDetector.getHandLocation(frame)
    # print(landmarksLoc)
    # checking if there is landmarks location
    if len(landmarksLoc) != 0:
        # prepare list of fingers 
        fingers = []

        # for right hand thumb
        if landmarksLoc[4][1] < landmarksLoc[20][1]:
            if landmarksLoc[4][1] < landmarksLoc[3][1] :
                fingers.append(1)
            else:
                fingers.append(0)
        # for left hand thumb
        if landmarksLoc[4][1] > landmarksLoc[20][1]:
            if landmarksLoc[4][1] > landmarksLoc[3][1] :
                fingers.append(1)
            else:
                fingers.append(0)
        # for other fingers
        for idx in top_idx:
            if landmarksLoc[idx][2] < landmarksLoc[idx-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        print(fingers)
        # counting the fingers with 1 in the list
        openFingers = fingers.count(1)
        cv2.rectangle(frame, (20,20),(200,200),(255,255,255),
        cv2.FILLED)
        cv2.putText(frame, str(int(openFingers)), (50,170),
        cv2.FONT_HERSHEY_PLAIN, 10, (255,0,0), 25)
    
    # getting fps and shows it
    currtime = time.time()
    fps = 1/(currtime-lastTime)
    lastTime = currtime
    cv2.putText(frame, str(int(fps)), (500,70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)
    
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()