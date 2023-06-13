import cv2
import mediapipe as mp
class HandDetector:
    def __init__(self, static_mode=False, maxhands=2, detection_confidence= 0.5, tracking_confidence=0.5):
        # properties initialization
        self.static_mode = static_mode
        self.maxhands = maxhands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands(self.static_mode, self.maxhands, self.detection_confidence, self.tracking_confidence)
        self.mpdraw = mp.solutions.drawing_utils
    
    def findHand(self, frame, draw_landmark=True):
        # making rgb first
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # process the image to get the hand result
        self.results = self.hands.process(img)
        # if hands detected
        if self.results.multi_hand_landmarks:
            # going to draw the landmarks and its connection
            for landmarks in self.results.multi_hand_landmarks:
                if draw_landmark:
                    self.mpdraw.draw_landmarks(frame, landmarks, self.mphands.HAND_CONNECTIONS)
        return frame
    
    def getHandLocation(self, frame, handNo=0, draw=True):
        landmarksLoc = []
        # if hands detected
        if self.results.multi_hand_landmarks:
            # getting the hand for the last hand shown
            myHand = self.results.multi_hand_landmarks[handNo]
            # for landmarks in the hand it will be numerated 0 till 20
            for idx,lm in enumerate(myHand.landmark):
                # getting the width and height
                h,w,_ = frame.shape
                # calculating center of landmarks by multiplying with w and h
                cx,cy = int(lm.x*w), int(lm.y*h)
                # appending to the list landmarkloc
                landmarksLoc.append([idx,cx,cy])
                if draw:
                    cv2.circle(frame, (cx,cy), 5, (255,0,255), 
                    cv2.FILLED)
        return landmarksLoc
    
    

    