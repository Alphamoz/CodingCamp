import mediapipe as mp
import numpy as np
import cv2

from tensorflow._api.v2.compat.v1 import ConfigProto
from tensorflow._api.v2.compat.v1 import InteractiveSession
from keras.models import load_model
import keras.utils as img_keras
from collections import deque

# Mediapipe initialization and configuration
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, 
circle_radius=0)

cap = cv2.VideoCapture(0)
writer = None

# configuring the tf and model
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
model = load_model('model/_trained.hdf5', compile=False)
Q = deque(maxlen=10)
emotions = ("Angry", "Disgusted", "Feared", "Happy", "Sad", 
"Surprise", "Neutral")


with mp_face_mesh.FaceMesh(min_detection_confidence=0.5,
min_tracking_confidence=0.5) as face_mesh:
    while True:
        _, frame = cap.read()
        # flipping and convert to rgb
        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        # processing the face to get landmarks
        results = face_mesh.process(frame)
        # converting color back
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # checking if there is face detected, then drawing all the landmarks
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                image=frame,
                landmark_list = 0,
                connections=mp_face_mesh.FACEMESH_FACE_OVAL,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)
                
                # initializing roi
                h,w,_ = frame.shape
                xInit,yInit,xEnd,yEnd = w,h,0,0
                for id, landmarkPos in enumerate(face_landmarks.landmark):
                    x,y = int(landmarkPos.x*w),int(landmarkPos.y*h)
                    # getting max and min value of the landmark position, assign to desired var
                    if x < xInit:
                        xInit=x
                    if y < yInit:
                        yInit=y
                    if x > xEnd:
                        xEnd=x
                    if y > yEnd:
                        yEnd=y     
                detected_face = frame[int(yInit):int(yEnd), int(xInit):int(xEnd)]
                detected_face = cv2.cvtColor(detected_face, 
                cv2.COLOR_BGR2GRAY)
                detected_face = cv2.resize(detected_face, (64, 64)) 
                frame_pixels = img_keras.img_to_array(detected_face)
                frame_pixels = np.expand_dims(frame_pixels, axis=0)
                frame_pixels /= 255
                emotion = model.predict(frame_pixels)[0]
                Q.append(emotion)
                results = np.array(Q).mean(axis=0)
                i = np.argmax(results)
                label = emotions[i]
                print(label)
                cv2.putText(frame, label, (xInit, yInit),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(frame, (xInit, yInit), (xEnd, yEnd), (0, 255, 0), 2)
        if writer is None:
            h, w, c = frame.shape
            fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
            writer = cv2.VideoWriter('output.avi', fourcc, 20, 
            (w, h), True)
        writer.write(frame)
        cv2.imshow('frame', frame)
        cv2.imshow('Detected Face', detected_face)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()