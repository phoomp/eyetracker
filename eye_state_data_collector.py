import numpy as np
from matplotlib import pyplot as plt
import cv2

import os
from pathlib import Path

import uuid

# Gathered Picture Destination. Make sure to add a slash at the end:
save_open_destination = 'eyestate/open/'
save_close_destination = 'eyestate/close/'

# Alternate mode: if you are having problems with detecting closed eyes, enable this mode. WARNING: This mode 
# does not take position of eye into account. Adjust the values and position your (closed) eyes inside the boxes.

alternate_mode = False

# Set what state you are training
# training_state = 'open'
training_state = 'open'

# Should be included with your opencv installation
face_cascade_path = 'eyetracker/lib/python3.9/site-packages/cv2/data/haarcascade_frontalface_default.xml'
eye_cascade_path = 'eyetracker/lib/python3.9/site-packages/cv2/data/haarcascade_eye.xml'


if not os.path.exists(save_open_destination):
    os.makedirs(save_open_destination)

if not os.path.exists(save_close_destination):
    os.makedirs(save_close_destination)

vcap = cv2.VideoCapture(2)

face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

while True:
    ret, frame = vcap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if not alternate_mode:
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y: y + h, x: x + w]
            roi_color = frame[y: y + h, x: x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)

            if len(eyes) == 2:
                for (ex, ey, ew, eh) in eyes:
                    roi_color = cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

                    print(roi_color.shape)

                    img = roi_gray[ey:ey+eh, ex:ex+eh]
                    
                    filename = str(uuid.uuid4()) + '.png'
                    
                    if training_state == 'open':
                        filename = save_open_destination + filename
                    else:
                        filename = save_close_destination + filename

                    cv2.imwrite(filename, img)
    else:
        frame = cv2.rectangle(frame, (460, 300), (540, 380), (255, 0, 0), 10)
        img = gray[300:380, 460:540]
        cv2.imshow('img', img)

        filename = str(uuid.uuid4()) + '.png'
                    
        if training_state == 'open':
            filename = save_open_destination + filename
        else:
            filename = save_close_destination + filename

        cv2.imwrite(filename, img)
                
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # fps =  vcap.get(cv2.CAP_PROP_FPS)
    # print(f'fps: {fps}')
    

