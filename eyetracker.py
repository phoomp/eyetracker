import numpy as np
import cv2

# Should be included with your opencv installation
face_cascade_path = 'eyetracker/lib/python3.9/site-packages/cv2/data/haarcascade_frontalface_default.xml'
eye_cascade_path = 'eyetracker/lib/python3.9/site-packages/cv2/data/haarcascade_eye.xml'

vcap = cv2.VideoCapture(2)

width = vcap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)

face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

print(f'width: {width}')
print(f'height: {height}')


# def filter_extra_eyes(eyes):
#     '''
#     Most "extra" detected eyes are nostrils, due to its round shape with a black hole in the middle. Therefore, we are going to eliminate this problem with a simple logic.
#     '''
#     eyesize_avg = []
#     for (_, _, ew, eh) in eyes:
#         eyesize_avg.append((ew + eh) / 2)

#     biggest_value = np.max(eyesize_avg)
#     biggest_value_index = np.where(eyesize_avg == biggest_value)

#     eyesize_avg = np.delete(eyesize_avg, biggest_value_index)

#     second_biggest_value = np.max(eyesize_avg)
#     second_biggest_value_index = np.where(eyesize_avg == second_biggest_value)

#     valid_eye_index = [biggest_value_index, second_biggest_value_index]
#     return valid_eye_index

while True:
    ret, frame = vcap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y: y + h, x: x + w]
        roi_color = frame[y: y + h, x: x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        # if len(eyes) > 2:
        #     valid_eye_index = filter_extra_eyes(eyes)
        #     for index in valid_eye_index:
        #         eyes = np.delete(eyes, index)


        if len(eyes) == 2:
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    fps =  vcap.get(cv2.CAP_PROP_FPS)
    print(f'fps: {fps}')
    

