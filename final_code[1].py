# -*- coding: utf-8 -*-
"""Final code.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18gMJoJG9SXsE-kQPEDg8zG4sjBiYYAzE
"""

import os
import dlib
import face_recognition
import numpy as np
import cv2


# use cv2.imshow is you are not using colab


"""



def count_faces_in_picture(path2):#path2 is the directory of the picture
  path2 = cv2.imread(path2)
   # Detect the coordinates
  detector = dlib.get_frontal_face_detector()
  gray = cv2.cvtColor(path2, cv2.COLOR_BGR2GRAY)
  faces = detector(gray)
  # Iterator to count faces
  i = 0
  for face in faces:
      # Get the coordinates of faces
    x, y = face.left(), face.top()
    x1, y1 = face.right(), face.bottom()
    cv2.rectangle(path2, (x, y), (x1, y1), (0, 255, 0), 2)

    # Increment iterator for each face in faces
    i = i+1

  # Display the box and faces
    cv2.putText(path2, 'face num'+str(i), (x-10, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    print(face, i)
  if i == 1:
    print('There is one face in the image')
  else:
    print(f'There are {i} faces in the image')
        # Display the resulting frame
  cv2_imshow(path2)
  """


detector = dlib.get_frontal_face_detector()
cap = cv2.VideoCapture(0)
# Capture frames continuously
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    # converting imge to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    # Iterator to count faces
    i = 0
    for face in faces:
        # Get the coordinates of faces
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

        # Increment iterator for each face in faces
        i = i+1

        # Display the box and faces
        cv2.putText(frame, 'face num'+str(i), (x-10, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        print(face, i)

        # Display the resulting frame
    cv2.imshow('frame', frame)
    print(f'There are {i} faces in the image')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Release the capture and destroy the windows
cap.release()
cv2.destroyAllWindows()
