import os
import dlib
import face_recognition
import numpy as np
import cv2

#import datetime to get the time of attendance
from datetime import datetime
folder = (r'C:\Users\user\Desktop\RussellProject\Images') #create a folder containing images of the people whose attendance you want to take

images = []
classNames = []
myList = os.listdir(folder)
print(myList)  # prints images in folder
for cls in myList:
    current_image = cv2.imread(f'{folder}/{cls}')
    images.append(current_image)
    classNames.append(os.path.splitext(cls)[0])  # revoming .jpg
print(classNames)


def find_encodings(images):
    encode_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)
        print('Encoding Complete')
    return encode_list


encodelist = find_encodings(images)


def mark_attendance(name):
    # you can create a new csv file
    with open(r'C:\Users\user\Desktop\RussellProject\attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        print(myDataList)
        newlist = []
        for line in myDataList:
            entry = line.split(',')
            newlist.append(entry[0])
            if name not in newlist:
                now = datetime.now()
                dfString = now.strftime('%H:%M:%S')
                f.writelines(f'\n{name} {dfString}')


# initializing videocam
cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    # reducing size
    imgSmall = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)
    face_location_current_frame = face_recognition.face_locations(imgSmall)
    encode_current_frame = face_recognition.face_encodings(
        img, face_location_current_frame)

    for encode_face, face_loc in zip(encode_current_frame, face_location_current_frame):
        matches = face_recognition.compare_faces(encodelist, encode_face)
        face_distance = face_recognition.face_distance(encodelist, encode_face)

        matchIndex = np.argmin(face_distance)

    # printing match name
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = face_loc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y1-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y1-6),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (225, 225, 225), 2)
    cv2.imshow('Webcam', img)
    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
