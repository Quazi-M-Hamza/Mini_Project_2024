import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = r'C:\Users\HANZA\Desktop\Attendance detection system\Face-Recognition-Attendance-Projects\Training_images'
images = []
classNames = []
# Mapping between file names and display names
name_mapping = {
    'Hamza.jpg': 'Hamza',
    'Raj.jpg': 'Raj',
    'Haider.jpg': 'Haider',
    'Al Pacino.jpg': 'Al Pacino'  # Uncomment if you want to include this entry
}
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    # Use the name mapping or the original file name if not found in the mapping
    classNames.append(name_mapping.get(cl, os.path.splitext(cl)[0]))
print(classNames)

# Check if the classNames and images lists are not empty
if len(classNames) == 0 or len(images) == 0:
    print("Error: No images or classNames found.")
    exit()

# Function to find encodings of faces in images
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Check if face encodings are found
        face_encodings = face_recognition.face_encodings(img)
        if len(face_encodings) > 0:
            encodeList.append(face_encodings[0])
        else:
            print("No face found in one or more images.")
    return encodeList

encodeListKnown = findEncodings(images)
print('Encoding Complete')

# Check if classNames and encodeListKnown have the same length
if len(classNames) != len(encodeListKnown):
    print("Error: Mismatch between number of classNames and encodeListKnown.")
    exit()

# Define the function to mark attendance
def markAttendance(name):
    # Write your code to mark attendance here
    pass  # Placeholder for now

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)
