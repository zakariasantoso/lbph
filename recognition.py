import time
from datetime import datetime
import cv2
import numpy as np
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.read('trainer/trainer.yml')
cascadePath = "cascade.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX

# iniciate id counter
id = 0
# names related to employee.csv but skip the first row
employee_file = open('employee.csv', 'r')
employee_data = employee_file.readlines()
names = []
print('Loading Face Database...')
# loop over employee data and append to names
for i in employee_data:
    # skip first row
    if i == employee_data[0]:
        continue
    # split the data
    i = i.split(';')
    # append list id = i[0] and name = i[1]
    names.append({'id': i[0], 'name': i[1].strip('\n')})
    print('ID: ' + i[0] + ' Nama: ' + i[1].strip('\n'))

cam = cv2.VideoCapture(0)
print('Memulai Kamera...')
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height

# define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(3)

print('Memulai pendeteksian wajah...')
while True:
    ret, img =cam.read()

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        # scaleFactor = 1.2,
        minNeighbors = 5,
        # minSize = (int(minW), int(minH)),
       )

    for(x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
                                                        
        # Cek nilai kecocokan histogram
        if (confidence < 70):
            print('Face detected')
            name = "unknown"
            for i in names:
                if int(i['id']) == int(id):
                    name = i['name']
                    print('Face Recognized! Name: ' + name)
                    # make rectangle around face and write name
                    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
                    cv2.putText(img, name, (x+5,y-5), font, 1, (255,255,255), 2)
                    # write the frame
                    cv2.imwrite("captured/User." + str(id) + '.' + str(datetime.now().strftime("%Y-%m-%d.%H.%M.%S")) + ".jpg", img)
                else:
                    # print('Face not recognized')
                    continue
            confidence = "  {0}%".format(100 - round(100 - confidence))
        else:
            # print("not confidence")
            name = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(img, str(name), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)

    # resize
    # img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    # set min width and height

    cv2.imshow('camera',img)

    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        print('Exiting...')
        break
    # release camera
cam.release()
cv2.destroyAllWindows()
