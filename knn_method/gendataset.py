import cv2
import os
import time

cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

face_detector = cv2.CascadeClassifier('train_cascade.xml')
cv2.CascadeClassifier('train_cascade.xml')

face_id = input('\n enter user name end press <enter> ==>  ')

print("\n [INFO] Initializing face capture. Look the camera and wait ...")
# initialize individual sampling face count
count = 0
multiple = 5

while(True):

    ret, img = cam.read()
    # img = cv2.flip(img, 1) # flip video image vertically
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(img, 1.3, 5)

    for(x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        # count += 1
        
        # save the captured image into the datasets folder
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        count += 1

        if count >= multiple:
            print("\n [INFO] Try another anggle, brightness or expression to increase the acuracy. And if important try use a glasses so i can detect you while using a glasses.")
            time.sleep(5)
            multiple = multiple + 5
            print("\n [INFO] Starting face detection again...")

        # check if folder <face_id> inside the train_dir exists
        if not os.path.exists('train_dir/' + str(face_id)):
            os.makedirs('train_dir/' + str(face_id))
            print("\n [INFO] Creating directory: " + str(face_id))
        # save the 300x300 captured image into the train_dir/<face_id> folder
        cv2.imwrite("train_dir/" + str(face_id) + "/" + str(face_id) + "_" + str(count) + ".jpg", img[y:y+h,x:x+w])
        cv2.imshow('image', img)
    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 30:
         break

    # Do a bit of cleanup
print("\n [INFO] Dataset generated!")
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()