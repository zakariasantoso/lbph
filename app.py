import tkinter
import os
import cv2
import time

path = 'dataset'

main_window = tkinter.Tk()
main_window.title("Face Recognizer")

# Create a label
headerLabel = tkinter.Label(main_window, text="Local Binary Pattern Histogram Face Recognizer")
headerLabel.pack()


def trainDataset():
    headerLabel.config(text="Training...")
    os.system("python training.py")
    headerLabel.config(text="Training Complete")

def getFaceId():
    headerLabel.config(text="Mohon masukkan face id: ")
    canvas = tkinter.Canvas(main_window, width = 300, height = 300)
    canvas.pack()
    face_id = tkinter.Entry(main_window)
    canvas.create_window(150, 150, window=face_id)
    button = tkinter.Button(text='Get the face id', command=face_id.get())
    canvas.create_window(150, 180, window=button)

def generateDataset():
    headerLabel.config(text="Generating Dataset...")
    cam = cv2.VideoCapture(0)
    cam.set(3, 640) # set video width
    cam.set(4, 480) # set video height
    face_detector = cv2.CascadeClassifier('train_cascade.xml')
    cv2.CascadeClassifier('train_cascade.xml')
    headerLabel.config(text="Membuat dataset untuk face id: " + getFaceId())
    headerLabel.config(text="Memulai face detection...")
    count = 0
    multiple = 5

    while(True):
        ret, img = cam.read()
        # img = cv2.flip(img, 1) # flip video image vertically
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for(x,y,w,h) in faces:

            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
            # count += 1
            
            # save the captured image into the datasets folder
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
            count += 1

            if count >= multiple:
                headerLabel.config(text="Try another anggle, brightness or expression to increase the acuracy. And if important try use a glasses so i can detect you while using a glasses.")
                time.sleep(1)
                multiple = multiple + 5
                headerLabel.config(text="Starting face detection again...")

            # save the captured image into the datasets folder
            cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
            cv2.imshow('image', img)
        k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break
        elif count >= 50:
            break

    # Do a bit of cleanup
    headerLabel.config(text="Dataset berhasil di generate")
    headerLabel.config(text="Menutup kamera...")
    cam.release()
    cv2.destroyAllWindows()









trainButton = tkinter.Button(main_window, text="Train", command=trainDataset)
trainButton.pack()

generateDatasetButton = tkinter.Button(main_window, text="Generate Dataset", command=getFaceId)
generateDatasetButton.pack()
main_window.mainloop()

