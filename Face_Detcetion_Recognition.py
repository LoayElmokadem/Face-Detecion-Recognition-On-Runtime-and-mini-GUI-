'''
note that this code train and test on runtime 
meaning that everything is dynamically means you can
train the model on your Choice by collecting a photo of people of your choice in a diffrent folders 
like i did in my example and also to choose a photo as you like to proccess it
'''
import os
import cv2 as cv
import numpy as np
import tkinter as tk
from tkinter.filedialog import askopenfilename,askdirectory
names =[]
cascade_model = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
def get_folder_names():
    print("please Choose the folder of folders image from the popped window")
    root = tk.Tk()
    root.withdraw() 

    folder_path = askdirectory()
    if folder_path:
        for folder in os.listdir(folder_path):
            if os.path.isdir(os.path.join(folder_path, folder)):
                names.append(folder)
    return folder_path

def train_face():
    DIR = get_folder_names()
    features =[]
    tags = []
    for folder_name in os.listdir(DIR):
        if not os.path.isdir(os.path.join(DIR, folder_name)):
            continue

    names = [person for person in os.listdir(DIR) if os.path.isdir(os.path.join(DIR, person))]
    for person in names:
        path = os.path.join(DIR, person)
        tag = names.index(person)
        if not os.path.isdir(path):
            print(f"error couldnot find'{person}' pics folder this will be skipped ")
            continue

        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            photo = cv.imread(img_path)
            if photo is None:
                continue
            grayed_image = cv.cvtColor(photo, cv.COLOR_BGR2GRAY)
            faces = cascade_model.detectMultiScale(grayed_image, scaleFactor=1.2, minNeighbors=5)
            for (x, y, w, h) in faces:
                faces_roi = grayed_image[y:y+h, x:x+w]
                features.append(faces_roi)
                tags.append(tag)

    features, tags = np.array(features, dtype='object'), np.array(tags)
    trained_model = cv.face.LBPHFaceRecognizer_create()
    trained_model.train(features, tags)
    print("finished training the Model")
    return trained_model

def Face_detection_recognitntion(photo, model=None):
    grayed_image = cv.cvtColor(photo, cv.COLOR_BGR2GRAY)
    faces = cascade_model.detectMultiScale(grayed_image, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        if model is not None:
            tag, confidence = model.predict(grayed_image[y:y+h, x:x+w])
            cv.putText(photo, str(names[tag]), (x, y - 10), cv.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), thickness=2)
            print(f"Predicted Face in Image is {str(names[tag])}")

        cv.rectangle(photo, (x, y), (x + w, y + h), (0, 255, 0), 2)
    display_img(photo)

def display_img(photo):
    cv.imshow("image", photo)
    cv.waitKey(0)
    cv.destroyAllWindows()

def get_img():
    print("please Choose an image from the popped window")
    root = tk.Tk()
    root.withdraw()
    img_path = askopenfilename(title="Select an image to be processed", filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")])
    root.destroy()
    if img_path:
        photo = cv.imread(img_path)
    else:
        print("please try again")
        return get_img()
    return photo

train_flag, trained_model = False, None

while True:
    choice = int(input("Face Detection (1) or Face Recognition (2) or Exit (-1)\n"))
    if choice == -1:
        print("thanks for using our project\n")
        break

    photo = get_img()
    if choice == 1:
        if photo is not None:
            Face_detection_recognitntion(photo)
        else:
            print("Try Again with a different image")
    elif choice == 2:
        if not train_flag:
            print("Please wait for the model to train")
            trained_model = train_face()
            train_flag = True
        Face_detection_recognitntion(photo, model=trained_model)
    else:
        print("Invalid Choice")
