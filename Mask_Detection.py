#Importing OpenCV 
import datetime
from keras.models import load_model
import cv2 
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
mymodel=load_model('mymodel.h5')
#Importing HARR CASCADE XML file
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Capture Video from web cam hence (0) or else add your own media file
cap = cv2.VideoCapture(0)

#Creating a loop to capture each frame of the video in the name of Img
while True:
    _,img = cap.read()

    #Converting to grey scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Allowing multiple face detection
    faces = face_cascade.detectMultiScale(gray, 1.1, 6)

    #Creating Rectangle around face
    for(x, y, w, h) in faces:
        face_img = img[y:y+h, x:x+w]
        input_image_resized = cv2.resize(face_img, (128,128))

        input_image_scaled = input_image_resized/255

        input_image_reshaped = np.reshape(input_image_scaled, [1,128,128,3])

        input_prediction = mymodel.predict(input_image_reshaped)
        #cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 250), 2)
        input_pred_label = np.argmax(input_prediction)

        if input_pred_label==1:
            #print(input_prediction)
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
            cv2.putText(img,'NO MASK',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
            datet=str(datetime.datetime.now())
            cv2.putText(img,datet,(400,450),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        else:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
            cv2.putText(img,'MASK',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
            datet=str(datetime.datetime.now())
            cv2.putText(img,datet,(400,450),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        
    #Displaying the image 
    cv2.imshow('Detected Mask ',  img)

    #Waiting for escape key for image to close adding the break statement to end the face detection screen
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


#Real-time releasing the captired frames
cap.release()
cv2.destroyAllWindows()

