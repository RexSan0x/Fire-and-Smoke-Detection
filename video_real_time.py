import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import keras.utils as image

#arduino part
import pyfirmata
import time

board = pyfirmata.Arduino('COM4')
'''
while True:
    board.digital[13].write(1)
    time.sleep(1)
    board.digital[13].write(0)
    time.sleep(10)
'''

#Load the saved model
model = tf.keras.models.load_model('D:/Downloads/fire_detection.h5')
video = cv2.VideoCapture(0)
PIEZO_PIN = board.get_pin('d:11:p')
while True:
        _, frame = video.read()
#Convert the captured frame into RGB
        
        im = Image.fromarray(frame, 'RGB')
#Resizing into 224x224 because we trained the model with this image size.
        im = im.resize((224,224))
        img_array = image.img_to_array(im)
        img_array = np.expand_dims(img_array, axis=0) / 255
        probabilities = model.predict(img_array)[0]
        #Calling the predict method on model to predict 'fire' on the image
        prediction = np.argmax(probabilities)
        #if prediction is 0, which means there is fire in the frame.
        if prediction == 0:
                cv2.putText(frame, "Fire and smoke Detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                print(probabilities[prediction])
                board.digital[4].write(0)
                board.digital[5].write(1)

                PIEZO_PIN.write(0.8)
                #board.digital[11].write(1)
        else:
            board.digital[5].write(0)
            board.digital[4].write(1)
            PIEZO_PIN.write(0)
            #board.digital[11].write(0)

        cv2.imshow("Capturing", frame)
        key=cv2.waitKey(1)
        if key == ord('q'):
                break
video.release()
cv2.destroyAllWindows()
