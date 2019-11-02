import cv2
from PIL import Image
import time
import numpy as np
from keras.models import load_model


model = load_model('cnn.h5')


# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    face = Image.fromarray(frame).convert('L')
    face = face.resize((85, 50), Image.ANTIALIAS)
    data = np.asarray(face)
    data = data / 255
    print(model.predict(data.reshape(1, 50, 85, 1)))

    # Display the resulting image
    # cv2.imshow('Video', face)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
