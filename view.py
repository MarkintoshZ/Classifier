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
    frame = cv2.flip(frame, 1)

    image = frame[0:300, 0:300]
    face = Image.fromarray(image)
    face = face.resize((100, 100), Image.ANTIALIAS)
    data = np.asarray(face)
    data = data / 255
    print(round(model.predict(data.reshape(1, 100, 100, 3))[0][0]))

    cv2.rectangle(frame, (0, 0), (300, 300), (0, 0, 255), 2)
    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
