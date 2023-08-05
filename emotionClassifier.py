from keras.models import load_model
import numpy as np
import cv2
from keras.preprocessing import image
import mediapipe as mp
  
# Used to convert protobuf message to a dictionary.
from google.protobuf.json_format import MessageToDict
  
# Initializing the Model
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
    max_num_hands=2)


detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'final_model.h5'
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["happy", "sad"]
HAND = ["none","both","left","right"]
handInd = 0
def emotion_testing():
    cap = cv2.VideoCapture(0)
    while True:
        handInd=0
        ret, test_img = cap.read()
        if not ret:
            continue
        gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

        faces_detected = face_detection.detectMultiScale(gray_img, 1.32, 5)
        img = cv2.flip(test_img, 1)
        for (x, y, w, h) in faces_detected:
            cv2.rectangle(img, (x, y), (x + w, y + h), (102, 204, 0), thickness=6)
            roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
            roi_gray = cv2.resize(roi_gray, (48, 48))
            img_pixels = image.img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255

            predictions = emotion_classifier.predict(img_pixels)

            # find max indexed array
            max_index = np.argmax(predictions[0])
            predicted_emotion = EMOTIONS[max_index]

            cv2.putText(img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 5, 230), 2)

        resized_img = cv2.resize(img, (1000, 700))
        # cv2.imshow('Facial emotion analysis ', resized_img)

        img = resized_img

        # img = cv2.flip(resized_img, 1)
  
    # Convert BGR image to RGB image
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  
    # Process the RGB image
        results = hands.process(imgRGB)
  
    # If hands are present in image(frame)
        if results.multi_hand_landmarks:
  
        # Both Hands are present in image(frame)
            if len(results.multi_handedness) == 2:
                    # Display 'Both Hands' on the image
                handInd = 1
                cv2.putText(img, 'Dancing', (250, 50),
                            cv2.FONT_HERSHEY_COMPLEX,
                            0.9, (0, 255, 0), 2)
            else:
                for i in results.multi_handedness:
                    label = MessageToDict(i)['classification'][0]['label']
                    if label == 'Left':
                        handInd = 2
                        # Display 'Left Hand' on
                        # left side of window
                        cv2.putText(img, 'Dancing',
                                    (20, 50),
                                    cv2.FONT_HERSHEY_COMPLEX, 
                                    0.9, (0, 255, 0), 2)
    
                    if label == 'Right':
                        handInd = 3
                        # Display 'Left Hand'
                        # on left side of window
                        cv2.putText(img, 'Dancing', (460, 50),
                                    cv2.FONT_HERSHEY_COMPLEX,
                                    0.9, (0, 255, 0), 2)
        
        cv2.imshow('Image', img)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
        
        # if cv2.waitKey(0) & 0xFF == ord('0'):
    cap.release()
    cv2.destroyAllWindows
        #     # break
        #     return
    return predicted_emotion,HAND[handInd]


