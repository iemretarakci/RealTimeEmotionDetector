from deepface import DeepFace
import cv2 as cv
import matplotlib.pyplot as plt
import time

model = DeepFace.build_model("Emotion")
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap= cv.VideoCapture(0)
while(1):
    #Get a frame from cam
    ret, frame = cap.read()
    if not ret:
        break
    #Face Detector
    faces=face_cascade.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=5,minSize=(30,30))
    for (x,y,w,h)in faces:
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),1)
    #Emotion Analyzer    
    if w > 30 and h > 30:  # Minimum kabul edilebilir yüz boyutu
        face_roi = frame[y:y+h, x:x+w]
        emotion_result = DeepFace.analyze(face_roi, ["emotion"], enforce_detection=False)
    #Getting result for correct emotion
    dominant_emotion=max(emotion_result[0]["emotion"],key=emotion_result[0]["emotion"].get)
    #Print result
    cv.putText(frame,f"emotion:{dominant_emotion}",(x,y-10),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
    #Show screen
    cv.imshow("EmotionDetection",frame)
    k=cv.waitKey(5) & 0xFF
    if k==27:
        break
cv.destroyAllWindows()    