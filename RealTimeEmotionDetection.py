from deepface import DeepFace
import cv2 as cv
import matplotlib.pyplot as plt
import time
import os

model = DeepFace.build_model("Emotion")
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

    
#Frame Processing Block
def frame_processor(target_frame):
    cap= cv.VideoCapture(0)
    current_frame=0
    while True:
        #Get a frame from cam
        ret, frame = cap.read()
        if not ret:
            break
        #Face Detector
        faces=face_cascade.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=5,minSize=(30,30))
        for (x,y,w,h)in faces:
            cv.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),1)
            #Emotion Analyzer
        if w > 30 and h > 30:  # Minimum acceptable face size
            face_roi = frame[y:y+h, x:x+w]
            emotion_result = DeepFace.analyze(face_roi, ["emotion"], enforce_detection=False)
            #Getting result for correct emotion
            dominant_emotion=max(emotion_result[0]["emotion"],key=emotion_result[0]["emotion"].get)
            #Print result
            cv.putText(frame,f"emotion:{dominant_emotion}",(x,y-10),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
            #Show Screen
            cv.imshow("EmotionDetection",frame)
            k=cv.waitKey(5) & 0xFF
            if k==27: #Press esc for quit
                break
        elif k== 112: #Press p for pause
            while True:
                key=cv.waitKey(0)&0xFF
                break    
            if key==115: #Press s for taking screenshot
                ss_path = os.path.join(ss_folder, f"screenshot{ss_index}.png")
                cv.imwrite(ss_path,frame)
                print(f"screenshot taken to{ss_path}.")
                ss_index +=1        
        elif current_frame==target_frame:
            break        
        
#Screenshot Block
program_dir= os.path.dirname(os.path.abspath(__file__))
ss_folder= os.path.join(program_dir,"Screenshots")
if not os.path.exists(ss_folder):
    os.makedirs(ss_folder)
ss_index=1
#Main Block
frame_processor(10)  
cv.destroyAllWindows()    