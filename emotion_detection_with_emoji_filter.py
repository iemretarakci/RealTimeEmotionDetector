import cv2 as cv
import os
import logging
from deepface import DeepFace
import time
model = DeepFace.build_model("Emotion")
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Screenshot Block
program_dir = os.path.dirname(os.path.abspath(__file__))
ss_folder = os.path.join(program_dir, "Screenshots")
if not os.path.exists(ss_folder):
    os.makedirs(ss_folder)
ss_index = 1

# Emoji directory
emoji_dir = os.path.join(program_dir, "emoji")

# Frame Processing Block
def frame_processor():
    cap = cv.VideoCapture(0)
    frame_count = 0  # Counter for frames processed
    while True:
            ret, frame = cap.read()
            if not ret:
                break

            faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)

            frame_count += 1
            if frame_count % 1 == 0:  # Perform analysis every 1 frame
                for (x, y, w, h) in faces:
                    if w > 30 and h > 30:
                        face_roi = frame[y:y+h, x:x+w]
                        emotion_result = DeepFace.analyze(face_roi, ["emotion"], enforce_detection=False,detector_backend="opencv")
                        dominant_emotion = max(emotion_result[0]["emotion"], key=emotion_result[0]["emotion"].get)
                        cv.putText(frame,f"emotion:{dominant_emotion}",(x,y-10),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)                        
                        # Take emotion info for filtering correct emoji 
                        if len(emotion_buffer) >= BUFFER_SIZE:
                            emotion_buffer.pop(0)
                        emotion_buffer.append(dominant_emotion)
                        average_emotion = max(set(emotion_buffer),key=emotion_buffer.count)
                        
                        emoji_path = os.path.join(emoji_dir, f"{average_emotion}.png")
                        emoji = cv.imread(emoji_path, cv.IMREAD_UNCHANGED)
                        if emoji is not None:
                            resized_emoji = cv.resize(emoji, (w, h))
                            alpha_s = resized_emoji[:,:,3]/255.0
                            alpha_l = 1.0 - alpha_s
                            for c in range(0, 3):
                                frame[y:y+h, x:x+w, c] = (alpha_s * resized_emoji[:,:,c] +
                                                           alpha_l * frame[y:y+h, x:x+w, c])

                cv.imshow('Frame', frame)
                k = cv.waitKey(5)
                time.sleep(0.05)
                if k == 27:  # Press esc for quit
                    break
                elif k == 112:  # Press p for pause
                    cv.waitKey(0)
                elif k == 115:  # Press s for taking screenshot
                    ss_path = os.path.join(ss_folder, f"screenshot{ss_index}.png")
                    cv.imwrite(ss_path, frame)
                    logging.info(f"screenshot taken to {ss_path}.")
                    ss_index += 1
    cap.release()
    cv.destroyAllWindows()

# Main Block
BUFFER_SIZE = 7  
emotion_buffer = []
frame_processor()
