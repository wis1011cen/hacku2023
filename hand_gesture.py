import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import time

def get_result(result: vision.GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global gesture_dict
    gestures =  result.gestures
    handedness = result.handedness
    gesture_dict = dict()
    if gestures:
        for hand, gesture in zip(handedness, gestures):
            gesture_dict[hand[0].category_name] = gesture[0].category_name
    return gesture_dict

options = vision.GestureRecognizerOptions(
    num_hands=2,
    base_options=python.BaseOptions(model_asset_path='gesture_recognizer.task'),
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=get_result)

recognizer = vision.GestureRecognizer.create_from_options(options)

gesture_dict = dict()
cap = cv2.VideoCapture(0)


frame_timestamp_ms = 0
pre_t = time.time()

while True:
    t = time.time() - pre_t
    pre_t = time.time()
    
    try:
        fps = 1/t
    except ZeroDivisionError:
        fps = 0
    
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    recognizer.recognize_async(mp_image, frame_timestamp_ms)
    i = 0
    for hand, gesture in gesture_dict.items():
        cv2.putText(frame, f'{hand}:{gesture}', (0, 60+30*i), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        i += 1
        
    frame_timestamp_ms += 1
    cv2.putText(frame, f'FPS:{fps:.1f}', (0, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('frame', frame)
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()