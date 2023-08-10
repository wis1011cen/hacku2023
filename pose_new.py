import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import time
base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode
# BaseOptions = mp.tasks.BaseOptions
# PoseLandmarker = mp.tasks.vision.PoseLandmarker
# PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
# PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult

# VisionRunningMode = mp.tasks.vision.RunningMode

# Create a gesture recognizer instance with the live stream mode:
def print_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    print('gesture recognition result: {}'.format(result))
    print()
    
def get_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global gesture_dict
    gestures =  result.gestures
    handedness = result.handedness
    gesture_dict = dict()
    if gestures:
        for hand, gesture in zip(handedness, gestures):
            # print(hand[0].category_name, gesture[0].category_name)
            gesture_dict[hand[0].category_name] = gesture[0].category_name
        # print(gesture_dict)
    return gesture_dict

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='pose_landmarker_lite.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

gesture_dict = dict()
cap = cv2.VideoCapture(0)

with PoseLandmarker.create_from_options(options) as recognizer:
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
        
        # i = 0
        # for hand, gesture in gesture_dict.items():
        #     cv2.putText(frame, f'{hand}:{gesture}', (0, 60+30*i), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        #     i += 1
            
        frame_timestamp_ms += 1
        cv2.putText(frame, f'FPS:{fps:.1f}', (0, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('frame', frame)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
          break