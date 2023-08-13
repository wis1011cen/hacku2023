import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import time
from matplotlib import pyplot as plt
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import math
import numpy as np

plt.rcParams.update({
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': False,
    'axes.spines.bottom': False,
    'xtick.labelbottom': False,
    'xtick.bottom': False,
    'ytick.labelleft': False,
    'ytick.left': False,
    'xtick.labeltop': False,
    'xtick.top': False,
    'ytick.labelright': False,
    'ytick.right': False
})

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def display_one_image(image, title, subplot, titlesize=16):
    """Displays one image along with the predicted category name and score."""
    plt.subplot(*subplot)
    plt.imshow(image)
    if len(title) > 0:
        plt.title(title, fontsize=int(titlesize), color='black', fontdict={'verticalalignment':'center'}, pad=int(titlesize/1.5))
    return (subplot[0], subplot[1], subplot[2]+1)

def display_batch_of_images_with_gestures_and_hand_landmarks(images, results):
    """Displays a batch of images with the gesture category and its score along with the hand landmarks."""
    # Images and labels.
    images = [image.numpy_view() for image in images]
    gestures = [top_gesture for (top_gesture, _) in results]
    multi_hand_landmarks_list = [multi_hand_landmarks for (_, multi_hand_landmarks) in results]

    # Auto-squaring: this will drop data that does not fit into square or square-ish rectangle.
    rows = int(math.sqrt(len(images)))
    cols = len(images) // rows

    # Size and spacing.
    FIGSIZE = 13.0
    SPACING = 0.1
    subplot=(rows,cols, 1)
    if rows < cols:
        plt.figure(figsize=(FIGSIZE,FIGSIZE/cols*rows))
    else:
        plt.figure(figsize=(FIGSIZE/rows*cols,FIGSIZE))

    # Display gestures and hand landmarks.
    for i, (image, gestures) in enumerate(zip(images[:rows*cols], gestures[:rows*cols])):
        title = f"{gestures.category_name} ({gestures.score:.2f})"
        dynamic_titlesize = FIGSIZE*SPACING/max(rows,cols) * 40 + 3
        annotated_image = image.copy()

        for hand_landmarks in multi_hand_landmarks_list[i]:
          hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
          hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
          ])

          mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        subplot = display_one_image(annotated_image, title, subplot, titlesize=dynamic_titlesize)

    # Layout.
    plt.tight_layout()
    plt.subplots_adjust(wspace=SPACING, hspace=SPACING)
    plt.show()
    
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def get_result(result, output_image, timestamp):
    global gesture_dict, annotated_frame
    annotated_frame = np.copy(cv2.cvtColor(output_image.numpy_view(), cv2.COLOR_RGB2BGR))
    # print(result)
    print()
    gestures =  result.gestures
    handedness = result.handedness
    hand_landmarks = result.hand_landmarks
    # print(hand_landmarks)
    multi_hand_landmarks_list = [multi_hand_landmarks for multi_hand_landmarks in hand_landmarks]
    
    gesture_dict = dict()
    if gestures:
        for hand, gesture in zip(handedness, gestures):
            gesture_dict[hand[0].category_name] = gesture[0].category_name
            
   
    for hand_landmarks in multi_hand_landmarks_list:
        print(hand_landmarks)
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks])

        mp_drawing.draw_landmarks(annotated_frame,
                                  hand_landmarks_proto,
                                  mp_hands.HAND_CONNECTIONS,
                                  mp_drawing_styles.get_default_hand_landmarks_style(),
                                  mp_drawing_styles.get_default_hand_connections_style())

    

options = vision.GestureRecognizerOptions(
    num_hands=2,
    base_options=python.BaseOptions(model_asset_path='gesture_recognizer.task'),
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=get_result)

recognizer = vision.GestureRecognizer.create_from_options(options)

gesture_dict = dict()
cap = cv2.VideoCapture(0)
ret, annotated_frame = cap.read()

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
    cv2.imshow('frame', annotated_frame)
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()