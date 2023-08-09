import cv2
import argparse
import time
import src.utils as utils
import src.roi as roi
import csv

client = None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set', action='store_true') 
    args = parser.parse_args()
    
    
    cap = cv2.VideoCapture(0)
        
    if not cap.isOpened():
        print("Cannot open a video capture.")
        exit(-1)
        
    # fps = int(cap.get(cv2.CAP_PROP_FPS))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # print(f'FPS:{fps}')
    print(f'resolution:{width}x{height}')
    
    roi_dict = {}
    
    # 自分で指定する場合
    if args.set:
        roi_dict = roi.select_roi(cap, 'video')
        roi_list = []
        for name, (x, y, w, h) in roi_dict.items():
            roi_list.append([name, x, y, w, h])
            
        with open('src/roi.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerows(roi_list)
            
    # csvファイルから読み込む場合 
    else:
        with open('src/roi.csv') as f:
            reader = csv.reader(f)
            for row in reader:
                name = row[0]
                x, y, w, h = map(int, row[1:])
                roi_dict[name] = (x, y, w, h)
        
        
            

        
    # メインループ部分
    pre_t = time.time()
    while True:
        t = time.time() - pre_t
        pre_t = time.time()
        try:
            fps = 1/t
        except ZeroDivisionError:
            fps = 0
            
        ret, frame = cap.read()
        if not ret:
            print('error')
            break
        #frame = cv2.flip(frame, 1)
        
        frame = utils.detect_pose(frame, roi_dict)
        
        for name, (x,y,w,h) in roi_dict.items():
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.rectangle(frame, (x,y),(x+w, y+h), (0,0,255), 2)

        cv2.putText(frame, f'FPS:{fps:.1f}', (0, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("video", frame)
    
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()
