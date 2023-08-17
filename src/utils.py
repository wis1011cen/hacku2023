import cv2
import numpy as np
import time
import ir.irrp as irrp

 
SCALE = 2
RATIO = 0.5

L_DURATION = 0.5
L_COOL_TIME = 1
R_DURATION = 0.5
R_COOL_TIME = 1
DEGREE_THRESHOLD = 120
VISIBILITY_THRESHOLD = 0.7
OPERATION_DISPLAY_TIME = 0.7



operation_time = 0
pre_wrist_pos = np.zeros(2)
operation_name = None
pre_name = None

def load_start_time_dict(input_start_time_dict):
    global start_time_dict
    start_time_dict = input_start_time_dict
 

def arm_operation(landmark_dict, annotated_frame, appliance_dict, gesture_dict):
    global start_time_dict
    
    cv2.putText(annotated_frame, f"Left: {gesture_dict['Left']}", (0, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    cv2.putText(annotated_frame, f"Right: {gesture_dict['Right']}", (0, 150), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        
    if operation_name is not None and (time.time() - operation_time) < OPERATION_DISPLAY_TIME:
        x, y, w, h = appliance_dict[operation_name.split('-')[0]]
        cv2.putText(annotated_frame, operation_name, (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        cv2.rectangle(annotated_frame, (x, y),(x+w, y+h), (0, 0, 255), 2)
  
    
    if landmark_dict['l_visibility'] > VISIBILITY_THRESHOLD:
        degree = calculate_degree(landmark_dict['l_shoulder'], landmark_dict['l_elbow'], landmark_dict['l_wrist'])
        if degree > DEGREE_THRESHOLD:
            l_ir_operation(annotated_frame, landmark_dict['l_wrist'], appliance_dict, landmark_dict['l_elbow'], gesture_dict['Left'])
        else:
            start_time_dict['Left'] = {name : 0 for name in start_time_dict['Left'].keys()}
   
    if landmark_dict['r_visibility'] > VISIBILITY_THRESHOLD:
        degree = calculate_degree(landmark_dict['r_shoulder'], landmark_dict['r_elbow'], landmark_dict['r_wrist'])
        if degree > DEGREE_THRESHOLD:
            r_ir_operation(annotated_frame, landmark_dict['r_wrist'], appliance_dict, landmark_dict['r_elbow'], gesture_dict['Right'])
        else:
            start_time_dict['Left'] = {name : 0 for name in start_time_dict['Left'].keys()}
        
        
# 3点のなす角度を求める
def calculate_degree(pos1, pos2, pos3):  
    vec_a = pos1 - pos2
    vec_b = pos3 - pos2      
    len_vec_a = np.linalg.norm(vec_a)
    len_vec_b = np.linalg.norm(vec_b)
    
    cos = np.inner(vec_a, vec_b)/(len_vec_a * len_vec_b)
    deg = np.rad2deg(np.arccos(cos))
    return deg


    
# 左腕で電源ON
def l_ir_operation(annotated_frame, wrist_pos, appliance_dict, elbow_pos, l_gesture):
    global operation_time, start_time_dict, pre_wrist_pos, pre_name
    extended_pos = wrist_pos + 20 * (wrist_pos - elbow_pos)    #腕を伸ばした先
    color = (255, 0, 0)
    
    if (time.time() - operation_time) > L_COOL_TIME or operation_time == 0:
        for appliance_name, appliance_pos in appliance_dict.items():
            color = (255, 0, 0)
            
            #線分が交わるか判定
            if hit_detection(wrist_pos, extended_pos, appliance_pos):
                if l_gesture == 'Thumb_Up':
                    end = 'up' 
                    ir_operation(appliance_name, end, 'Left')
                elif l_gesture == 'Thumb_Down':
                    end = 'down'
                    ir_operation(appliance_name, end, 'Left')
                elif l_gesture == 'Pointing_Up':
                    end = 'on'
                    ir_operation(appliance_name, end, 'Left')
                            
                color = (0, 0, 255)
                break

            
    cv2.line(annotated_frame, wrist_pos, extended_pos, color, 2)
    
# 右腕:チャンネル変更 
def r_ir_operation(annotated_frame, wrist_pos, appliance_dict, elbow_pos, r_pre_gesture):
    global start_time_dict, operation_time, pre_wrist_pos, pre_name
    ex_pos = wrist_pos + 20 * (wrist_pos - elbow_pos)    #腕を伸ばした先
    color = (255, 0, 0)
    if (time.time() - operation_time) > R_COOL_TIME or operation_time == 0:
        for appliance_name, appliance_pos in appliance_dict.items():
            color = (255, 0, 0)
            # 線分が交わるか判定
            if hit_detection(wrist_pos, ex_pos, appliance_pos):
                if r_pre_gesture == 'Thumb_Up':
                    end = 'up' 
                    ir_operation(appliance_name, end, 'Right')
                elif r_pre_gesture == 'Thumb_Down':
                    end = 'down'
                    ir_operation(appliance_name, end, 'Right') 
                color = (0, 0, 255)
                break   
    
            
    cv2.line(annotated_frame, wrist_pos, ex_pos, color, 2)
  


def ir_operation(name, end, hand):
    global operation_time, operation_name
    operation_name = f'{name}-{end}'
    print(operation_name)
    #irrp.ir_lightning(operation_name)

    operation_time = time.time()
    
    # start_time_dict[hand][name] = 0
                        
    
# 操作対象と線の当たり判定
def hit_detection(pos1, pos2, appliance_pos):
    x, y, w, h = appliance_pos
    ul_pos = np.array([x, y])       #左上
    ur_pos = np.array([x+w, y])     #右上
    lr_pos = np.array([x+w, y+h])   #右下
    ll_pos = np.array([x, y+h])     #左下

    left_edge_TF: bool = cross_detection(pos1, pos2, ul_pos, ll_pos)
    right_edge_TF: bool = cross_detection(pos1, pos2, ur_pos, lr_pos)
    top_edge_TF: bool = cross_detection(pos1, pos2, ul_pos, ur_pos)
    bottom_edge_TF: bool = cross_detection(pos1, pos2, ll_pos, lr_pos)
    
    if left_edge_TF or right_edge_TF or top_edge_TF or bottom_edge_TF:
        return True
    else:
        return False
    
# 線分と線分の当たり判定
def cross_detection(pos_a, pos_b, pos_c, pos_d):
    vec_ab = pos_b - pos_a
    vec_ac = pos_c - pos_a
    vec_ad = pos_d - pos_a
    product1 = np.cross(vec_ab, vec_ac) * np.cross(vec_ab, vec_ad)
        
    vec_cd = pos_d - pos_c
    vec_cb = pos_b - pos_c
    vec_ca = - vec_ac
    product2 = np.cross(vec_cd, vec_cb) * np.cross(vec_cd, vec_ca)
    
    if product1 < 0 and product2 < 0:
        return True
    else:
        return False
    

def rotate_and_crop(landmark_dict, input_frame, hand):

    wrist = f'{hand}_wrist'
    elbow = f'{hand}_elbow'
    shoulder = f'{hand}_shoulder'
    index = f'{hand}_index'
    hip = f'{hand}_hip'
    
    angle = calculate_degree(landmark_dict[wrist], landmark_dict[shoulder], np.array((0, landmark_dict[shoulder][1])))
    y_wrist = landmark_dict[wrist][1] 
    y_elbow = landmark_dict[elbow][1]

    if angle < 90:
        if y_wrist > y_elbow:
            angle1 = - 90 - angle
            angle2 = -angle
        else:
            angle1 = angle - 90
            angle2 = angle
    else:
        if y_wrist > y_elbow:
            angle1 = 270 -angle
            angle2 = 180 -angle 
        else:
            angle1 = angle -90
            angle2 = angle -180

    center = (int(landmark_dict[index][0]), int(landmark_dict[index][1]))
    rotate_matrix1 = cv2.getRotationMatrix2D(center=center, angle=angle1, scale=1)
    rotate_matrix2 = cv2.getRotationMatrix2D(center=center, angle=angle2, scale=1)
    rotated_frame1 = cv2.warpAffine(src=input_frame, M=rotate_matrix1, dsize=(640*SCALE, 360*SCALE))
    rotated_frame2 = cv2.warpAffine(src=input_frame, M=rotate_matrix2, dsize=(640*SCALE, 360*SCALE))
    
    size = abs(landmark_dict[shoulder][1] - landmark_dict[hip][1])
    
    min_y = max(int(landmark_dict[index][1] - RATIO*size) , 0)
    min_x = max(int(landmark_dict[index][0] - RATIO*size), 0)
    max_y = int(landmark_dict[index][1] + RATIO*size)
    max_x = int(landmark_dict[index][0] + RATIO*size)
    
    cropped_frame1 = rotated_frame1[min_y : max_y, min_x : max_x]
    cropped_frame2 = rotated_frame2[min_y : max_y, min_x : max_x]
    
    cv2.rectangle(rotated_frame1, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
    cv2.rectangle(rotated_frame2, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
    
    if not(cropped_frame1.shape[0] > 0 and cropped_frame1.shape[1] > 0):
        cropped_frame1 = np.zeros((360*SCALE, 640*SCALE, 3), dtype=np.uint8)
    if not(cropped_frame2.shape[0] > 0 and cropped_frame2.shape[1] > 0):
        cropped_frame2 = np.zeros((360*SCALE, 640*SCALE, 3,), dtype=np.uint8)
        
    return rotated_frame1, rotated_frame2, cropped_frame1, cropped_frame2

def concat_rotated_and_cropped(landmark_dict, input_frame):
    l_rotated_frame1, l_rotated_frame2, l_cropped_frame1, l_cropped_frame2 = rotate_and_crop(landmark_dict, input_frame, 'l')
    r_rotated_frame1, r_rotated_frame2, r_cropped_frame1, r_cropped_frame2 = rotate_and_crop(landmark_dict, input_frame, 'r')
    rotated_frame = cv2.hconcat([r_rotated_frame1, r_rotated_frame2, l_rotated_frame1, l_rotated_frame2])
    
    diff = l_cropped_frame1.shape[0] - r_cropped_frame1.shape[0]
    
    if diff > 0:
        l_padding1 = l_cropped_frame1
        l_padding2 = l_cropped_frame2
        r_padding1 = cv2.copyMakeBorder(r_cropped_frame1, 0, diff, 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
        r_padding2 = cv2.copyMakeBorder(r_cropped_frame2, 0, diff, 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
    else:
        l_padding1 = cv2.copyMakeBorder(l_cropped_frame1, 0, -diff, 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
        l_padding2 = cv2.copyMakeBorder(l_cropped_frame2, 0, -diff, 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
        r_padding1 = r_cropped_frame1
        r_padding2 = r_cropped_frame2
        
    
    cropped_frame = cv2.hconcat([r_padding1, r_padding2, l_padding1, l_padding2])
    
    return rotated_frame, cropped_frame
 

    
    