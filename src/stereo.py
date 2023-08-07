
# import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import numpy as np
import cv2
import json
import sys

# from lib.extended_json import ExtendedJsonEncoder

"""
CONFIG
"""
# print(cv2.__version__)
# L_CAMERA_DEVISE = 0
# R_CAMERA_DEVICE = 1 
# width = 640
# height = 480


"""
"""

def load_caliblation_data(width, height):
    CALIBRATION_DATA = "stereo_calibration_data.json"
    print("load calibaration data")

    fr = open(CALIBRATION_DATA, 'r')
    calibration_data = json.load(fr)

    cameramatrixl = np.array(calibration_data['cameramatrixl'])
    cameramatrixr = np.array(calibration_data['cameramatrixr'])
    distcoeffsl = np.array(calibration_data['distcoeffsl'])
    distcoeffsr = np.array(calibration_data['distcoeffsr'])
    R = np.array(calibration_data['R'])
    T = np.array(calibration_data['T'])

    print("End load calibaration data")
    
    # SGBM Parameters -----------------
    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size. image (1300px and above); 5 Works nicely
    window_size = 11
    min_disp = 4
    num_disp = 128  # max_disp has to be dividable by 16 f. E. HH 192, 256
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=min_disp,  # 視差の下限
        numDisparities=num_disp,  # 視差の上限
        blockSize=window_size,  # 窓サイズ 3..11
        P1=8 * 3 * window_size**2,  # 視差の滑らかさを制御するパラメータ1
        P2=32 * 3 * window_size**2,  # 視差の滑らかさを制御するパラメータ2
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=50,  # 視差の滑らかさの最大サイズ. 50-100
        speckleRange=2,  # 視差の最大変化量. 1 or 2
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    # FILTER Parameters
    lmbda = 80000
    sigma = 1.2
    visual_multiplier = 1.0

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    # 平行化変換のためのRとPおよび3次元変換行列Qを求める
    flags = 0
    alpha = 1
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
        cameramatrixl, distcoeffsl, cameramatrixr, distcoeffsr, (width, height), R, T, flags, alpha, (width, height))

    # 平行化変換マップを求める
    m1type = cv2.CV_32FC1
    map1_l, map2_l = cv2.initUndistortRectifyMap(
        cameramatrixl, distcoeffsl, R1, P1, (width, height), m1type)  # m1type省略不可
    map1_r, map2_r = cv2.initUndistortRectifyMap(
        cameramatrixr, distcoeffsr, R2, P2, (width, height), m1type)
    
    return map1_l, map2_l, map1_r, map2_r, wls_filter, left_matcher, right_matcher


    
def depth_estimate(framel, framer, map1_l, map2_l, map1_r, map2_r, wls_filter, left_matcher, right_matcher):

    framel_gray = cv2.cvtColor(framel, cv2.COLOR_BGR2GRAY)
    framer_gray = cv2.cvtColor(framer, cv2.COLOR_BGR2GRAY)


    # ReMapにより平行化を行う
    interpolation = cv2.INTER_NEAREST  # INTER_RINEARはなぜか使えない
    framel_gray = cv2.remap(framel_gray, map1_l, map2_l,
                        interpolation)  # interpolation省略不可
    framer_gray = cv2.remap(framer_gray, map1_r, map2_r, interpolation)

    framel_gray = cv2.GaussianBlur(framel_gray, (5, 5), 0)
    framer_gray = cv2.GaussianBlur(framer_gray, (5, 5), 0)

    # cv2.imshow("Left", imgl_gray)
    # cv2.imshow("Right", imgr_gray)

    # cv2.imshow("Left", imgl)
    # cv2.imshow("Right", imgr)

    displ = left_matcher.compute(framel_gray, framer_gray)
    dispr = right_matcher.compute(framer_gray, framel_gray)
    # displ = left_matcher.compute(imgl, imgr)
    # dispr = right_matcher.compute(imgr, imgl)

    # cv2.imshow("Disparity", (displ.astype(
    #     np.float32) / 16.0 - min_disp) / num_disp)

    displ = np.int16(displ)
    dispr = np.int16(dispr)

    depth_map = wls_filter.filter(displ, framel, None, dispr)
  
    
    return framel_gray, framer_gray, depth_map
    
    
def test():
    L_CAMERA_DEVISE = 0
    R_CAMERA_DEVICE = 1 
    WIDTH = 640
    HEIGHT = 480
    
    capl = cv2.VideoCapture(L_CAMERA_DEVISE)
    capr = cv2.VideoCapture(R_CAMERA_DEVICE)

    capl.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    capl.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    capr.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    capr.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    capl.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
    capr.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
    capl.set(cv2.CAP_PROP_EXPOSURE, -5)
    capr.set(cv2.CAP_PROP_EXPOSURE, -5)
    
    
    map1_l, map2_l, map1_r, map2_r, wls_filter, left_matcher, right_matcher = load_caliblation_data(WIDTH, HEIGHT)
    
    while True:
        retl, framel = capl.read()
        retr, framer = capr.read()
        
        if not retl or not retr:
            print('No more frames')
            break
            
        framel_gray, framer_gray, filtered_frame = depth_estimate(framel, framer, map1_l, map2_l, map1_r, map2_r, wls_filter, left_matcher, right_matcher)
        
        cv2.imshow('Stereo', cv2.hconcat([framel_gray, framer_gray, filtered_frame]))
        k = cv2.waitKey(1)
            
        if k == ord('q'):
            break
        

    capl.release()
    capr.release()

    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    test()
