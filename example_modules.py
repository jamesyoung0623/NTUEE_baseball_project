import argparse
import numpy  as np
import math
import cv2
import sys
import time

def new_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i1', '--input1', type=str, default='material/LHB_240FPS/Lin_toss_1227 (1).avi', help='route of video1')
    parser.add_argument('-i2', '--input2', type=str, default='material/LHB_240FPS/Lin_toss_1227 (2).avi', help='route of video2')
    parser.add_argument('-w', '--waitKey', type=int, default=1, help='-w 0 for control of the clip, keep pressing any key to play')
    
    return parser

def read_clip_mono(path):
    cap = cv2.VideoCapture(path)
    clip_buf=[]
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            break 
        mono = frame[:,:,1]
        clip_buf.append(mono)
    return clip_buf

def read_clip_rgb(path):
    cap = cv2.VideoCapture(path)
    clip_buf=[]
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            break 
        clip_buf.append(frame)
    return clip_buf

class MovingBallDetector(object):
    def __init__(self, frame_1, frame_2, hist=8, thres=16, kr=7):
        self.WINDOW_NAME = "Example image"
        self.roi_1 = self.cut_roi(frame_1)
        self.roi_2 = self.cut_roi(frame_2)
        self.frame_count_1 = 0
        self.frame_count_2 = 0
        self.flag_1 = 0
        self.flag_2 = 0
        self.H, self.W = self.roi_1.shape 
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=hist, varThreshold=thres, detectShadows=False) 
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kr,kr))  
        blob_params = self.set_blob_params()
        self.blob_detector = cv2.SimpleBlobDetector_create(blob_params)

    def cut_roi(self, img):
        return img[0:540,0:720]

    def gen_differential_img(self, frame_1, frame_2, mog=False):
        fgmask_1 = self.fgbg.apply(frame_1) 
        fgmask_2 = self.fgbg.apply(frame_2) 
        if mog:
            fgmask_mog2_1 = fgmask_1
            fgmask_mog2_1 = cv2.morphologyEx(fgmask_mog2_1, cv2.MORPH_CLOSE, self.kernel) 
            fgmask_mog2_1 = cv2.morphologyEx(fgmask_mog2_1, cv2.MORPH_OPEN, self.kernel) 
            fgmask_mog2_2 = fgmask_2
            fgmask_mog2_2 = cv2.morphologyEx(fgmask_mog2_2, cv2.MORPH_CLOSE, self.kernel) 
            fgmask_mog2_2 = cv2.morphologyEx(fgmask_mog2_2, cv2.MORPH_OPEN, self.kernel)
            return fgmask_mog2_1, fgmask_mog2_2
        return fgmask_mog2_1, fgmask_mog2_2
    
    def set_blob_params(self):
        ball_r = 10 
        params = cv2.SimpleBlobDetector_Params()
        # Change thresholds
        params.minThreshold = 10;
        params.maxThreshold = 100;
        # Filter by Area.
        params.filterByArea = True  # radius = 20~30 pixels (plate width = 265 pixels)
        params.minArea = ball_r*ball_r*math.pi *0.5 #    
        params.maxArea = ball_r*ball_r*math.pi *1.8 # 
        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.5
        # Filter by Convexity
        params.filterByConvexity = False
        params.minConvexity = 0.6
        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.6
        return params

    def draw_blob_detected_ball_on_img(self, img_1, img_2):
        inv_img_1 = cv2.bitwise_not(img_1)
        inv_img_2 = cv2.bitwise_not(img_2)
        keypoints_1 = self.blob_detector.detect(inv_img_1)
        keypoints_2 = self.blob_detector.detect(inv_img_2)
        img_1 = cv2.drawKeypoints(img_1, keypoints_1, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        img_2 = cv2.drawKeypoints(img_2, keypoints_2, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        if self.flag_1 == 0:
            if len(keypoints_1) != 0:
                self.frame_count_1 += 1 
            elif self.frame_count_1 > 10:
                self.flag_1 += 1
            else:
                self.frame_count_1 = 0
        elif self.flag_2 == 0:
            self.flag_1 += 1
            
        if self.flag_2 == 0:
            if len(keypoints_2) != 0:
                self.frame_count_2 += 1 
            elif self.frame_count_2 > 10:
                self.flag_2 += 1
            else:
                self.frame_count_2 = 0
        elif self.flag_1 == 0:
            self.flag_2 += 1
        
        delta = self.flag_1-self.flag_2
        #for kp in keypoints_1:
        #    print("kp_1.size="+str(kp.size))
        #    print("kp_1.xy = (%d, %d)\n"%(kp.pt[0], kp.pt[1]))
        #    img = cv2.hconcat([img_1, img_2])
        #    cv2.imshow(self.WINDOW_NAME, img)
        #    cv2.waitKey(0)
            
        #for kp in keypoints_2:
        #    print("kp_2.size="+str(kp.size))
        #    print("kp_2.xy = (%d, %d)\n"%(kp.pt[0], kp.pt[1]))
        #    img = cv2.hconcat([img_1, img_2])
        #    cv2.imshow(self.WINDOW_NAME, img)
        #    cv2.waitKey(0)
                    
        return img_1, img_2, delta

    def demo_video(self, clip_1, clip_2):
        b1, b2 = [], []
        for i in range(len(clip_1)):
            blob = cv2.vconcat([clip_1[i], clip_2[i]])
            fgmask_1, fgmask_2 = self.gen_differential_img(clip_1[i], clip_2[i], mog=True)
            blob_1, blob_2, delta = self.draw_blob_detected_ball_on_img(fgmask_1, fgmask_2)
            b1.append(blob_1)
            b2.append(blob_2)
            #blob = cv2.hconcat([blob_1, blob_2])
            #cv2.imshow(self.WINDOW_NAME, blob)
            #cv2.waitKey(1)
            
        return b1, b2, delta
        
def run_param_for_bgs():
    parser = new_parser()
    arg = parser.parse_args()
    WINDOW_NAME = "Example image"
    clip_1 = read_clip_mono(arg.input1)
    clip_2 = read_clip_mono(arg.input2)
    frame_total= len(clip_1)
    
    for thres in [6, 32]:
        for hist in [32, 64]:
            t0=time.time()
            print("Sychronizing with hist = {0}, thres = {1}...".format(hist, thres))
            ball_detector = MovingBallDetector(clip_1[0], clip_2[0], hist=hist, thres=thres, kr=3)
            b1, b2, delta = ball_detector.demo_video(clip_1[0:frame_total], clip_2[0:frame_total])
    
            if delta > 0:
                for i in range(len(clip_1)-delta):
                    blob = cv2.hconcat([b1[i], b2[i+delta]])
                    cv2.imshow(WINDOW_NAME, blob)
                    cv2.waitKey(arg.waitKey)
                for i in range(len(clip_1)-delta):
                    clip = cv2.hconcat([clip_1[i], clip_2[i+delta]])
                    cv2.imshow(WINDOW_NAME, clip)
                    cv2.waitKey(arg.waitKey)
            else:
                for i in range(len(clip_1)+delta-1):
                    blob = cv2.hconcat([b1[i-delta+1], b2[i]])
                    cv2.imshow(WINDOW_NAME, blob)
                    cv2.waitKey(arg.waitKey)
                for i in range(len(clip_1)+delta-1):
                    clip = cv2.hconcat([clip_1[i-delta+1], clip_2[i]])
                    cv2.imshow(WINDOW_NAME, clip)
                    cv2.waitKey(arg.waitKey)
            print("ms per frame: {}".format((time.time()-t0)*1000/frame_total))
            
run_param_for_bgs()
# test_clip()