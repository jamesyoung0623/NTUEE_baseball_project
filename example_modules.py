import argparse
import numpy  as np
import cv2
import time
import math

def new_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i1', '--input1', type=str, default='material/LHB_240FPS/Lin_toss_1227 (1).avi', help='route of video1')
    parser.add_argument('-i2', '--input2', type=str, default='material/LHB_240FPS/Tang_toss_0101.avi', help='route of video2')
    parser.add_argument('-w', '--waitKey', type=int, default=1, help='-w 0 for control of the clip, keep pressing any key to play')
    parser.add_argument('-hist', '--history', type=int, default=32, help='history')
    parser.add_argument('-t', '--threshold', type=int, default=64, help='threshold')
    
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
    def __init__(self, frame, hist=8, thres=16, kr=7):
        self.WINDOW_NAME = "Example image"
        self.roi = self.cut_roi(frame)
        self.H, self.W = self.roi.shape 
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=hist, varThreshold=thres, detectShadows=False) 
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kr,kr))  
        blob_params = self.set_blob_params()
        self.blob_detector = cv2.SimpleBlobDetector_create(blob_params)

    def cut_roi(self, img):
        return img[0:540,0:720]

    def gen_differential_img(self, frame, mog=False):
        fgmask = self.fgbg.apply(frame)  
        if mog:
            fgmask_mog2 = fgmask
            fgmask_mog2 = cv2.morphologyEx(fgmask_mog2, cv2.MORPH_CLOSE, self.kernel) 
            fgmask_mog2 = cv2.morphologyEx(fgmask_mog2, cv2.MORPH_OPEN, self.kernel) 
            return fgmask_mog2
        return fgmask_mog2
    
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

    def draw_blob_detected_ball_on_img(self, img):
        inv_img = cv2.bitwise_not(img)
        keypoints = self.blob_detector.detect(inv_img)
        img = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        return img

    def demo_video(self, clip):
        for i in range(len(clip)):
            fgmask = self.gen_differential_img(clip, mog=True)
            blob = self.draw_blob_detected_ball_on_img(fgmask)
            cv2.imshow(self.WINDOW_NAME, blob)
            cv2.waitKey(1)
    
    def find_batting_frame(self, clip):
        #observe the long consecutive existing keypoint before batting frame
        consecutive_keypoint_frame_count = 0
        
        for i in range(len(clip)):
            fgmask = self.gen_differential_img(clip[i], mog=True)
            inv_img = cv2.bitwise_not(fgmask)
            keypoints = self.blob_detector.detect(inv_img)
            img = cv2.drawKeypoints(fgmask, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            
            if len(keypoints) != 0:
                consecutive_keypoint_frame_count += 1 
            elif consecutive_keypoint_frame_count > 10: #consider 10 long enough
                return i
            else:
                consecutive_keypoint_frame_count = 0
        return len(clip)        
                
def run_param_for_bgs():
    parser = new_parser() 
    arg = parser.parse_args()
    WINDOW_NAME = "Example image"
    clip_1 = read_clip_mono(arg.input1)
    clip_2 = read_clip_mono(arg.input2)
    frame_total= len(clip_1)
    hist = arg.history
    thres = arg.threshold
    #t0=time.time()

    print("Sychronizing with hist = {0}, thres = {1}...".format(hist, thres))
    ball_detector_1 = MovingBallDetector(clip_1[0], hist=hist, thres=thres, kr=3)
    ball_detector_2 = MovingBallDetector(clip_2[0], hist=hist, thres=thres, kr=3)

    batting_frame_1 = ball_detector_1.find_batting_frame(clip_1)
    batting_frame_2 = ball_detector_2.find_batting_frame(clip_2)

    delta = batting_frame_1 - batting_frame_2

    if delta > 0:
        for i in range(len(clip_1)-delta):
            clip = cv2.hconcat([clip_1[i+delta], clip_2[i]])
            cv2.imshow(WINDOW_NAME, clip)
            cv2.waitKey(arg.waitKey)
    else:
        for i in range(len(clip_2)+delta):
            clip = cv2.hconcat([clip_1[i], clip_2[i-delta]])
            cv2.imshow(WINDOW_NAME, clip)
            cv2.waitKey(arg.waitKey)
    #print("ms per frame: {}".format((time.time()-t0)*1000/frame_total))
            
run_param_for_bgs()
# test_clip()