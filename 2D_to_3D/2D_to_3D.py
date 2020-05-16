import numpy  as np
import cv2
import math
import matplotlib.pyplot as plt

mount_angle = 23*math.pi/180
cam_1_theta = 20.75*math.pi/180
cam_1_horizontal_visual_angle = 4.74*math.pi/180
cam_1_vertical_visual_angle = 4.74*math.pi/180
cam_1_fi = 2.36*math.pi/180
cam_2_fi = 11.6*math.pi/180

#y = 0 in cam1 u = 530
#x = 0 in cam2 u = 310

#v = 220~260 in camera 2 is at z = 1
#v = 400~440 in camera 2 is at z = 0

#v = 250~260 in camera 1 is z = 1
#v = 90~100 in camera 1 is z = 0
#u = 340~360 in camera 1 is y = 0
#u = 480~500 in camera 1 is y = 1
cap = cv2.VideoCapture('1635_43_cam_2.avi')
ret, frame = cap.read()
cv2.imshow('test', frame)
plt.imshow(frame)
plt.show()
cap = cv2.VideoCapture('1635_43_cam_0.avi')
ret, frame = cap.read()
plt.imshow(frame)

def calculate_x(cam_1_coord, cam_2_coord):
    x = (cam_2_coord[0]-310)/190
    return x

def calculate_y(cam_1_coord, cam_2_coord):
    x = calculate_x(cam_1_coord, cam_2_coord)
    #the number of pixels from cam to the screen
    #with estimated horizontal visual angle
    pix_to_screen = 360/math.tan(cam_1_horizontal_visual_angle)
    #the angle between the line from cam to the target and x axis
    obj_angle = math.atan((cam_1_coord[0]-360)/pix_to_screen) + cam_1_theta
    #the x distance between cam and the target in the real world
    delta = 31*math.cos(mount_angle) - x 
    #y coord by adjusting the distance along y axis with the plate
    y = (delta*math.tan(obj_angle) - 31*math.sin(mount_angle))/1.4 #fine tune
    return y
    
def calculate_z(cam_1_coord, cam_2_coord):
    x = calculate_x(cam_1_coord, cam_2_coord)
    y = calculate_y(cam_1_coord, cam_2_coord)
    #the number of pixels from cam to the screen
    #with estimated vertical visual angle
    pix_to_screen = 270/math.tan(cam_1_vertical_visual_angle)
    #the angle between the line from cam to the target and x axis
    obj_angle = cam_1_fi - math.atan((cam_1_coord[1]-270)/pix_to_screen)
    #the x distance between cam and the target in the real world
    delta = math.sqrt((31*math.cos(mount_angle) - x)**2 + (31*math.sin(mount_angle) + y)**2)
    #z coord by adjusting the distance along z axis with the plate
    z = (2.9 - delta*math.tan(obj_angle))/1.4 #fine tune
    return z
    
    
def threeDreconstrution(cam_1_coord, cam_2_coord):
    x = calculate_x(cam_1_coord, cam_2_coord)
    y = calculate_y(cam_1_coord, cam_2_coord)
    z = calculate_z(cam_1_coord, cam_2_coord)
    return (x, y, z)

cam_1_coord = list(input('Input coordinate of camera 1: '))
cam_1_coord[0], cam_1_coord[1] = float(cam_1_coord[0]), float(cam_1_coord[1])
cam_2_coord = list(input('Input coordinate of camera 2: '))
cam_2_coord[0], cam_2_coord[1] = float(cam_2_coord[0]), float(cam_2_coord[1])

threeD_coord = threeDreconstrution(cam_1_coord, cam_2_coord)
print(threeD_coord)