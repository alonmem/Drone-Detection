import cv2
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from win32api import GetSystemMetrics
import glob


# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util

cap1 = cv2.VideoCapture(1) #2
cap2 = cv2.VideoCapture(0) #1
#drone image:
droneIMG = cv2.resize(cv2.imread('backgroundIMG/drone_image.jpg'), (50,50))

screenX = GetSystemMetrics(0)
screenY = GetSystemMetrics(1)

# for SIFT
MIN_MATCH_COUNT = 1

# What model to download.
MODEL_NAME = 'drone_graph'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('training', 'object-detection.pbtxt')

NUM_CLASSES = 1

print("Preparing TensorFlow...")
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


#calibrate cameras
print("Calibrating cameras...")
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7*9,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:9].T.reshape(-1,2)


# Arrays to store object points and image points from all the images.
objpointsL = [] # 3d point in real world space
imgpointsL = [] # 2d points in image plane.
objpointsR = []
imgpointsR = []

imgX = []
imgY = []

images = glob.glob('calibration/images/left*.jpg')

for fname in images:
    img = cv2.imread(fname)
    grayL = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, cornersL = cv2.findChessboardCorners(grayL, (7,9),None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpointsL.append(objp)

        cv2.cornerSubPix(grayL,cornersL,(11,11),(-1,-1),criteria)
        imgpointsL.append(cornersL)
        for i in cornersL:
          imgX.append(i[0][0])
          imgY.append(i[0][1])


images = glob.glob('calibration/images/right*.jpg')


for fname in images:
    img = cv2.imread(fname)
    grayR = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, cornersR = cv2.findChessboardCorners(grayR, (7,9),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpointsR.append(objp)

        cv2.cornerSubPix(grayR,cornersR,(11,11),(-1,-1),criteria)
        imgpointsR.append(cornersR)
        for i in cornersR:
          imgX.append(i[0][0])
          imgY.append(i[0][1])
        

cameraMatrix1 =None
cameraMatrix2 = None
distCoeffs1 = None
distCoeffs2 = None
R =None
T = None
E = None
F = None


# calibrate two cameras:
stereocalib_retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(objpointsL,
                                                                                                             imgpointsL,
                                                                                                             imgpointsR,
                                                                                                             cameraMatrix1,
                                                                                                             distCoeffs1,
                                                                                                             cameraMatrix2,
                                                                                                             distCoeffs2,
                                                                                                             (640,480),
                                                                                                             R,T, E, F,
                                                                                                             flags = 0,criteria = criteria)

R1 = None
R2 = None
P1 = None
P2 = None
# rectify:
R1, R2, P1, P2, Q, roi1,roi2 = cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, (640,480), R, T,
                 R1, R2, P1, P2, Q=None, flags=0, alpha=-1, newImageSize=(0, 0))


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def compare_SIFT(box1, img1, box2, img2):
  """code is in SIFTry, right now not working with live feed, so returning center of both boxes"""
  pass


def trangulate(point1, point2):  
  x1 = float(point1[0])
  y1 = float(point1[1])
  x2 = float(point2[0])
  y2 = float(point2[1])

  newMat = [  R[0][0]*(x2-T[0][0])+ R[0][1]*(y2-T[1][0])+ R[0][2]*(0-T[2][0]) ,
              R[1][0]*(x2-T[0][0])+ R[1][1]*(y2-T[1][0])+ R[1][2]*(0-T[2][0]) ,
              R[2][0]*(x2-T[0][0])+ R[2][1]*(y2-T[1][0])+ R[2][2]*(0-T[2][0])  ]
  
  equalTo = [ [x1,0,0-R[0][2]] ,
              [0,y1,0-R[1][2]] ,
              [0,0,1-R[2][2]] ]
  
  Pr = np.linalg.solve(equalTo,newMat)

  newMat2 = [ R[0][0]*(x1-T[0][0])+ R[0][1]*(y1-T[1][0])+ R[0][2]*(0-T[2][0]) ,
              R[1][0]*(x1-T[0][0])+ R[1][1]*(y1-T[1][0])+ R[1][2]*(0-T[2][0]) ,
              R[2][0]*(x1-T[0][0])+ R[2][1]*(y1-T[1][0])+ R[2][2]*(0-T[2][0])  ]
  
  equalTo2 = [ [x2,0,0-R[0][2]] ,
               [0,y2,0-R[1][2]] ,
               [0,0,1-R[2][2]] ]

  Pl = np.linalg.solve(equalTo2,newMat2)
  arr = np.array([])
  arr = cv2.triangulatePoints(P1, P2, np.array([Pr[0], Pr[1]]), np.array([Pl[0], Pl[1]]), arr)
  return (Pl, Pr)
  

print("Ready!")
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    while True:
      retL,imageL = cap1.read()
      retR,imageR = cap2.read()
      whiteIMG = cv2.resize(cv2.imread('backgroundIMG/white.jpg'), (int(screenX/2),screenY))

      # for Left camera:
      
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expandedL = np.expand_dims(imageL, axis=0)
      image_tensorL = detection_graph.get_tensor_by_name('image_tensor:0')
      
      # Each box represents a part of the image where a particular object was detected.
      boxesL = detection_graph.get_tensor_by_name('detection_boxes:0')
      
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scoresL = detection_graph.get_tensor_by_name('detection_scores:0')
      classesL = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detectionsL = detection_graph.get_tensor_by_name('num_detections:0')
      
      # Actual detection.
      (boxesL, scoresL, classesL, num_detectionsL) = sess.run([boxesL, scoresL, classesL, num_detectionsL],feed_dict={image_tensorL: image_np_expandedL})

      # Visualization of the results of a detection.
      boxL = vis_util.visualize_boxes_and_labels_on_image_array(imageL, np.squeeze(boxesL), np.squeeze(classesL).astype(np.int32),
          np.squeeze(scoresL),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)
      

      # for right camera:
      
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expandedR = np.expand_dims(imageR, axis=0)
      image_tensorR = detection_graph.get_tensor_by_name('image_tensor:0')
      
      # Each box represents a part of the image where a particular object was detected.
      boxesR = detection_graph.get_tensor_by_name('detection_boxes:0')
      
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scoresR = detection_graph.get_tensor_by_name('detection_scores:0')
      classesR = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detectionsR = detection_graph.get_tensor_by_name('num_detections:0')
      
      # Actual detection.
      (boxesR, scoresR, classesR, num_detectionsR) = sess.run([boxesR, scoresR, classesR, num_detectionsR],feed_dict={image_tensorR: image_np_expandedR})

      # Visualization of the results of a detection.
      boxR = vis_util.visualize_boxes_and_labels_on_image_array(imageR, np.squeeze(boxesR), np.squeeze(classesR).astype(np.int32),
          np.squeeze(scoresR),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)

      
      if boxL!=None and boxR!=None:
        # un-normalize box(integer values):
        im_widthL, im_heightL = Image.fromarray(np.uint8(imageL)).convert('RGB').size
        im_widthR, im_heightR = Image.fromarray(np.uint8(imageR)).convert('RGB').size
        
        yminL, xminL, ymaxL, xmaxL = boxL
        boxL = (xminL * im_widthL, xmaxL * im_widthL, yminL * im_heightL, ymaxL * im_heightL)
        yminL, xminL, ymaxL, xmaxL = boxL

        yminR, xminR, ymaxR, xmaxR = boxR
        boxR = (xminR * im_widthR, xmaxR * im_widthR, yminR * im_heightR, ymaxR * im_heightR)
        yminR, xminR, ymaxR, xmaxR = boxR
        
        #find same dots in both cameras:
        #just center of two boxes (problem with SIFT)
        #compare_SIFT(box1,image_np, box2, image_np2)
        centerL = ((yminL+xminL)/2, (ymaxL+xmaxL)/2)
        centerR = ((yminR+xminR)/2, (ymaxR+xmaxR)/2)
        
        #trangulate:
        Pl, Pr = trangulate(centerL, centerR)          

        cv2.circle(imageR, (int(centerR[0]), int(centerR[1])), 5, (0,0,255),10)
        cv2.circle(imageL, (int(centerL[0]), int(centerL[1])), 5, (0,0,255),10)
        
        cv2.imshow('right', cv2.resize(imageR, ((int(screenX/2),int(screenY/2 -50)))))
        cv2.imshow('left', cv2.resize(imageL, (int(screenX/2),int(screenY/2 -50))))

        tranX = int(centerR[0] + Pr[0]) -25
        tranY = int(centerR[1] + Pr[1]) -25 +120
        whiteIMG[ tranY:50+tranY , tranX:50+tranX] = droneIMG
        cv2.circle(whiteIMG, (320,360), 5, (0,0,0),10)
        cv2.line(whiteIMG, (320,360),(tranX+25,tranY+25), (0,0,255),5)
        whiteIMG = cv2.flip(whiteIMG,1)
        distirng = "distance: " + "{}".format(round(Pr[2]/300,2))
        cv2.putText(whiteIMG,distirng, (20,80), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
        cv2.imshow('world', whiteIMG)
        
      if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        print("Done!")
        break
