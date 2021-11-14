import cv2
import numpy as np
import time
import sys
import argparse
import os
import glob

class Camera:
    def __init__(self,width,height,calibrated=False):
        self.shape = (width,height)
        self.calibrated=calibrated
        self.per_point=0
        if (self.calibrated):
            self.camera_matrix = np.loadtxt(".\imagesnap\cameraMatrix.txt",dtype='f',delimiter=',')
            self.distrotion_matrix = np.array([np.loadtxt(".\imagesnap\cameraDistortion.txt",dtype='f',delimiter=',')])
            self.new_camera_matrix = np.array([np.loadtxt(".\imagesnap/newcameraMatrix.txt",dtype='f',delimiter=',')])
            
            

    def _applyCalibate(self,imgNotGood):
        img = imgNotGood
        h,  w = img.shape[:2]
        
        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(self.camera_matrix,self.distrotion_matrix,(w,h),1,(w,h))

        # undistort
        mapx,mapy = cv2.initUndistortRectifyMap(self.camera_matrix,self.distrotion_matrix,None,newcameramtx,(w,h),5)
        dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
        # cv2.imshow("uncrop image",dst)
        # crop the image
        x,y,w,h = roi
        dst = dst[y:y+h, x:x+w]
        
        return dst 
    def snap(self,calib=False,name="snapshot", folder="."):        
        camera_index = 0
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if self.shape[0] > 0 and self.shape[1] > 0:
            print("Setting the custom Width and Height")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.shape[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.shape[1])
            self.cap.set(cv2.CAP_PROP_FPS,200)
        try:
            if not os.path.exists(folder):
                os.makedirs(folder)
                    # ----------- CREATE THE FOLDER -----------------
                folder = os.path.dirname(folder)
                try:
                    os.stat(folder)
                except:
                    os.mkdir(folder)
        except:
            pass
        nSnap   = 0
        w       = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h       = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        fileName    = "%s/%s_%d_%d_" %(folder, name, w, h)
        while True:
            ret, frame = self.cap.read()
            cv2.imshow('camera', frame)
            key = cv2.waitKey(1) & 0xFF
            if(calib):
                if key == ord('q'):
                    break
                if key == ord(' '):
                    print("Saving image ", nSnap)
                    cv2.imwrite("%s%d.jpg"%(fileName, nSnap), frame)
                    nSnap += 1
            else:
                if key == ord(' '):
                    print('snapp!')
                    self.img = frame
                    cv2.imshow('snap_image',self.img)
                    break
        self.cap.release()
        cv2.destroyAllWindows()

    def calibrate(self):
        #---------------------- SET THE PARAMETERS
        nRows = 9
        nCols = 7
        dimension = 20 #- mm
        workingFolder   = ".\imagesnap"
        imageType       = 'jpg'
        #------------------------------------------

        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, dimension, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((nRows*nCols,3), np.float32)
        objp[:,:2] = np.mgrid[0:nCols,0:nRows].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.       
        # Find the images files
        filename    = workingFolder + "/*." + imageType
        images      = glob.glob(filename)

        print(len(images))
        if len(images) < 9:
            print("Not enough images were found: at least 9 shall be provided!!!")
            sys.exit()
        else:
            nPatternFound = 0
            imgNotGood = images[1]

            for fname in images:
                if 'calibresult' in fname: continue
                #-- Read the file and convert in greyscale
                img     = cv2.imread(fname)
                gray    = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

                print("Reading image ", fname)

                # Find the chess board corners
                ret, corners = cv2.findChessboardCorners(gray, (nCols,nRows),None)

                # If found, add object points, image points (after refining them)
                if ret == True:
                    print("Pattern found! Press ESC to skip or ENTER to accept")
                    #--- Sometimes, Harris cornes fails with crappy pictures, so
                    corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

                    # Draw and display the corners
                    cv2.drawChessboardCorners(img, (nCols,nRows), corners2,ret)
                    cv2.imshow('img',img)
                    # cv2.waitKey(0)
                    k = cv2.waitKey(0) & 0xFF
                    if k == 27: #-- ESC Button
                        print("Image Skipped")
                        imgNotGood = fname
                        continue

                    print("Image accepted")
                    nPatternFound += 1
                    objpoints.append(objp)
                    imgpoints.append(corners2)

                    # cv2.waitKey(0)
                else:
                    imgNotGood = fname
        cv2.destroyAllWindows()
        if (nPatternFound > 1):
            print("Found %d good images" % (nPatternFound))
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
            img = cv2.imread(imgNotGood)
            h,  w = img.shape[:2]
            print("Image to undistort: ", imgNotGood)
            newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

            # undistort
            mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
            dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
            
            print("Calibrated picture saved as calibresult.png")
            print("Calibration Matrix: ")
            print(mtx)
            print("Disortion: ", dist)

            #--------- Save result
            filename = workingFolder + "/cameraMatrix.txt"
            np.savetxt(filename, mtx, delimiter=',')
            filename = workingFolder + "/cameraDistortion.txt"
            np.savetxt(filename, dist, delimiter=',')
            filename = workingFolder + "/newcameraMatrix.txt"
            np.savetxt(filename, newcameramtx, delimiter=',')

            mean_error = 0
            for i in range(len(objpoints)):
                imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
                error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
                mean_error += error

            print("total error: ", mean_error/len(objpoints))

        else:
            print("In order to calibrate you need at least 9 good pictures... try again")  
    def appplyPerspective(self,frame,pts=((0,0),(0,0),(0,0),(0,0)),size=(0,0)):
        cv2.circle(frame, pts[0], 5, (0, 0, 255), -1)
        cv2.circle(frame, pts[1], 5, (0, 0, 255), -1)
        
        cv2.circle(frame,pts[2], 5, (0, 0, 255), -1)
        cv2.circle(frame, pts[3], 5, (0, 0, 255), -1)
        pts1 = np.float32([list(pts[0]), list(pts[1]), list(pts[2]), list(pts[3])])
        pts2 = np.float32([[0, 0], [size[0], 0], [0, size[1]], [size[0], size[1]]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        result = cv2.warpPerspective(frame, matrix, size)
        return result
    def open(self,camera_index = 0):
        camera_index = camera_index
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.shape[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.shape[1])
        self.cap.set(cv2.CAP_PROP_FPS,60)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        # self.cap.set(cv2.CAP_PROP_EXPOSURE,-7.5)
        # self.cap.set(cv2.CAP_PROP_EXPOSURE,-5.5)
        print("Open Camera with chanel : ",camera_index,"\nResolution ",self.shape[0]," x ",self.shape[1],"\nFPS 60\nExposure ",self.cap.get(cv2.CAP_PROP_EXPOSURE))
        return self.cap
    def show(self,name,img):
        cv2.imshow(str(name),img)
    
    def createcardspacetrackbar(self):
        def nothing(x):
            pass
        cv2.namedWindow('track')
        f = open("./imagesnap/"+"per_point"+".txt","r")
        msg = list(map(int,f.readline().split(',')))
        blank_image = np.zeros(shape=[601,601, 3], dtype=np.uint8)
        cv2.createTrackbar('x1', 'track',msg[0],self.shape[0],nothing)
        cv2.createTrackbar('x2', 'track',msg[1],self.shape[0],nothing)
        cv2.createTrackbar('x3', 'track',msg[2],self.shape[0],nothing)
        cv2.createTrackbar('x4', 'track',msg[3],self.shape[0],nothing)
        cv2.createTrackbar('y1', 'track',msg[4],self.shape[1],nothing)
        cv2.createTrackbar('y2', 'track',msg[5],self.shape[1],nothing)
        cv2.createTrackbar('y3', 'track',msg[6],self.shape[1],nothing)
        cv2.createTrackbar('y4', 'track',msg[7],self.shape[1],nothing)
        # return blank_image
        cv2.imshow('track',blank_image)
    def readcardspacetrackerbar(self):
        x1 = cv2.getTrackbarPos('x1','track')
        x2 = cv2.getTrackbarPos('x2','track')
        x3 = cv2.getTrackbarPos('x3','track')
        x4 = cv2.getTrackbarPos('x4','track')
        y1 = cv2.getTrackbarPos('y1','track')
        y2 = cv2.getTrackbarPos('y2','track')
        y3 = cv2.getTrackbarPos('y3','track')
        y4 = cv2.getTrackbarPos('y4','track')
        
        return x1,x2,x3,x4,y1,y2,y3,y4
    
        

