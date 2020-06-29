import sys
import os
import cv2
#import math
import numpy as np
import numpy.linalg as la
import pandas as pd
import matplotlib.pyplot as plt
#from sklearn.cluster import KMeans
#from sklearn.cluster import DBSCAN
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from cvt.TrAQ.Trial import Trial
#from cvt.Analysis.Math import angle_diff
from cvt.utils import *


class CVTracer:
    
    def __init__(self, trial, n_pixel_blur = 3, block_size = 15, 
                 threshold_offset = 13, min_area = 20, max_area = 400, 
                 max_frame2frame_distance = None, significant_displacement = None,
                 adaptiveMethod = 'gaussian', threshType = 'inv', RGB = False, 
                 MOG2 = False, MOG_history = 1000, MOG_varThreshold = 25,
                 MOG_initial_skip = 10, MOG_learning_rate = 0.01,
                 live_preview = False, GPU = False ):

        self.trial          = trial
        self.fvideo_in      = trial.video_file
        self.fvideo_ext     = "mp4"
        self.fvideo_out     = os.path.join(trial.output_dir, 'traced.mp4')
        trial.traced_file   = self.fvideo_out

        # initialize video playback details
        self.RGB            = RGB
        self.codec          = 'mp4v'
        if ( self.fvideo_ext == ".avi" ): self.codec = 'DIVX' 
        self.online_viewer  = live_preview
        self.online_window  = 'CVTracer live tracking'
        self.GPU            = GPU

        # initialize openCV video capture
        self.cap            = None
        self.frame          = None
        self.frame_num      = -1
        self.frame_start    = trial.frame_start
        self.frame_end      = trial.frame_end
        self.fps            = trial.fps
        self.init_video_capture()
        
        self.MOG_history    = MOG_history
        self.MOG_varThreshold = MOG_varThreshold
        self.MOG_initial_skip = MOG_initial_skip
        self.MOG_learning_rate = MOG_learning_rate
        if MOG2:
            self.MOG2 = True
            print("Using MOG2 Background Subtraction")
            self.init_background_subtractor()

        # initialize contour working variables
        self.n_pix_avg      = n_pixel_blur
        self.thresh         = []
        self.block_size     = block_size
        self.offset         = threshold_offset
        self.threshMax      = 100
        if adaptiveMethod == 'gaussian':
            print("Using Gaussian Adaptive Threshold")
            self.adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        else:
            print("Using Mean Adaptive Threshold with [%i, 0]" % self.threshMax)
            self.adaptiveMethod = cv2.ADAPTIVE_THRESH_MEAN_C
        if threshType == 'inv':
            print("Using Inverted Binary Threshold with [0, %i]." % self.threshMax)
            self.threshType     = cv2.THRESH_BINARY_INV
        else:
            print("Using Non-Inverted Binary Threshold")
            self.threshType     = cv2.THRESH_BINARY
        self.min_area       = max(min_area,1)
        self.max_area       = max_area
        self.ideal_area     = (min_area+max_area)/2
        self.max_frame2frame_distance = max_frame2frame_distance
        if self.max_frame2frame_distance == None:
            # Maximum distance a fish can reasonably travel over a single frame.
            # sqrt(max_area) works as a rough estimate for the body length.
            self.max_frame2frame_distance = np.sqrt(self.max_area)*2
        self.significant_displacement = significant_displacement
        if significant_displacement == None:
            self.significant_displacement = np.sqrt(self.max_area)*0.2
    
        self.contours       = []
        self.contour_repeat = []
        self.moments        = []
        self.df             = self.trial.df
        
        self.colors = color_list     
        

    ############################
    # cv2.VideoCapture functions
    ############################


    def init_video_capture(self):
        self.cap = cv2.VideoCapture(self.fvideo_in)
        if self.cap.isOpened() == False:
            sys.exit('Video file cannot be read! Please check input_vidpath ' 
                     + 'to ensure it is correctly pointing to the video file. \n %s' % self.fvideo_in)
             
        # Video writer class to output video with contour and centroid of tracked
        # object(s) make sure the frame size matches size of array 'final'
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        ret, self.frame = self.cap.read()
        self.frameSize = self.frame.shape[1], self.frame.shape[0]
        self.out = cv2.VideoWriter( filename = self.fvideo_out, fourcc = fourcc, 
                                    fps = self.fps, frameSize = self.frameSize, 
                                    isColor = self.RGB )
        self.frame_num = self.frame_start
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_num)
        if self.frame_end < 0:
            self.frame_end = self.n_frames()


    def release(self):
        # release the capture
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        
        sys.stdout.write("\n")
        sys.stdout.write("       Video capture released.\n")
        sys.stdout.flush()


    def tstamp(self):
        return float(self.frame_num)/self.fps


    def n_frames(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1

    
    def tracked_frames(self):
        return self.frame_num - self.frame_start

    def set_frame(self, i):
        self.frame_num = i
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)

    def get_frame(self):
        if self.cap.isOpened():
            ret, self.frame = self.cap.read()
            if ret == True:
                self.frame_num += 1
                if self.GPU:
                    self.frame = cv2.UMat(self.frame)
                return True
        return False


    def init_background_subtractor(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                                    history=self.MOG_history, 
                                    varThreshold=self.MOG_varThreshold,
                                    detectShadows=False)
        for i in range(0,self.MOG_history,self.MOG_initial_skip):
            self.set_frame(self.frame_start + self.MOG_history - i)
            self.get_frame()
            self.print_current_frame()
            self.mask_tank()
            self.bg_subtractor.apply(self.frame,learningRate=self.MOG_learning_rate)
    
    
    def init_live_preview(self):
        if (self.online_viewer):
            create_named_window(self.online_window)
    

    def write_frame(self):
        if self.GPU:
            self.frame = cv2.UMat.get(self.frame)
        return self.out.write(self.frame)

        
    def post_frame(self,delay=None):
        if delay==None or delay<1:
            delay = int(1000/self.fps)
        if ( self.online_viewer ):
            wname = self.online_window
            cv2.imshow(wname,self.frame)
            k = wait_on_named_window(wname,delay)
            if k==-2:
                return 0
            if k==space_key:
                while True:
                    k2 = wait_on_named_window(wname,delay)
                    if k2==-2:
                        return 0
                    if k2==space_key:
                        break
        return 1
 

    def print_current_frame(self):
        t_csc = int(self.frame_num/self.fps * 100)
        t_sec = int(t_csc/100) 
        t_min = t_sec/60
        t_hor = t_min/60
        sys.stdout.write("       Current tracking time: %02i:%02i:%02i:%02i \r" 
                         % (t_hor, t_min % 60, t_sec % 60, t_csc % 100) )
        sys.stdout.flush()
        

    ############################
    # Contour functions
    ############################


    def threshold_detect(self, hist = False):
        # blur and current image for smoother contours
        blur = cv2.GaussianBlur(self.frame, (self.n_pix_avg, self.n_pix_avg), 0)
       
        # convert to grayscale
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

        if hist:
            plt.title("Raw image grayscale histogram")
            plt.hist(self.frame.ravel()[self.frame.ravel() > 0],256)
            plt.show()
        
            plt.title("Blurred grayscale histogram")
            plt.hist(blur.ravel()[blur.ravel() > 0],256)
            plt.show()
        
            plt.title("Grayscale histogram")
            plt.hist(gray.ravel()[gray.ravel() > 0],256)
            plt.show()
    
        # calculate adaptive threshold using cv2
        #   more info:
        #       https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html
        self.thresh = cv2.adaptiveThreshold( gray, 
                                             maxValue = self.threshMax, 
                                             adaptiveMethod = self.adaptiveMethod,
                                             thresholdType = self.threshType,
                                             blockSize = self.block_size, 
                                             C = self.offset )
    
    
    def detect_contours(self):
        self.threshold_detect()
        # The [-2:] makes this work with openCV versions 3 and 4.
        self.contours, hierarchy = cv2.findContours( self.thresh, 
                                                     cv2.RETR_TREE, 
                                                     cv2.CHAIN_APPROX_SIMPLE )[-2:]

        # Dismiss contours with out-of-bounds area.
        self.contours = [ c for c in self.contours if 
                          self.min_area<=cv2.contourArea(c)<=self.max_area ]
        
        # Compute contour moments.
        self.moments  = [ cv2.moments(c) for c in self.contours ]
        
        # Compute tentative new coordinates.
        # If there aren't enough contours, fill with NaN.
        n = max(len(self.contours),self.trial.n_ind)
        self.new = np.empty((n,4),dtype=float)
        self.new.fill(np.nan)
        for i,M in enumerate(self.moments):
            x     = M['m10'] / M['m00']
            y     = M['m01'] / M['m00']
            theta = 0.5 * np.arctan2(2*M['mu11'], M['mu20']-M['mu02'])
            area  = M['m00']
            
#            # This is a bit more costly but provides the semi-axes as well.
#            mu = np.array([[M['mu20'],M['mu11']],[M['mu11'],M['mu02']]])/M['m00']
#            eVal,eVec = la.eigh(mu)
#            theta = np.arctan2(eVec[1,1],eVec[1,0])
            
            self.new[i] = [x, y, theta, area]
        

    #############################
    # Frame-to-frame functions
    #############################


    # this is the main algorithm the tracer follows when trying 
    # to associate individuals identified in this frame with 
    # those identified in the previous frame/s
    def connect_frames(self):
        
        # If there is no previous frame, use the n_ind best contours.
        # If there aren't enough contours, fill with NaN.
        if len(self.df) == 0:
            if len(self.new)>self.trial.n_ind:
                # If there are too many contours, reorder by "ideality" (for now 
                # closeness to ideal area) then pick the first n_ind.
                # TODO: Take another look at the cluster algorithm approach used 
                # in cvtracer. What situation is it meant to address? Is that idea 
                # that some fish may be broken into multiple contours?
                d = np.absolute(self.new[:,3]-self.ideal_area) # distance ideal contour area
                self.new = self.new[np.argsort(d)][:self.trial.n_ind]
        else:
            # If there is a valid previous frame, first set missing positions aside,
            # then solve assignment problem on remaining positions.
            # TODO: Bring back prediction of positions based on past positions.
            old     = self.df.iloc[-1].values.reshape((self.trial.n_ind,4))
            xy_old  = old[:,:2]
            xy_new  = self.new[:,:2]
            d       = cdist(xy_old,xy_new)
            # Putting a high cost on connections involving a missing position (NaN)
            # effectively makes the algorithm match all valid contours first.
            # This likely causes problems when multiple fish are lost.
            # Using the last known position would probably be better.
            d[np.isnan(d)] = 1e8
            Io,In = linear_sum_assignment(d)
            self.new = self.new[In]
            
            # TODO: When a fish is missing look for its last known position. 
            # Even better: use last few known positions to predict current position.
            
            # TODO: Identify cases in which a disappeared fish likely
            # merged contours with another fish (e.g. detect likely occlusions).
            # This will involve looking for area changes and making sure it doesn't
            # involve a jump longer than max_frame2frame_distance.
        
        
            # Fix spurious orientation reversals:
            # 1. Look for continuity with previous frame.
            dtheta  = self.new[:,2]-old[:,2]
            I       = ~np.isnan(dtheta)
            self.new[I,2] -= np.pi*np.rint(dtheta[I]/np.pi)
            # 2. Align direction with motion unless it's too slow.
            dxy     = self.new[:,:2]-old[:,:2]
            dot     = dxy[:,0]*np.cos(self.new[:,2])+dxy[:,1]*np.sin(self.new[:,2])
            I       = (dot<-self.significant_displacement)
            self.new[I,2] += np.pi
            # TODO: Measure the orientation directly from the contour, probably by 
            # looking at the sign of the third moment along the unsigned orientation.
        
        
        self.df.loc[self.frame_num] = self.new.flatten()
    
        
    ############################
    # Masking functions
    ############################
    
        
    def mask_tank(self):
        row_c = int(self.trial.tank.row_c) + 1
        col_c = int(self.trial.tank.col_c) + 1
        R     = int(self.trial.tank.r) + 1
        if self.GPU:
            tank_mask = cv2.UMat(np.zeros_like(self.frame))
        else:
            tank_mask = np.zeros_like(self.frame)
        
        cv2.circle(tank_mask, (col_c,row_c), R, (255, 255, 255), thickness=-1)
        self.frame = cv2.bitwise_and(self.frame, tank_mask)


    def mask_background(self):
        self.fgmask = self.bg_subtractor.apply(self.frame)
#        fgmask_bin = self.fgmask > 1
#        frame_nobg = np.full_like(self.frame,255) 
#        frame_nobg[fgmask_bin] = self.frame[fgmask_bin]
#        self.frame = frame_nobg
        self.frame[self.fgmask<=1] = 255
        return


    def detect_clusters(self, eps=2):
        points = np.where(self.frame < 170)
        points = np.stack(points, axis=-1)
        points = np.float32(points)
        fish = []
        self.clust_center = []
        self.clust_count = []
        self.clust_points = []

        if len(points) > self.trial.group.n:
            db = DBSCAN(eps=eps, min_samples=self.trial.group.n).fit(points)
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            labels = db.labels_
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise_ = list(labels).count(-1) 
            print("labels.shape ", labels.shape)
            unique_labels = set(labels)
            colors = [plt.cm.Spectral(each) for each in np.linspace(0,1, len(unique_labels))]
            if n_clusters_ > 0:
                print("n_cluster, n_noise = ", n_clusters_, n_noise_)
            for k, col in zip(unique_labels, colors):
                class_member_mask = (labels == k)
                xy = points[class_member_mask & core_samples_mask]
                self.clust_points.append(np.array(xy))

        # test found contours against area constraints
        i = 0
        while i < len(self.clust_points):
            area = len(self.clust_points[i])
            if area < self.min_area or area > self.max_area:
                del self.clust_points[i]
            else:
                i += 1
        self.clust_points = np.array(self.clust_points)


    def analyze_clusters(self):
        self.coord_pre = self.coord_now.copy()
        self.coord_now = []
        for i in range(len(self.clust_count)):
            # test found contours against area constraints
            M = cv2.moments(self.clust_points[i])
            if M['m00'] != 0:
                cx   = M['m10'] / M['m00']
                cy   = M['m01'] / M['m00']
                mu20 = M['m20'] / M['m00'] - pow(cx,2)
                mu02 = M['m02'] / M['m00'] - pow(cy,2)
                mu11 = M['m11'] / M['m00'] - cx*cy
            else:
                cx = 0
                cy = 0
            ry = 2 * mu11
            rx = mu20 - mu02
            theta = 0.5 * np.arctan2(ry, rx)
            self.coord_now.append([cx, cy, theta])


    def mask_contours(self):
        self.contour_masks = []
        # add contour areas to mask
        for i in range(len(self.contours)):
            mask = np.zeros_like(self.frame, dtype=bool)
            cv2.drawContours(mask, self.contours, i, 1, -1)
            self.contour_masks.append(mask)


    ############################
    # Drawing functions
    ############################

    # Show the current frame. Press any key or click "close" button to exit.
    def show_current_frame(self):
        window_name = create_named_window('current frame')
        cv2.imshow(window_name,self.frame)
        wait_on_named_window(window_name)
        cv2.destroyAllWindows()
        return 1


    def draw(self, tank=True, repeat_contours=True, all_contours=False, 
             contour_color=(0,0,0), contour_thickness=1,
             points=True, directors=True, timestamp=True):
        if tank:
            self.draw_tank(self.trial.tank)
        if all_contours:
            self.draw_contours(contour_color, contour_thickness)
        elif len(self.contours) != self.trial.n_ind:
            self.draw_contour_repeat(contour_color, contour_thickness)
        if points:
            self.draw_points()
        if directors:
            self.draw_directors()
        if timestamp:
            self.draw_tstamp()


    def draw_tstamp(self):
        color = (0,0,0) if self.RGB else 0
        font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
        t_str = "%02i:%02i:%02i.%02i" % ( 
                int(self.frame_num / self.fps / 3600),
                int(self.frame_num / self.fps / 60 % 60),
                int(self.frame_num / self.fps % 60),
                int(self.frame_num % self.fps * 100 / self.fps) )
        cv2.putText(self.frame, t_str, (5,30), font, 1, color, 2)

    
    def draw_tank(self, tank):
        color = (0,0,0) if self.RGB else 0
        cv2.circle(self.frame, (int(tank.col_c), int(tank.row_c)), int(tank.r),
                   color, thickness=7)

    
    # Draw problematic contours.
    def draw_contour_repeat(self,contour_color=(0,255,0), contour_thickness=1):
        color = contour_color if self.RGB else 0
        for i in self.contours:
            cv2.drawContours(self.frame, self.contour_repeat, i, color, int(contour_thickness))


    # Draw every contour.
    def draw_contours(self,contour_color=(0,0,255), contour_thickness=1):
        color = contour_color if self.RGB else 0
        cv2.drawContours(self.frame, self.contours, -1, color, int(contour_thickness))


    # Draw center of each individual as a dot.
    def draw_points(self):
        XY = self.new[:,:2]
        for i,(x,y) in enumerate(XY):
            if not (np.isnan(x) or np.isnan(y)):
                x,y = int(x),int(y)
                color = self.colors[i%len(self.colors)] if self.RGB else 0
                cv2.circle(self.frame, (x,y), 3, color, -1, cv2.LINE_AA)


    # Draw orientation of each individual as an arrow.
    def draw_directors(self, arrow_size=20):
        XY = self.new[:,:2]
        Th = self.new[:,2]
        U  = arrow_size/2*np.array([np.cos(Th),np.sin(Th)]).T
        for i in range(self.trial.n_ind):
            if not np.isnan(XY[i]).any():
                color = self.colors[i%len(self.colors)] if self.RGB else 255
                (x1,y1),(x2,y2) = (XY[i]-U[i]).astype(int),(XY[i]+U[i]).astype(int)
                cv2.arrowedLine(self.frame, (x1,y1), (x2,y2), color=color, 
                                thickness=2, tipLength=0.3)


